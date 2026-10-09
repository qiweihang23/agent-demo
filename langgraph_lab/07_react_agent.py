"""07_react_agent.py — 用 LangGraph 重写 ReAct Agent（v2：含口头完成校验）

把 agent.py 里手写的「模型 ⇄ 工具」循环替换成 LangGraph 图：
  START → call_model → (有 tool_calls?) → call_tools → call_model → ... → END
                               ↓ 无 tool_calls
                          check_claim → (有口头完成?) → call_model（注入纠错重试）
                               ↓ 无问题 / 超重试上限
                              END

复用项目已有：
  - tools.py 的 TOOLS（schema）+ TOOL_FUNCTIONS（实现）
  - agent.py 的 load_system_prompt + has_change_intent + check_claim + MAX_CLAIM_RETRY

新增：
  - ChatOpenAI + bind_tools 替代裸 OpenAI SDK
  - LangGraph 图 + SqliteSaver checkpointer 替代手写循环 + save_message
  - check_claim 节点：检测「声称做了但没调工具」，命中则注入纠错消息重试
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Any, cast
from typing_extensions import TypedDict
from dotenv import load_dotenv
from pydantic import SecretStr

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

# 把项目根目录加到 sys.path，复用已有模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import TOOLS, TOOL_FUNCTIONS, set_current_session, set_current_user
from agent import (
    load_system_prompt,
    has_change_intent,
    check_claim,
    MAX_CLAIM_RETRY,
)

load_dotenv()

# ── LLM：DashScope 兼容 OpenAI 接口，bind_tools 直接吃 TOOLS schema ──
# 注：TOOLS 的元素是 OpenAI SDK 的 ChatCompletionToolParam（TypedDict），
# 运行时就是 dict，但静态类型和 bind_tools 的 dict[str, Any] 不匹配，用 cast 消除误报
llm = ChatOpenAI(
    model=os.environ["OPENAI_MODEL"],
    api_key=SecretStr(os.environ["DASHSCOPE_API_KEY"]),
    base_url=os.environ["OPENAI_BASE_URL"],
    temperature=0.4,
).bind_tools(cast(Any, TOOLS))


class State(TypedDict):
    # add_messages reducer：节点返回的新消息自动追加，不覆盖
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    user_id: str
    user_input: str        # 用户原始输入，供 check_claim 判断是否有改变意图
    claim_retries: int     # 本轮 claim 校验已重试次数


def call_model(state: State) -> dict:
    """调 LLM，返回 AIMessage（可能带 tool_calls）。

    system prompt 每次动态拼到消息列表头部、不入 state ——
    和 agent.py 里 self.messages[0] = load_system_prompt() 的做法一致。
    """
    set_current_session(state["session_id"])
    set_current_user(state["user_id"])

    sys_prompt = load_system_prompt(state["session_id"])
    messages = [SystemMessage(content=sys_prompt)] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


def call_tools(state: State) -> dict:
    """执行 AIMessage 里的 tool_calls，返回 ToolMessage 列表。

    直接复用 TOOL_FUNCTIONS 字典查表执行 ——
    和 agent.py 里 _execute_calls 的查表逻辑一致。
    """
    set_current_session(state["session_id"])
    set_current_user(state["user_id"])

    last_msg = state["messages"][-1]
    assert isinstance(last_msg, AIMessage), "call_tools 节点收到非 AIMessage"
    results: list[ToolMessage] = []
    for tc in last_msg.tool_calls:
        name = tc["name"]
        args = tc["args"]
        func = TOOL_FUNCTIONS.get(name)
        try:
            if func is None:
                result = f"错误：工具 {name} 不存在"
            else:
                result = func(**args)
        except Exception as e:
            result = f"工具 {name} 执行出错：{e}"
        results.append(
            ToolMessage(content=str(result), tool_call_id=tc["id"], name=name)
        )
    return {"messages": results}


def should_continue(state: State) -> str:
    """条件边：最后一条消息有 tool_calls → call_tools；否则 → check_claim"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "call_tools"
    return "check_claim"


def check_claim_node(state: State) -> dict:
    """检测口头完成：声称做了某动作但没调对应工具。

    和 agent.py 的 check_claim + has_change_intent 逻辑一致：
    1. 用户没改变意图 → 不校验，直接放行
    2. 回复含否定词（"并未""未调用"） → 不校验，放行
    3. 回复含声称词（"已记住""已保存到待办"）但没调对应工具 → 命中
    4. 命中且未超重试上限 → 注入纠错 HumanMessage，回到 call_model 重试
    5. 命中但超重试上限 → 放行（和 agent.py 的 else 分支一致）
    """
    messages = state["messages"]
    last_msg = messages[-1]
    assert isinstance(last_msg, AIMessage), "check_claim 收到非 AIMessage"

    reply = str(last_msg.content)

    # 用户没要求改变状态 → 不校验
    if not has_change_intent(state.get("user_input", "")):
        return {}

    # 从后往前扫到第一条 HumanMessage，收集本轮调用的工具名
    # （check_claim 注入的纠错消息也是 HumanMessage，扫到它就停，
    #  只收集它之后的 ToolMessage = 重试这一轮的工具调用）
    called: set[str] = set()
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage) and msg.name:
            called.add(msg.name)

    warn = check_claim(reply, called)
    if not warn:
        return {}

    retries = state.get("claim_retries", 0)
    if retries >= MAX_CLAIM_RETRY:
        print(f"  [校验] {warn} → 已达重试上限，放行")
        return {}

    print(f"  [校验] {warn} → 已要求模型真正执行")
    return {
        "messages": [HumanMessage(content=(
            f"{warn}这是不实回复。请立即调用对应工具真正执行；"
            f"如果用户其实没有要求执行，就如实说明你并未做任何改动。"
        ))],
        "claim_retries": retries + 1,
    }


def claim_route(state: State) -> str:
    """check_claim 后的路由：最后是 HumanMessage（纠错）→ call_model；否则 → END"""
    last_msg = state["messages"][-1]
    if isinstance(last_msg, HumanMessage):
        return "call_model"
    return END


# ── 组装图 ──
builder = StateGraph(State)
builder.add_node("call_model", call_model)
builder.add_node("call_tools", call_tools)
builder.add_node("check_claim", check_claim_node)
builder.add_edge(START, "call_model")
builder.add_conditional_edges("call_model", should_continue, ["call_tools", "check_claim"])
builder.add_edge("call_tools", "call_model")  # 工具结果回灌 → 再调模型
builder.add_conditional_edges("check_claim", claim_route, ["call_model", END])


def main():
    DB = "langgraph_lab/checkpoints.db"
    config: RunnableConfig = {
        "configurable": {"thread_id": "react-v2"},
        # 最坏情况：8轮工具 × 2步 + claim重试 × 3步 + 收尾 ≈ 25步
        "recursion_limit": 30,
    }

    print("=" * 60)
    print("07 ReAct Agent v2 — 真实 LLM + 真实工具 + 口头完成校验")
    print("输入 quit 退出 | 同一 thread_id 跨重启历史不丢")
    print("=" * 60 + "\n")

    with SqliteSaver.from_conn_string(DB) as saver:
        graph = builder.compile(checkpointer=saver)

        while True:
            try:
                user_input = input("你: ").strip()
            except EOFError:
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            # stream_mode="updates" 逐节点打印，能看到 ReAct 循环的每一步
            for event in graph.stream(
                {
                    "messages": [HumanMessage(content=user_input)],
                    "session_id": config["configurable"]["thread_id"],
                    "user_id": "default",
                    "user_input": user_input,
                    "claim_retries": 0,
                },
                config,
                stream_mode="updates",
            ):
                for node_name, update in event.items():
                    # 节点返回 {} 时 stream 给的 update 是 None，需要兜底
                    for msg in (update or {}).get("messages", []):
                        if isinstance(msg, AIMessage):
                            if msg.tool_calls:
                                for tc in msg.tool_calls:
                                    print(f"  [调用] {tc['name']}({tc['args']})")
                            else:
                                print(f"AI: {msg.content}")
                        elif isinstance(msg, ToolMessage):
                            preview = str(msg.content)[:120].replace("\n", " ")
                            print(f"  [结果] {preview}")
                        # check_claim 注入的纠错消息是 HumanMessage，不打印（避免和用户输入混淆）
            print()


if __name__ == "__main__":
    main()
