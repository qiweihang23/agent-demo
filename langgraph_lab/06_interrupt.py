from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_core.runnables import RunnableConfig

class State(TypedDict):
    draft: str       # 拟发送的消息内容
    status: str      # 最终状态

# 用一个全局计数器当"探针"，记录"真正发送"这个副作用发生了几次
SEND_COUNT = 0

def draft_node(state: State) -> dict:
    return {"draft": "会议改到下午3点", "status": "待审批"}

def send_node(state: State) -> dict:
    global SEND_COUNT
    print("【send_node 开始执行】")          # 恢复时你会看到这行打印两次！

    # 运行到这里暂停，把待审批内容抛给外部；恢复时本行直接返回人的决定
    decision = interrupt({
        "提示": "请审批以下飞书消息",
        "内容": state["draft"],
    })
    print("  收到审批决定 =", decision)

    # ★ 副作用放在 interrupt 之后：第一次暂停时根本走不到，恢复重跑时才真正执行一次
    if decision == "approved":
        SEND_COUNT += 1
        print("  → 真实调用飞书 API 发送（第", SEND_COUNT, "次）")
        return {"status": "已发送"}
    return {"status": "已取消"}

builder = StateGraph(State)
builder.add_node("draft", draft_node)
builder.add_node("send", send_node)
builder.add_edge(START, "draft")
builder.add_edge("draft", "send")
builder.add_edge("send", END)

graph = builder.compile(checkpointer=InMemorySaver())
config: RunnableConfig = {"configurable": {"thread_id": "approval-1"}}

# —— 第一段：跑到 interrupt 处会暂停并返回 ——
print("===== 第1次 invoke：发起 =====")
graph.invoke({"draft": "", "status": ""}, config)

snap = graph.get_state(config)
print("暂停后 next =", list(snap.next))
# 抛出的审批 payload 藏在 state.tasks 里
print("待审批信息 =", snap.tasks[0].interrupts[0].value)
print("此刻 SEND_COUNT =", SEND_COUNT, "（应为0：还没真发）\n")

# —— 第二段：人做决定后，用 Command(resume=...) 从断点恢复 ——
print("===== 第2次 invoke：人工同意，恢复 =====")
result = graph.invoke(Command(resume="approved"), config)
print("最终状态 =", result)
print("最终 SEND_COUNT =", SEND_COUNT, "（应为1：只发了一次）")