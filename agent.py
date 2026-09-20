from __future__ import annotations
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import structlog
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionMessageToolCall
from tools import READ_ONLY_TOOLS, TOOL_FUNCTIONS, TOOLS

load_dotenv()

# 日志只显示 WARNING 以上，避免污染对话界面；想看调试信息改成 logging.INFO
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)
log = structlog.get_logger()
PROMPT_DIR = Path(__file__).parent / "prompts"


def load_system_prompt() -> str:
    """按文件名顺序拼接 prompts/ 下所有 .md，并注入当前日期。"""
    parts = [p.read_text(encoding="utf-8") for p in sorted(PROMPT_DIR.glob("*.md"))]
    prompt = "\n\n".join(parts)
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return prompt.replace("{date}", f"{now:%Y-%m-%d} 星期{weekday}")
MAX_TOOL_ROUNDS = 8
MAX_CLAIM_RETRY = 1 
# 回复里出现这些词 = 声称状态已改变。若本轮没调对应工具，就是"口头完成"。
# ① 用户是否表达了「改变状态」的意图。注意：这里宁可漏检，也不要误报。
INTENT_HINTS: tuple[str, ...] = (
    "标完成", "标记", "做完", "完成", "删掉", "删除", "去掉", "移除",
    "记住", "记下来", "保存", "存起来", "添加", "加个", "新增", "记录",
)

# ③ 「声称本轮执行成功」的严格说法。只收动作短语，不收「已完成」这种状态词。
CLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "complete_todo": ("已标记完成", "标记为完成", "已完成标记", "已标记为完成"),
    "add_todos": ("已保存到待办", "已添加到待办", "已为你保存", "已帮你添加", "已存入待办"),
    "delete_todo": ("已删除待办", "已删掉该待办", "已移除待办"),
}

# ② 出现这些词 = 回复是在「回顾既有状态」或「承认没做」，不是声称本轮执行
NEGATIVE_HINTS: tuple[str, ...] = (
    "未执行", "并未", "没有调用", "未调用", "状态未变", "未发生",
    "未做任何", "虚假", "此前已存", "之前已", "原本就是", "未改动",
)


def has_change_intent(user_text: str) -> bool:
    """用户这句话是否要求改变待办状态。没有意图就不校验，避免误报。"""
    return any(h in user_text for h in INTENT_HINTS)


def check_claim(reply: str, called: set[str]) -> str | None:
    """检测「口头完成」：声称做了某动作，但本轮没调对应工具。

    判据刻意保守 —— 误报会把正确回复打成失败，代价远高于漏检。
    """
    if any(h in reply for h in NEGATIVE_HINTS):
        return None
    for tool, words in CLAIM_PATTERNS.items():
        if tool in called:
            continue
        for w in words:
            if w in reply:
                return f"回复中出现了「{w}」，但本轮未调用 {tool}，该状态实际上并未改变。"
    return None


class Agent:
    def __init__(self, model: str | None = None, system_prompt: str | None = None):
        self.model = model or os.environ["OPENAI_MODEL"]
        self.client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
        # Agent 的记忆就是这个列表。没有 system prompt 时它一开始是空的。
        self.messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})
    def _run_tool(self, call: ChatCompletionMessageToolCall) -> str:
        """执行单个工具调用。出错也要返回字符串，给模型自己纠正的机会。"""
        name = call.function.name
        func = TOOL_FUNCTIONS.get(name)
        if func is None:
            return f"错误：没有名为 {name} 的工具。"
        try:
            args = json.loads(call.function.arguments)
            log.info("tool_call", tool=name, args=args)
            return func(**args)
        except Exception as e:
            return f"工具 {name} 执行失败：{e}"
    def _execute_calls(self, calls: list[ChatCompletionMessageToolCall]) -> list[str]:
        """执行一批工具调用。

        只读 / 网络类并发执行省 I/O 等待，写库类串行执行避免 sqlite 锁竞争。
        返回结果顺序严格等于 calls 顺序 —— tool_call_id 的对应关系不能乱。
        """
        if len(calls) <= 1:
            return [self._run_tool(c) for c in calls]

        concurrent = [c for c in calls if c.function.name in READ_ONLY_TOOLS]
        sequential = [c for c in calls if c.function.name not in READ_ONLY_TOOLS]

        results: dict[str, str] = {}

        if concurrent:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(len(concurrent), 8)) as pool:
                # pool.map 保证输出顺序与输入一致
                for c, r in zip(concurrent, pool.map(self._run_tool, concurrent)):
                    results[c.id] = r
            print(
                f"  [并行] {len(concurrent)} 个只读工具并发执行，"
                f"耗时 {time.perf_counter() - t0:.2f}s"
            )

        for c in sequential:
            results[c.id] = self._run_tool(c)

        return [results[c.id] for c in calls]

    def _turn(self) -> tuple[str, set[str]]:
        """跑完一整轮「模型 ⇄ 工具」交互，返回 (最终回复, 本轮调用过的工具名)。"""
        called: set[str] = set()

        for _round in range(MAX_TOOL_ROUNDS):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=TOOLS,
                temperature=0.4,
            )
            msg = resp.choices[0].message

            if not msg.tool_calls:
                # 没有工具调用 = 说完了
                self.messages.append({"role": "assistant", "content": msg.content or ""})
                return msg.content or "", called

            # 只保留 function 类型的工具调用（custom tool 我们没启用）
            calls = [
                c
                for c in msg.tool_calls
                if isinstance(c, ChatCompletionMessageToolCall)
            ]
            if not calls:
                reply = msg.content or "（模型请求了不支持的工具类型）"
                self.messages.append({"role": "assistant", "content": reply})
                return reply, called

            # 关键 1：把「模型要求调工具」这条 assistant 消息写进历史
            self.messages.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )

            # 关键 2：执行工具，结果用 role="tool" 回灌，tool_call_id 必须一一对应
            results = self._execute_calls(calls)
            for c, result in zip(calls, results):
                called.add(c.function.name)
                print(f"  [工具] {c.function.name} -> {result}")
                self.messages.append(
                    {"role": "tool", "tool_call_id": c.id, "content": result}
                )



        return "（工具调用超过上限，已停止）", called

    def chat(self, user_text: str) -> str:
        """发一轮对话，必要时调用工具，并在检测到「口头完成」时要求模型重来。"""
        self.messages.append({"role": "user", "content": user_text})

        reply = ""
        warn: str | None = None

        for _attempt in range(MAX_CLAIM_RETRY + 1):
            reply, called = self._turn()
            warn = check_claim(reply, called) if has_change_intent(user_text) else None
            if not warn:
                break

            log.warning("claim_without_tool", attempt=_attempt, reply=reply)
            print(f"  [校验] {warn} → 已要求模型真正执行")
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{warn}这是不实回复。请立即调用对应工具真正执行；"
                        f"如果用户其实没有要求执行，就如实说明你并未做任何改动。"
                    ),
                }
            )
        else:
            reply = f"[未通过校验] {warn or '回复与工具调用不一致'}\n\n{reply}"

        log.info("turn_done", history_len=len(self.messages))
        return reply



def main() -> None:
    system_prompt = load_system_prompt()
    agent = Agent(system_prompt=system_prompt)
    print(f"Agent 就绪（提示词 {len(system_prompt)} 字）。输入 exit 退出。\n")


    while True:
        try:
            text = input("你 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见")
            break

        if not text:
            continue
        if text.lower() in {"exit", "quit", "退出"}:
            break

        try:
            print(f"AI > {agent.chat(text)}\n")
        except Exception as e:
            log.error("chat_failed", error=str(e))
            print(f"[出错] {e}\n")

if __name__ == "__main__":
    main()