from __future__ import annotations
from tools import archive_messages
from tools import remember
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import sys
from collections.abc import Iterator
from typing import Any, cast
import structlog
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat import ChatCompletionMessageToolCall
from tools import (
    READ_ONLY_TOOLS, TOOL_FUNCTIONS, TOOLS, list_documents, list_memories,clear_messages,
    save_message, load_recent_messages, clear_session, clear_user, get_summary, save_summary,archive_messages,set_current_session, set_current_user,
)
from tracing import start_trace, trace_span, preview
from mcp_client import merged_tools, close_shared_hub, external_tools_prompt

load_dotenv()

# 日志只显示 WARNING 以上，避免污染对话界面；想看调试信息改成 logging.INFO
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING)
)
log = structlog.get_logger()
VERBOSE = False  # 运行时调试信息（历史加载/token/压缩/校验重试）开关；事件流里已有关键信息
PROMPT_DIR = Path(__file__).parent / "prompts"

def _format_summary(fine: str, coarse: str) -> str:
    """把两层摘要拼成注入 system prompt 的字符串。粗摘要在前（核心事实），细摘要在后（近期要点）。"""
    parts = []
    if coarse:
        parts.append(f"【核心事实】\n{coarse}")
    if fine:
        parts.append(f"【近期要点】\n{fine}")
    return "\n\n".join(parts) if parts else ""
def load_system_prompt(session_id: str) -> str:
    """按文件名顺序拼接 prompts/ 下所有 .md，并注入当前日期与知识库清单。"""
    parts = [p.read_text(encoding="utf-8") for p in sorted(PROMPT_DIR.glob("*.md"))]
    prompt = "\n\n".join(parts)
    now = datetime.now()
    weekday = "一二三四五六日"[now.weekday()]
    return (
        prompt.replace("{date}", f"{now:%Y-%m-%d} 星期{weekday}")
        .replace("{documents}", list_documents())
        .replace("{memories}", list_memories())
        .replace("{conversation_summary}", _format_summary(*get_summary(session_id)))
    )
MAX_TOOL_ROUNDS = 8
MAX_CLAIM_RETRY = 1 
CTX_SOFT_LIMIT = 6000      # prompt token 超过这个值，就把早期对话压成摘要
HISTORY_LOAD_LIMIT = 40    # 启动时最多加载多少条历史消息
KEEP_RECENT = 12           # 每次压缩后保留最近多少条原文
FINE_SUMMARY_MAX_LEN = 600   # 细摘要超过这个字数，就把旧细摘要升级进粗摘要
# 回复里出现这些词 = 声称状态已改变。若本轮没调对应工具，就是"口头完成"。
# ① 用户是否表达了「改变状态」的意图。注意：这里宁可漏检，也不要误报。
INTENT_HINTS: tuple[str, ...] = (
    "标完成", "标记", "做完", "完成", "删掉", "删除", "去掉", "移除",
    "记住", "记下来", "保存", "存起来", "添加", "加个", "新增", "记录",
)

# ③ 「声称本轮执行成功」的严格说法。只收动作短语，不收「已完成」这种状态词。
CLAIM_PATTERNS: dict[str, tuple[str, ...]] = {
    "complete_todo": ("已标记完成", "标记为完成", "已完成标记", "已标记为完成"),
    # "已为你保存" 不带"待办"，会和 remember 的回复撞车，收紧为必须带"待办"
    "add_todos": ("已保存到待办", "已添加到待办", "已帮你添加到待办", "已存入待办"),
    "delete_todo": ("已删除待办", "已删掉该待办", "已移除待办"),
    "remember": ("已记住", "已记下", "已保存到记忆", "已为你记下", "已更新记忆"),
    "forget": ("已忘记", "已删除记忆", "已从记忆中删除"),
}

# ② 出现这些词 = 回复是在「回顾既有状态」或「承认没做」，不是声称本轮执行
NEGATIVE_HINTS: tuple[str, ...] = (
    "未执行", "并未", "没有调用", "未调用", "状态未变", "未发生",
    "未做任何", "虚假", "此前已存", "之前已", "原本就是", "未改动",
)
EXTRACT_PROMPT = """你在为用户维护一份长期记忆库。扫一遍下面这段对话，把**值得跨会话长期保留的事实**提取出来。

只提取：
- 用户的身份、职业、所在地、家庭情况
- 稳定的偏好、习惯、长期项目背景
- 用户明确要求记住的事

一律不要提取：
- 一次性任务（查天气、读某个文件、算某笔账）
- documents 里能检索到的事实（文档内容不属于用户记忆）
- 你的推测、推断、归类
- 敏感信息（密码、卡号、证件号）
- 模糊不确定的话

要求：
- 每条一句话、以"用户"开头，例如「用户在杭州工作」
- 已在旧记忆里的事实不要重复提取
- 没有值得记的就返回空数组 []
- 只输出一个 JSON 字符串数组，不要任何解释。例如：["用户在杭州工作","用户偏好简洁回答"]

旧记忆（避免重复提取）：
{memories}

本轮对话：
{dialog}"""

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
def sanitize_history(messages: list[Any]) -> list[Any]:
    """保证 tool 消息紧跟在带 tool_calls 的 assistant 之后。

    按条截断历史时，很容易把 assistant(tool_calls) 切掉、留下孤儿 tool 消息 ——
    这种上下文发给 API 会被直接拒绝。这里把孤儿丢掉。
    """
    out: list[Any] = []
    pending = 0
    for m in messages:
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            out.append(m)
            pending = len(m["tool_calls"])
        elif role == "tool":
            if pending > 0:
                out.append(m)
                pending -= 1
        else:
            pending = 0
            out.append(m)

    # 末尾还欠着 tool 结果 = 最后一组被截断了，整组丢掉
    if pending > 0:
        while out and out[-1].get("role") == "tool":
            out.pop()
        if out and out[-1].get("role") == "assistant":
            out.pop()
    return out


def safe_split_index(messages: list[Any], want_cut: int) -> int:
    """返回 >= want_cut 的第一个「安全切分点」（该位置之前 tool 配对完整）。"""
    pending = 0
    safe: list[int] = []
    for i, m in enumerate(messages):
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            pending = len(m["tool_calls"])
        elif role == "tool":
            pending = max(0, pending - 1)
        if pending == 0:
            safe.append(i + 1)
    for s in safe:
        if s >= want_cut:
            return s
    return safe[-1] if safe else 0


class Agent:
    def __init__(
        self,
        model: str | None = None,
        system_prompt: str | None = None,
        session_id: str = "default",
        user_id: str = "default",
        tools: list | None = None,
        tool_functions: dict | None = None,
        reload_system_prompt: bool = True,
        enable_claim_check: bool = True,
        persist_messages: bool = True,
    ):
        self.model = model or os.environ["OPENAI_MODEL"]
        self.client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
        self.session_id = session_id
        self.user_id = user_id
        self.last_prompt_tokens = 0
        # 工具可注入：默认用全局 TOOLS/TOOL_FUNCTIONS（普通 agent 路径）。
        # worker 传自己的子集实现专精；supervisor 传 handoff 工具实现分发。
        self.tools = tools if tools is not None else TOOLS
        self.tool_functions = (
            tool_functions if tool_functions is not None else TOOL_FUNCTIONS
        )
        self.reload_system_prompt = reload_system_prompt
        self.enable_claim_check = enable_claim_check
        # worker 不落库：业务工具作用到主会话数据，但自己的对话不持久化、不读历史。
        self.persist_messages = persist_messages

        # Agent 的记忆就是这个列表。没有 system prompt 时它一开始是空的。
        self.messages: list[ChatCompletionMessageParam] = []
        if system_prompt:
            self.messages.append({"role": "system", "content": system_prompt})

        # 把上次聊的接上：历史按条存，取最近 N 条，再做配对清洗
        # 不落库的 agent（worker）也不读历史 —— 临时专精角色，每轮独立
        if self.persist_messages:
            history = sanitize_history(
                load_recent_messages(self.session_id, HISTORY_LOAD_LIMIT)
            )
            if history:
                self.messages.extend(history)
                if VERBOSE:
                    print(f"  [历史] 已加载 {len(history)} 条历史消息")

    #执行工具，输入工具名和参数，把json参数解析python字典再转为关键字参数，传入工具，输出工具执行结果
    def _run_tool(
        self,
        call: ChatCompletionMessageToolCall,
        tr: Any = None,
        parent_id: int | None = None,
    ) -> str:
        """执行单个工具调用。出错也要返回字符串，给模型自己纠正的机会。

        tr/parent_id 只服务于链路追踪，显式传入——线程池工作线程里没有
        隐式上下文可以依赖。
        """
        name = call.function.name
        func = self.tool_functions.get(name)
        if func is None:
            return f"错误：没有名为 {name} 的工具。"
        try:
            args = json.loads(call.function.arguments)
        except json.JSONDecodeError as e:
            return f"工具 {name} 参数不是合法 JSON：{e}"
        log.info("tool_call", tool=name, args=args)
        # span 里只留参数/结果短预览，完整内容仍照常回灌给模型
        with trace_span(tr, f"tool:{name}", parent_id, args=preview(args)) as sp:
            try:
                result = func(**args)
                sp.set_attr("result", preview(result))
                sp.set_attr("result_chars", len(result))
                return result
            except Exception as e:
                # 业务异常不抛出（要给模型自纠机会），但必须在 span 上留痕
                sp.set_attr("error", str(e))
                return f"工具 {name} 执行失败：{e}"
    def _execute_calls(
        self,
        calls: list[ChatCompletionMessageToolCall],
        tr: Any = None,
        parent_id: int | None = None,
    ) -> list[str]:
        """执行一批工具调用。

        只读 / 网络类并发执行省 I/O 等待，写库类串行执行避免 sqlite 锁竞争。
        返回结果顺序严格等于 calls 顺序 —— tool_call_id 的对应关系不能乱。
        tr/parent_id 透传给每个工具 span（Trace 内部带锁，并发开 span 安全）。
        """
        if len(calls) <= 1:
            return [self._run_tool(c, tr, parent_id) for c in calls]

        concurrent = [c for c in calls if c.function.name in READ_ONLY_TOOLS]
        sequential = [c for c in calls if c.function.name not in READ_ONLY_TOOLS]

        results: dict[str, str] = {}

        if concurrent:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=min(len(concurrent), 8)) as pool:
                # pool.map 保证输出顺序与输入一致；tr 是普通对象，闭包直接捕获
                for c, r in zip(
                    concurrent,
                    pool.map(lambda x: self._run_tool(x, tr, parent_id), concurrent),
                ):
                    results[c.id] = r
            if VERBOSE:
                print(
                    f"  [并行] {len(concurrent)} 个只读工具并发执行，"
                    f"耗时 {time.perf_counter() - t0:.2f}s"
                )

        for c in sequential:
            results[c.id] = self._run_tool(c, tr, parent_id)

        return [results[c.id] for c in calls]

    def _turn_stream(
        self, called: set[str], tr: Any = None
    ) -> Iterator[dict[str, Any]]:
        """流式跑完一整轮「模型 ⇄ 工具」交互。

        与旧 _turn 逻辑一致，区别只在 LLM 调用改 stream=True：
        - 文本增量边到边 yield（含工具轮次的中间话）
        - tool_calls 参数是分片到达的，按 index 攒齐后再执行
        - 攒齐后构造的消息结构与非流式完全一致，后续执行/回灌逻辑不变
        """
        for _round in range(MAX_TOOL_ROUNDS):
            with trace_span(
                tr,
                "llm.round",
                tr.root.span_id if tr else None,
                round=_round + 1,
            ) as rsp:
                yield {"type": "round_start"}
                stream = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.messages,
                    tools=self.tools,
                    temperature=0.4,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                text_parts: list[str] = []
                tc_acc: dict[int, dict[str, str]] = {}
                usage = None
                for chunk in stream:
                    if chunk.usage:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    delta = chunk.choices[0].delta
                    if delta.content:
                        text_parts.append(delta.content)
                        yield {"type": "text", "content": delta.content}
                    for tc in delta.tool_calls or []:
                        slot = tc_acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                        if tc.id:
                            slot["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                slot["name"] += tc.function.name
                            if tc.function.arguments:
                                slot["arguments"] += tc.function.arguments
                if usage:
                    self.last_prompt_tokens = usage.prompt_tokens
                    rsp.set_attr("prompt_tokens", usage.prompt_tokens)

                if not tc_acc:
                    # 没有工具调用 = 说完了
                    rsp.set_attr("reply_chars", len("".join(text_parts)))
                    self.messages.append({"role": "assistant", "content": "".join(text_parts)})
                    return

                calls = [
                    ChatCompletionMessageToolCall(
                        id=s["id"],
                        type="function",
                        function=cast(Any, {"name": s["name"], "arguments": s["arguments"]}),
                    )
                    for s in tc_acc.values()
                ]
                rsp.set_attr("tools", [c.function.name for c in calls])
                rsp.set_attr("reply_chars", len("".join(text_parts)))
                # 关键 1：把「模型要求调工具」这条 assistant 消息写进历史
                self.messages.append(
                    {
                        "role": "assistant",
                        "content": "".join(text_parts) or None,
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
                # 工具 span 挂在本轮 llm.round 下；无 trace 时 parent 取 None
                results = self._execute_calls(
                    calls, tr, getattr(rsp, "span_id", None)
                )
                for c, result in zip(calls, results):
                    called.add(c.function.name)
                    yield {"type": "tool_result", "name": c.function.name, "result": result}
                    self.messages.append(
                        {"role": "tool", "tool_call_id": c.id, "content": result}
                    )

        yield {"type": "timeout"}

    def chat_stream(self, user_text: str) -> Iterator[dict[str, Any]]:
        """对外流式入口：每轮对话开一条 trace，具体逻辑在 _chat_stream_inner。

        事件契约（第二步 FastAPI/SSE 直接序列化这一层的 dict）：
          {"type": "round_start"}                       新一轮模型调用开始
          {"type": "text", "content": str}            文本增量（工具轮的中间话也会吐）
          {"type": "tool_result", "name", "result"}  某个工具执行完
          {"type": "claim_warning", "message": str}   口头完成校验命中，正在重试
          {"type": "final", "reply", "trace_id"}     本轮结束，保证恰好一次
        """
        # trace 在生成器挂起期间一直开着；流中途异常/客户端断开都会经 __exit__ 收尾
        with start_trace(
            "chat",
            session_id=self.session_id,
            user_id=self.user_id,
            input_chars=len(user_text),
        ) as tr:
            yield from self._chat_stream_inner(user_text, tr)

    def _chat_stream_inner(
        self, user_text: str, tr: Any
    ) -> Iterator[dict[str, Any]]:
        """chat_stream 的原有编排逻辑，运行在根 trace 上下文内。"""
        # 上一轮可能刚写入/删除了记忆，system prompt 必须跟着变，
        # 否则新记忆要等到下次重启才生效，用户会觉得"记了跟没记一样"。
        set_current_session(self.session_id)
        set_current_user(self.user_id)
        if self.reload_system_prompt and self.messages:
            self.messages[0]["content"] = load_system_prompt(self.session_id)
        start = len(self.messages)          # 本轮新增的消息都要落库
        self.messages.append({"role": "user", "content": user_text})

        reply = ""
        warn: str | None = None
        called_all: set[str] = set()
        retries = 0
        for _attempt in range(MAX_CLAIM_RETRY + 1):
            # called_all 传进生成器原地累积，跨重试不丢上一轮的工具成果
            for ev in self._turn_stream(called_all, tr):
                t = ev["type"]
                if t == "round_start":
                    # 每轮换新：只攒当前轮文本，工具轮的中间话不算最终回复
                    reply = ""
                elif t == "text":
                    reply += ev["content"]
                elif t == "timeout":
                    reply = "（工具调用超过上限，已停止）"
                yield ev

            warn = (
                check_claim(reply, called_all)
                if (self.enable_claim_check and has_change_intent(user_text))
                else None
            )
            if not warn:
                break

            log.warning("claim_without_tool", attempt=_attempt, reply=reply)
            retries += 1
            if VERBOSE:
                print(f"  [校验] {warn} → 已要求模型真正执行")
            yield {"type": "claim_warning", "message": warn}
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

        # 本轮产生的所有消息（user / assistant / tool）落库
        # worker 不落库：避免它的专精对话污染主会话历史
        if self.persist_messages:
            for m in self.messages[start:]:
                save_message(self.session_id, dict(m))

        if VERBOSE:
            print(
                f"  [token] prompt={self.last_prompt_tokens} "
                f"（阈值 {CTX_SOFT_LIMIT}）｜上下文 {len(self.messages)} 条"
            )
        if self.persist_messages:
            self._maybe_compact(tr)
        log.info("turn_done", history_len=len(self.messages))
        tr.root.set_attr("claim_retries", retries)
        tr.root.set_attr("reply_chars", len(reply))
        tr.root.set_attr("history_len", len(self.messages))
        tr.root.set_attr("prompt_tokens", self.last_prompt_tokens)
        yield {"type": "final", "reply": reply, "trace_id": tr.trace_id}

    def chat(self, user_text: str) -> str:
        """非流式封装：跑完整个事件流只取最终回复。CLI / eval.py 无需改动。"""
        reply = ""
        for ev in self.chat_stream(user_text):
            if ev["type"] == "final":
                reply = ev["reply"]
        return reply

    def _maybe_compact(self, tr: Any = None) -> None:
        """prompt token 超阈值时，把早期对话压成摘要，保留最近若干条原文。"""
        if self.last_prompt_tokens < CTX_SOFT_LIMIT:
            return

        body = self.messages[1:]                 # system 不参与压缩
        want_cut = len(body) - KEEP_RECENT
        if want_cut <= 0:
            return
        cut = safe_split_index(body, want_cut)
        if cut <= 0 or cut >= len(body):
            return                                # 没得压，或会全压掉

        old, keep = body[:cut], body[cut:]
        if VERBOSE:
            print(f"  [压缩] {len(old)} 条早期对话 → 摘要，保留最近 {len(keep)} 条原文")
        
        old_fine, old_coarse = get_summary(self.session_id)
        new_fine, new_coarse = self._summarize_messages(old, old_fine, old_coarse, tr)
        save_summary(self.session_id, fine=new_fine, coarse=new_coarse)
        # 归档要被压缩掉的早期消息 —— 原文不删，挪到 archived_messages 供后续检索 
        archive_messages(self.session_id, [dict(m) for m in old])
        # 只清消息，别动摘要（clear_session 会把刚存进去的摘要一起删掉）
        clear_messages(self.session_id)
        for m in keep:
            save_message(self.session_id, dict(m))

        self.messages = [self.messages[0]] + keep

    def _summarize_messages(
        self, msgs: list[Any], old_fine: str, old_coarse: str, tr: Any = None
    ) -> tuple[str, str]:
        """把一段对话压成两层摘要，返回 (新细摘要, 新粗摘要)。

        两级策略：
        - 细摘要 = 旧细摘要 + 新对话 的合并摘要，保留近期细节
        - 粗摘要 = 跨多轮的核心事实，不轻易改
        - 当细摘要太长时，把旧细摘要压进粗摘要，细摘要重置为只含新对话
          → 核心事实沉淀到粗摘要，不受近期闲聊稀释
        """
        lines = self._msgs_to_lines(msgs)

        # 第一步：旧细摘要 + 新对话 → 候选细摘要
        candidate_fine = self._llm_summarize(
            lines, old_fine, tr,
            brief="把已有摘要和新对话合并成一份要点摘要，400 字以内，不要丢旧摘要内容。",
        )

        if len(candidate_fine) <= FINE_SUMMARY_MAX_LEN:
            # 细摘要还能装下，粗摘要不动
            return candidate_fine, old_coarse

        # 第二步：细摘要太长了 → 把旧细摘要升级进粗摘要
        new_coarse = self._llm_summarize(
            [], old_coarse + ("\n" if old_coarse else "") + old_fine, tr,
            brief=(
                "把下面这段旧的对话要点进一步压缩成核心事实摘要，200 字以内。"
                "只保留：用户的身份/偏好/长期结论、出现过的关键数字与事实。"
                "丢掉：寒暄、近期闲聊、工具调用细节。"
            ),
        ) if old_fine else old_coarse

        # 细摘要重置：只含本次新对话（旧细摘要已进粗摘要）
        new_fine = self._llm_summarize(
            lines, "", tr,
            brief="把下面这段对话压缩成要点摘要，300 字以内，保留关键数字与事实。",
        )
        return new_fine, new_coarse

    def _msgs_to_lines(self, msgs: list[Any]) -> list[str]:
        """把消息列表转成给 LLM 看的文本行。"""
        lines: list[str] = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                lines.append(f"用户：{content}")
            elif role == "assistant":
                if content:
                    lines.append(f"助手：{content}")
                for tc in m.get("tool_calls", []):
                    fn = tc.get("function", {})
                    lines.append(f"助手调用工具：{fn.get('name')}（{fn.get('arguments')}）")
            elif role == "tool":
                lines.append(f"工具返回：{str(content)[:300]}")
        return lines

    def _llm_summarize(
        self, lines: list[str], prev: str, tr: Any = None, brief: str = ""
    ) -> str:
        """调 LLM 做摘要。不传 tools，避免递归调工具。tr 仅用于追踪。"""
        dialog = chr(10).join(lines) if lines else "（无新对话）"
        prompt = f"""{brief}

已有内容：
{prev or "（无）"}

新对话：
{dialog}"""
        with trace_span(
            tr, "llm.summarize", tr.root.span_id if tr else None,
            prompt_chars=len(prompt),
        ):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
        return resp.choices[0].message.content or ""
    def extract_memories(self, tr: Any = None) -> list[str]:
        """扫一遍最近对话，自动提取值得长期保留的事实，写进记忆库。

        复用 remember 的去重/相似更新逻辑，不重复造轮子。
        提取用纯 LLM 调用（不传 tools），避免递归调工具。
        只看当前上下文里的对话 —— 已被压缩进摘要的旧事实看不到，这是已知盲点。
        """
        dialog_msgs = [m for m in self.messages if m.get("role") != "system"][-30:]
        lines: list[str] = []
        for m in dialog_msgs:
            role = m.get("role")
            content = m.get("content")
            if role == "user" and content:
                lines.append(f"用户：{content}")
            elif role == "assistant" and content:
                lines.append(f"助手：{content}")
            # tool 消息跳过：那是工具返回的事实，不是用户告诉我的
        if not lines:
            return []

        prompt = EXTRACT_PROMPT.replace("{memories}", list_memories()).replace(
            "{dialog}", "\n".join(lines)
        )
        with trace_span(
            tr, "llm.extract_memories", tr.root.span_id if tr else None,
            lines=len(lines),
        ):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
        items = self._parse_json_list(resp.choices[0].message.content or "")
        if not items:
            return []

        saved: list[str] = []
        for text in items:
            text = text.strip()
            if not text:
                continue
            result = remember(text)  # 复用去重 / 相似更新
            if "已记住" in result or "已更新记忆" in result:
                saved.append(result)
        return saved

    def _parse_json_list(self, raw: str) -> list[str]:
        """从模型回复里抠出 JSON 字符串数组。容错：去 markdown 围栏、定位 [ ]。"""
        s = raw.strip()
        if s.startswith("```"):
            nl = s.find("\n")
            if nl > 0:
                s = s[nl + 1 :]
            if s.endswith("```"):
                s = s[:-3].rstrip()
        lo, hi = s.find("["), s.rfind("]")
        if lo < 0 or hi <= lo:
            return []
        try:
            obj = json.loads(s[lo : hi + 1])
            if isinstance(obj, list):
                return [x for x in obj if isinstance(x, str)]
        except json.JSONDecodeError:
            return []
        return []

def main() -> None:
    session_id = "default"
    if "--new" in sys.argv:
        clear_session(session_id)
        print("已清空历史会话，重新开始。\n")

    system_prompt = load_system_prompt(session_id) + external_tools_prompt()
    tools, tool_functions = merged_tools()  # 本地工具 + 外部 MCP 工具
    agent = Agent(
        system_prompt=system_prompt, session_id=session_id, user_id="default",
        tools=tools, tool_functions=tool_functions,
    )
    ext = len(tools) - len(TOOLS)
    print(f"Agent 就绪（提示词 {len(system_prompt)} 字，外部 MCP 工具 {ext} 个）。输入 exit 退出。\n")

    try:
        while True:
            try:
                text = input("你 > ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n再见")
                break

            if not text:
                continue
            if text.lower() in {"exit", "quit", "退出"}:
                print("  [记忆] 回顾本轮对话，沉淀值得保留的事实…")
                # Ctrl-C / EOF 退出路径 不触发 提取（用户急退就别挡路），只在显式`exit/quit/退出` 时跑。
                try:
                    saved = agent.extract_memories()
                    if saved:
                        print("  [记忆] 已自动沉淀：")
                        for s in saved:
                            print(f"    - {s}")
                    else:
                        print("  [记忆] 本轮没有需要新记的长期事实。")
                except Exception as e:
                    log.error("extract_memories_failed", error=str(e))
                    print(f"  [记忆] 自动沉淀失败：{e}")
                print("\n再见")
                break
            try:
                print("AI > ", end="", flush=True)
                for ev in agent.chat_stream(text):
                    t = ev["type"]
                    if t == "text":
                        print(ev["content"], end="", flush=True)
                    elif t == "tool_result":
                        # 工具轮：换行打印工具动作，再重新起一行继续接后续文本
                        print(f"\n  [工具] {ev['name']} -> {ev['result']}")
                        print("AI > ", end="", flush=True)
                    elif t == "claim_warning":
                        print(f"\n  [校验] {ev['message']} → 重试…")
                        print("AI > ", end="", flush=True)
                    elif t == "final":
                        print("\n")
            except Exception as e:
                log.error("chat_failed", error=str(e))
                print(f"[出错] {e}\n")
    finally:
        close_shared_hub()  # 断开外部 MCP 子进程，避免退出后残留

if __name__ == "__main__":
    main()