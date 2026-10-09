"""回归测试：把踩过的坑固化成用例，一条命令跑完看 PASS/FAIL。

用法（在项目根目录执行）：
    uv run python tests/eval.py
    uv run python tests/eval.py --only mem      # 只跑 id 前缀为 mem 的用例
    uv run python tests/eval.py -v              # FAIL 时打印完整回复
    uv run python tests/eval.py --no-retry      # 关闭 FAIL 自动重试
"""
from __future__ import annotations
from typing import Any
import argparse
import contextlib
import io
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # 不加这行，从根目录跑会 import 不到 agent/tools

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import tools
import rag
from agent import Agent, load_system_prompt
import agent as agent_mod
agent_mod.CTX_SOFT_LIMIT = 1500   # 测试时压低阈值，两轮就触发压缩，不必灌长对话
# ---- 隔离：整个测试用独立库，绝不碰 data/assistant.db ----
tools.DB_PATH = ROOT / "data" / "eval.db"
tools._init_db()   # 模块导入时已对真库建过表，这里对新库再建一次（建表语句只有这一处）

CASES_FILE = Path(__file__).parent / "cases.jsonl"


def reset_state() -> None:
    """清空全局表。todos/memories 没有 session 维度，只能靠这个做用例隔离。"""
    with contextlib.closing(sqlite3.connect(tools.DB_PATH)) as conn:
        conn.execute("DELETE FROM todos")
        conn.execute("DELETE FROM memories")
        conn.commit()


def run_step(agent: Agent, text: str) -> tuple[str, str, set[str]]:
    """跑一轮对话，同时抓住 print 输出（[工具]/[校验]/[压缩] 都在里面）。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        reply = agent.chat(text)
    out = buf.getvalue()
    called = set(re.findall(r".工具. (\w+) ->", out))
    return reply, out, called


def check(exp: dict[str,Any], reply: str, out: str, called: set[str], sid: str) -> list[str]:
    bad: list[str] = []

    for t in exp.get("tools", []):
        if t not in called:
            bad.append(f"未调用工具 {t}（实际调用：{sorted(called) or '无'}）")
    for t in exp.get("tools_not", []):
        if t in called:
            bad.append(f"不应调用 {t}")

    has = exp.get("reply_has")
    if has:
        pats = [has] if isinstance(has, str) else has
        if not any(p in reply for p in pats):
            bad.append(f"回复缺少 {pats}")
    for p in exp.get("reply_has_none", []):
        if p in reply:
            bad.append(f"回复不应出现「{p}」")

    if exp.get("no_warn") and "[校验]" in out:
        bad.append("触发了「口头完成」校验（误报）")
    if exp.get("compacted") and "[压缩]" not in out:
        bad.append("未触发压缩（阈值可能没到）")
    if exp.get("summary_not_empty"):
        fine, coarse = tools.get_summary(sid)
        if not (fine or coarse).strip():
            bad.append("压缩后摘要为空")
    if exp.get("summary_fine_not_empty"):
        fine, _ = tools.get_summary(sid)
        if not fine.strip():
            bad.append("细摘要为空")
    if exp.get("summary_coarse_not_empty"):
        _, coarse = tools.get_summary(sid)
        if not coarse.strip():
            bad.append("粗摘要为空")

    # 归档历史计数：压缩时早期消息应被归档而非删除
    with contextlib.closing(sqlite3.connect(tools.DB_PATH)) as c:
        n_arch = c.execute(
            "SELECT COUNT(*) FROM archived_messages WHERE session_id=?", (sid,)
        ).fetchone()[0]
    if "archived_count" in exp and n_arch != exp["archived_count"]:
        bad.append(f"归档消息数应为 {exp['archived_count']}，实际 {n_arch}")
    if "archived_count_min" in exp and n_arch < exp["archived_count_min"]:
        bad.append(f"归档消息数应不少于 {exp['archived_count_min']}，实际 {n_arch}")

    mem = tools.list_memories()
    for p in exp.get("memory_has", []):
        if p not in mem:
            bad.append(f"记忆库缺少 {p}")
    for p in exp.get("memory_not", []):
        if p in mem:
            bad.append(f"记忆库不应含 {p}")
    # 验证旧值被归档（不是被删，也不是被覆盖）
    for p in exp.get("archived_has", []):
        with contextlib.closing(sqlite3.connect(tools.DB_PATH)) as c:
            row = c.execute(
                "SELECT content FROM memories WHERE valid_to IS NOT NULL AND content LIKE ?",
                (f"%{p}%",),
            ).fetchone()
        if not row:
            bad.append(f"归档记忆中找不到「{p}」（旧值可能被删或被覆盖，而非归档）")
    if "memory_count" in exp:
        n = mem.count(exp["memory_count"])
        if n != 1:
            bad.append(f"「{exp['memory_count']}」应只出现 1 次，实际 {n} 次")

    todos = tools.list_todos()
    if "todos_total" in exp:
        m = re.search(r"共 (\d+) 条", todos)
        if not m or int(m.group(1)) != exp["todos_total"]:
            bad.append(f"待办总数应为 {exp['todos_total']}，实际 {m.group(1) if m else '?'}")
    if "todos_undone" in exp:
        m = re.search(r"未完成 (\d+) 条", todos)
        if not m or int(m.group(1)) != exp["todos_undone"]:
            bad.append(f"未完成应为 {exp['todos_undone']}，实际 {m.group(1) if m else '?'}")

    return bad


def run_case(case: dict[str,Any], verbose: bool) -> list[str]:
    sid = f"eval_{case['id']}"
    tools.clear_session(sid)
    reset_state()

    agent = Agent(system_prompt=load_system_prompt(sid), session_id=sid)
    fails: list[str] = []

    for i, step in enumerate(case["steps"], 1):
        if step.get("new_session"):          # 模拟关掉重开：换 session，但 memories 表共享
            sid = f"{sid}_s{i}"
            tools.clear_session(sid)
            agent = Agent(system_prompt=load_system_prompt(sid), session_id=sid)

        reply, out, called = "", "", set()
        if step.get("call"):                 # 直接调工具，不经 LLM：测确定性逻辑
            fn_name = step["call"]["fn"]
            # 工具可能在 tools 或 rag 里
            fn = getattr(tools, fn_name, None) or getattr(rag, fn_name, None)
            if fn is None:
                raise AttributeError(f"找不到工具函数 {fn_name}（tools 和 rag 里都没有）")
            with contextlib.redirect_stdout(io.StringIO()):
                reply = fn(**step["call"].get("args", {}))
            out = f"  [工具] {fn_name} -> {reply}\n"
            called = {fn_name}
        else:
            reply, out, called = run_step(agent, step["say"])

        if step.get("extract"):              # 隐式记忆提取：模拟"退出时自动沉淀"
            with contextlib.redirect_stdout(io.StringIO()):
                agent.extract_memories()
        bad = check(step.get("expect", {}), reply, out, called, sid)
        if bad:
            fails += [f"step{i}: {b}" for b in bad]
            if verbose:
                fails.append(f"step{i} 回复全文：\n{reply}")
                fails.append(f"step{i} 控制台输出：\n{out}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="只跑 id 为此前缀的用例，如 mem")
    ap.add_argument("-v", action="store_true", help="FAIL 时打印完整回复")
    ap.add_argument("--no-retry", action="store_true", help="关闭 FAIL 自动重试")
    args = ap.parse_args()

    cases: list[dict[str,Any]] = []
    for line in CASES_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    if args.only:
        cases = [c for c in cases if c["id"].startswith(args.only)]

    passed = 0
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        fails = run_case(case, args.v)
        note = ""
        if fails and not args.no_retry:      # LLM 有随机性，FAIL 再给一次机会
            fails = run_case(case, args.v)
            if not fails:
                note = "（重试通过）"
        dt = time.perf_counter() - t0

        if fails:
            print(f"[{i:2}/{len(cases)}] {case['id']:<9} {case['name']:<22} FAIL  {dt:.1f}s")
            for f in fails:
                print(f"           └ {f}")
        else:
            passed += 1
            print(f"[{i:2}/{len(cases)}] {case['id']:<9} {case['name']:<22} PASS  {dt:.1f}s{note}")

    print(f"\n{passed}/{len(cases)} PASS")
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    sys.exit(main())
