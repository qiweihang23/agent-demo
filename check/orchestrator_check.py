"""B-1 验证：supervisor + 2 worker 的 handoff 模式。

测试用例刻意逼出 handoff：单 worker 干不完，必须先查天气、再写待办。
  "查杭州天气，如果湿度高于 70% 就加待办'带伞'"
  → supervisor 必须先 handoff_to_weather 拿天气事实
  → 再 handoff_to_todo 写待办
  → 最后汇总回复

跑法：uv run python orchestrator_check.py
"""
from __future__ import annotations

from orchestrator import build_orchestrator
from tools import clear_session, list_todos

SESSION = "orch"

def main() -> None:
    # 清场，保证可重复跑（supervisor 会落库对话，不清就会越积越多）
    clear_session(SESSION)

    sup = build_orchestrator(session_id=SESSION, user_id="orch-user")
    query = "查杭州天气，如果湿度高于70%就加待办'带伞'"
    print(f"你 > {query}\n")
    print("AI > ", end="", flush=True)

    for ev in sup.chat_stream(query):
        t = ev["type"]
        if t == "text":
            print(ev["content"], end="", flush=True)
        elif t == "tool_result":
            # handoff 工具的 result = worker 跑完的最终回复
            print(f"\n  [handoff] {ev['name']} -> {ev['result']}")
            print("AI > ", end="", flush=True)
        elif t == "final":
            print(f"\n  [trace] {ev['trace_id']}")
    print()

    # 验证：todo 应该出现在主会话（worker 用主会话写数据，但对话不落库）
    print("\n=== 主会话待办（应能看到'带伞'）===")
    print(list_todos(session_id=SESSION))

if __name__ == "__main__":
    main()