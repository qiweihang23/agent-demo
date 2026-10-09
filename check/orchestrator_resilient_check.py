"""B-4 验证：失败回溯 + 计划动态修正。

测试用例逼出 update_plan：
  "查杭州天气，如果湿度高于70%就加待办'带伞'，否则加待办'不用带伞'"
  → make_plan([查天气, 加'带伞'待办])  ← 计划写死了「带伞」分支
  → handoff(1) 查天气
  → 若实际湿度<70%，原计划第二步不合适
  → supervisor 应主动 update_plan('replace', 2, "加'不用带伞'待办")
  → handoff(2)
  → evaluate_task

跑法：uv run python orchestrator_resilient_check.py
"""
from __future__ import annotations

from orchestrator import build_resilient_orchestrator
from tools import clear_session, list_todos

SESSION = "orch-resilient"

def main() -> None:
    clear_session(SESSION)
    sup, plan = build_resilient_orchestrator(session_id=SESSION, user_id="orch-user")

    query = "查杭州天气，如果湿度高于70%就加待办'带伞'，否则加待办'不用带伞'"
    print(f"你 > {query}\n")
    print("AI > ", end="", flush=True)

    for ev in sup.chat_stream(query):
        t = ev["type"]
        if t == "text":
            print(ev["content"], end="", flush=True)
        elif t == "tool_result":
            name = ev["name"]
            result = ev["result"]
            if name == "make_plan":
                print(f"\n  [PLAN] {result}")
            elif name == "update_plan":
                print(f"\n  [UPDATE-PLAN] {result}")
            elif name == "decide_after_failure":
                print(f"\n  [DECIDE] {result}")
            elif name.startswith("evaluate"):
                print(f"\n  [EVAL] {name} -> {result}")
            else:
                print(f"\n  [handoff] {name} -> {result[:80]}...")
            print("AI > ", end="", flush=True)
        elif t == "final":
            print(f"\n  [trace] {ev['trace_id']}")
    print()

    print("\n=== 计划最终状态（B-4 特征：可能看到步骤被 replace/insert/delete）===")
    print(plan.format())
    print(f"\n全部通过: {plan.all_done()}")

    print("\n=== 主会话待办 ===")
    print(list_todos(session_id=SESSION))

if __name__ == "__main__":
    main()