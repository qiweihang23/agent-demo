"""B-3 验证：自我评估循环。

测试用例逼出自我评估：
  "查杭州天气，如果湿度高于70%就加待办'带伞'"
  → make_plan → handoff(1) → evaluate_step(1) → handoff(2) → evaluate_step(2) → evaluate_task

跑完后 plan.format() 能看到每步 evaluation/retry_count/feedback —— B-2 看不到这些。

跑法：uv run python orchestrator_self_eval_check.py
"""
from __future__ import annotations

from orchestrator import build_self_evaluating_orchestrator
from tools import clear_session, list_todos

SESSION = "orch-eval"

def main() -> None:
    clear_session(SESSION)
    sup, plan = build_self_evaluating_orchestrator(session_id=SESSION, user_id="orch-user")

    query = "查杭州天气，如果湿度高于70%就加待办'带伞'"
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
            elif name.startswith("evaluate"):
                print(f"\n  [EVAL] {name} -> {result}")
            else:
                print(f"\n  [handoff] {name} -> {result[:80]}...")
            print("AI > ", end="", flush=True)
        elif t == "final":
            print(f"\n  [trace] {ev['trace_id']}")
    print()

    print("\n=== 计划最终状态（B-3 特征：含 evaluation/retry_count）===")
    print(plan.format())
    print(f"\n全部通过: {plan.all_done()}")

    print("\n=== 主会话待办 ===")
    print(list_todos(session_id=SESSION))

if __name__ == "__main__":
    main()