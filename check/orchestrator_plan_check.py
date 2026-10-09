"""B-2 验证：显式规划 + 计划-执行分离。

测试用例和 B-1 一样，但这次 supervisor 必须先 make_plan 再 handoff。
跑完后打印 plan.format()，能看到每步的 status/result —— 这是 L2 做不到的。

跑法：uv run python orchestrator_plan_check.py
"""
from __future__ import annotations

from orchestrator import build_planning_orchestrator
from tools import clear_session, list_todos

SESSION = "orch-plan"

def main() -> None:
    clear_session(SESSION)
    sup, plan = build_planning_orchestrator(session_id=SESSION, user_id="orch-user")

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
            else:
                print(f"\n  [handoff] {name} -> {result[:80]}...")
            print("AI > ", end="", flush=True)
        elif t == "final":
            print(f"\n  [trace] {ev['trace_id']}")
    print()

    # 关键验证：plan 是可观察的对象，跑完后能打印完整进度
    # L2 的 supervisor 做不到这点 —— 它的"计划"只在脑子里，跑完就没了
    print("\n=== 计划最终状态（L3 特征：可观察、可追踪）===")
    print(plan.format())
    print(f"\n全部完成: {plan.all_done()}")

    print("\n=== 主会话待办 ===")
    print(list_todos(session_id=SESSION))

if __name__ == "__main__":
    main()