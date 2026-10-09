"""B-6 验证：自驱循环（无人输入，agent 自己推进目标）。

场景设计（关键是后两轮）：
  对话 1（用户）：创建目标「这周读完《活着》」→ next_step = 等待今日反馈
  对话 2（用户）：今天读了 35 页 → next_step = 添加明天的阅读任务
  自驱轮 1（无用户）：调度器触发 → agent 读 next_step，自己加明日待办 + advance_goal
  自驱轮 2（无用户）：再次触发 → next_step 已变成"等用户反馈" → 应跳过，不伪造用户输入

跑法：uv run python orchestrator_selfdrive_check.py
"""
from __future__ import annotations

from contextlib import closing

from orchestrator import build_goal_oriented_orchestrator, run_proactive_round
from tools import clear_session, set_current_user, _connect

SESSION = "orch-selfdrive"
USER = "selfdrive-user"

def run_round(sup, query: str, label: str) -> None:
    print(f"\n========== {label} ==========")
    print(f"你 > {query}\n")
    print("AI > ", end="", flush=True)
    for ev in sup.chat_stream(query):
        t = ev["type"]
        if t == "text":
            print(ev["content"], end="", flush=True)
        elif t == "tool_result":
            print(f"\n  [{ev['name']}] {ev['result'][:100]}")
            print("AI > ", end="", flush=True)
        elif t == "final":
            print(f"\n  [trace] {ev['trace_id']}")
    print()

def dump_goal() -> None:
    set_current_user(USER)
    from tools import get_goal_next_steps
    print("\n--- goals 表当前状态 ---")
    for g in get_goal_next_steps():
        print(f"  #{g['goal_id']} [{g['status']}] {g['description']}")
        print(f"    next_step：{g['next_step']}")

def main() -> None:
    # 清场：清对话 + 清该用户所有 goals
    clear_session(SESSION)
    set_current_user(USER)
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM goals WHERE user_id = ?", (USER,))
        conn.commit()

    # 对话 1：用户建立目标
    sup1, _ = build_goal_oriented_orchestrator(session_id=SESSION, user_id=USER)
    run_round(sup1, "我想这周读完《活着》，请帮我跟踪进度", "对话1：新建目标（用户触发）")

    # 对话 2：用户反馈进度 → next_step 应变成"添加明日任务"
    sup2, _ = build_goal_oriented_orchestrator(session_id=SESSION, user_id=USER)
    run_round(sup2, "今天读了35页，帮我跟踪进度", "对话2：反馈进度（用户触发）")

    dump_goal()

    # 自驱轮 1：没有用户输入 —— 应自己加明日待办并 advance_goal
    print("\n========== 自驱轮1：调度器触发（无用户输入）==========")
    r1 = run_proactive_round(SESSION, USER, verbose=True)
    print(f"\n[自驱轮1 最终回复] {(r1 or '(None)')[:150]}")

    dump_goal()

    # 自驱轮 2：next_step 已是"等用户反馈" → 应回复无可推进项
    print("\n========== 自驱轮2：再次触发（应无可推进项）==========")
    r2 = run_proactive_round(SESSION, USER, verbose=True)
    print(f"\n[自驱轮2 最终回复] {(r2 or '(None)')[:150]}")

    dump_goal()
    print(
        "\n验证点：\n"
        "  1. 自驱轮1 新增一条明日待办 + advance_goal 记录（progress_log 多一条）\n"
        "  2. 自驱轮2 回复'无可推进项'，goals 表无新增日志（没伪造用户回复）\n"
        "  3. 整个过程只有对话 1/2 有人说话，后两轮全是 agent 自驱"
    )

if __name__ == "__main__":
    main()