"""B-5 验证：跨对话目标记忆。

测试场景：跨两次对话推进同一个 goal。
  对话 1：用户提"想这周读完《活着》" → supervisor 应 create_goal
  对话 2：用户回来说"今天读了多少" → supervisor 启动时 get_goal 注入目标，
           应能接上推进（advance_goal 写下次该干啥）

跑法：uv run python orchestrator_goal_check.py
"""
from __future__ import annotations

from orchestrator import build_goal_oriented_orchestrator
from tools import clear_session, set_current_user, get_goal

SESSION = "orch-goal"
USER = "goal-user"

def run_round(sup, query: str, label: str) -> None:
    print(f"\n========== {label} ==========")
    print(f"你 > {query}\n")
    print("AI > ", end="", flush=True)
    for ev in sup.chat_stream(query):
        t = ev["type"]
        if t == "text":
            print(ev["content"], end="", flush=True)
        elif t == "tool_result":
            name = ev["name"]
            result = ev["result"]
            if name in ("create_goal", "advance_goal", "get_goal", "update_goal_status"):
                print(f"\n  [GOAL-{name}] {result[:120]}")
            else:
                print(f"\n  [{name}] {result[:80]}")
            print("AI > ", end="", flush=True)
        elif t == "final":
            print(f"\n  [trace] {ev['trace_id']}")
    print()

def main() -> None:
    # 清场（清对话 + 清该用户的所有 goals）
    clear_session(SESSION)
    set_current_user(USER)
    from tools import _connect, DB_PATH
    from contextlib import closing
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM goals WHERE user_id = ?", (USER,))
        conn.commit()

    # 对话 1：建立目标
    sup1, _ = build_goal_oriented_orchestrator(session_id=SESSION, user_id=USER)
    run_round(sup1, "我想这周读完《活着》，请帮我跟踪进度", "对话1：新建目标")

    # 看一下 goals 表里有什么
    print("\n--- 对话1结束后 goals 表内容 ---")
    print(get_goal())

    # 动态拿最新 goal_id（不硬编码，避免 SQLite 自增残留导致查不到）
    from contextlib import closing as _cl
    from tools import _connect as _c
    with _cl(_c()) as conn:
        row = conn.execute(
            "SELECT id FROM goals WHERE user_id = ? ORDER BY id DESC LIMIT 1",
            (USER,),
        ).fetchone()
        gid = row["id"] if row else None

    # 对话 2：换新 supervisor 实例（模拟"用户下次回来"），看能否接上
    # 输入给具体页数，逼出 advance_goal 验证跨对话接续
    sup2, _ = build_goal_oriented_orchestrator(session_id=SESSION, user_id=USER)
    run_round(sup2, "今天读了35页，帮我跟踪进度", "对话2：接续推进")

    print(f"\n--- 对话2结束后 goal #{gid} 完整进度日志 ---")
    if gid:
        print(get_goal(gid))
    else:
        print("没有目标")

if __name__ == "__main__":
    main()