"""阶段 5 验证：记忆语义查重（字面 + 向量双路），兼测阶段 4 隔离回归。

运行：uv run python verify.py
"""
import os
import sys
import sqlite3
from contextlib import closing
from pathlib import Path

# 用独立库，绝不碰 data/assistant.db
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
import tools  # noqa: E402

tools.DB_PATH = ROOT / "data" / "verify.db"
# 删掉旧的 verify.db，保证每次从零开始
if tools.DB_PATH.exists():
    tools.DB_PATH.unlink()
tools._init_db()  # 重新建表

def assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"FAIL: {msg}")
    print(f"  PASS: {msg}")

# ===================== 1. 待办按会话隔离 =====================
print("\n[1] 待办按会话隔离")

tools.set_current_session("sess_A")
tools.add_todos(["买牛奶", "写周报"])

tools.set_current_session("sess_B")
tools.add_todos(["预约体检"])

# A 会话应该只看到自己的 2 条
tools.set_current_session("sess_A")
out_a = tools.list_todos()
assert_true("买牛奶" in out_a and "写周报" in out_a, "会话A能看到自己的待办")
assert_true("预约体检" not in out_a, "会话A看不到会话B的待办")

# B 会话应该只看到自己的 1 条
tools.set_current_session("sess_B")
out_b = tools.list_todos()
assert_true("预约体检" in out_b, "会话B能看到自己的待办")
assert_true("买牛奶" not in out_b, "会话B看不到会话A的待办")

# A 会话的待办不能被 B 会话的 complete/delete 操作影响
tools.set_current_session("sess_B")
# 尝试删除 A 会话的待办（id=1 属于 sess_A）
result = tools.delete_todo(1)
assert_true("没有编号为 #1" in result, "会话B无法删除会话A的待办（隔离生效）")

# ===================== 2. 记忆按用户隔离 + 查重不串味 =====================
print("\n[2] 记忆按用户隔离 + 查重不串味")

tools.set_current_user("user_张三")
tools.remember("用户在杭州工作")

tools.set_current_user("user_李四")
# 李四说"在上海工作"——相似度高，如果隔离没做好会把张三的记忆覆盖掉
tools.remember("用户在上海工作")

# 张三的记忆应该还在，且是"杭州"
tools.set_current_user("user_张三")
mem_zs = tools.list_memories()
assert_true("杭州" in mem_zs, "张三的记忆（杭州）未被李四覆盖")
assert_true("上海" not in mem_zs, "张三看不到李四的记忆")

# 李四的记忆是"上海"
tools.set_current_user("user_李四")
mem_ls = tools.list_memories()
assert_true("上海" in mem_ls, "李四的记忆（上海）正确")
assert_true("杭州" not in mem_ls, "李四看不到张三的记忆")

# 李四不能删张三的记忆
tools.set_current_user("user_李四")
result = tools.forget("杭州")
assert_true("没有找到" in result, "李四无法删除张三的记忆")

# ===================== 3. clear_session / clear_user 互不影响 =====================
print("\n[3] clear_session / clear_user 互不影响")

# 清掉 B 会话的待办，A 会话的应该还在
tools.clear_session("sess_B")
tools.set_current_session("sess_A")
assert_true("买牛奶" in tools.list_todos(), "清 sess_B 不影响 sess_A 的待办")

# 清掉李四的记忆，张三的应该还在
tools.clear_user("user_李四")
tools.set_current_user("user_张三")
assert_true("杭州" in tools.list_memories(), "清 user_李四 不影响 user_张三 的记忆")
tools.set_current_user("user_李四")
assert_true("暂无记忆" in tools.list_memories(), "user_李四 的记忆已被清空")

# ===================== 4. 阈值校准数据（先看数据再断言，坑 8 教训） =====================
print("\n[4] 语义相似度实测（高段直接归档，中段交 LLM 判冲突）")
pairs = [
    ("用户在杭州工作", "我搬到北京了", "互斥更新→中段，LLM 判定"),
    ("用户在杭州工作", "用户喜欢吃辣", "无关→LLM 应判无冲突"),
    ("用户在杭州工作", "用户在上海工作", "同构换词→高段直接归档"),
]
vs = tools._embed([t for p in pairs for t in p[:2]])
for i, (a, b, label) in enumerate(pairs):
    if vs is not None:
        print(f"  {tools._cosine(vs[i*2], vs[i*2+1]):.4f}  {label}: 「{a}」↔「{b}」")
    else:
        print(f"  [向量嵌入失败]  {label}: 「{a}」↔「{b}」")

# ===================== 5. 语义互斥更新 → 归档（阶段 5 核心） =====================
print("\n[5] 语义互斥更新 → 归档")
tools.set_current_user("user_语义")
tools.remember("用户在杭州工作")
out = tools.remember("我搬到北京了")
print(f"  remember 返回: {out}")
assert_true("已更新" in out, "字面无关但语义互斥 → 走归档更新")
mems = tools.list_memories()
assert_true("北京" in mems and "杭州" not in mems, "当前真相只剩北京，矛盾不并存")
out = tools.remember("用户喜欢吃辣")
assert_true("已记住" in out, "无关事实普通新增，不误伤")

# ===================== 6. 数据库层确认列已正确写入 =====================
print("\n[6] 数据库层确认列已正确写入")
with closing(sqlite3.connect(tools.DB_PATH)) as conn:
    todos_cols = {r[1] for r in conn.execute("PRAGMA table_info(todos)")}
    mems_cols = {r[1] for r in conn.execute("PRAGMA table_info(memories)")}
    assert_true("session_id" in todos_cols, "todos 表有 session_id 列")
    assert_true("user_id" in mems_cols, "memories 表有 user_id 列")
    assert_true("vec" in mems_cols, "memories 表有 vec 列")
    n_vec = conn.execute("SELECT COUNT(*) FROM memories WHERE vec IS NOT NULL").fetchone()[0]
    print(f"  有向量的记忆条数: {n_vec}")

    # 张三的记忆行 user_id 确实是 user_张三
    row = conn.execute(
        "SELECT user_id FROM memories WHERE content LIKE '%杭州%'"
    ).fetchone()
    assert_true(row is not None and row[0] == "user_张三", "记忆行的 user_id 字段正确")

print("\n✅ 全部验证通过：阶段 5 语义查重生效，阶段 4 隔离回归通过")