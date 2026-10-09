"""工具定义与实现。

约定：每个工具返回一个「给模型看的字符串」。
不要返回 dict —— 模型要的是自然语言结果，它自己会组织语言。
"""
from __future__ import annotations
from array import array
from datetime import datetime
import os
import re
import math
from collections.abc import Callable,Sequence
import sqlite3
from contextlib import closing
from pathlib import Path
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
import difflib
import json
# 当前会话 id，供 retrieve_history 等需要按会话过滤的工具读取
# 由 agent.chat() 在每轮开始时设置
_CURRENT_SESSION: str = "default"

def set_current_session(session_id: str) -> None:
    """设置当前会话 id，供工具内部按会话过滤数据时读取。"""
    global _CURRENT_SESSION
    _CURRENT_SESSION = session_id

# 当前用户 id，供 remember/forget/list_memories 等需要按用户过滤数据的工具读取
_CURRENT_USER: str = "default"

def set_current_user(user_id: str) -> None:
    """设置当前用户 id，由 agent 在启动时注入。工具签名里永远不暴露这个参数。"""
    global _CURRENT_USER
    _CURRENT_USER = user_id
from typing import Any
load_dotenv()
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "assistant.db"
DOCS_DIR = Path(__file__).parent / "documents"
SUPPORTED_SUFFIX = {".md", ".txt", ".csv", ".xlsx", ".pdf", ".docx"}
FULL_TEXT_LIMIT = 8000  # 超过这个字数就不再返回全文，改走分块摘要
CHUNK_SIZE = 3000       # 每块字符数
CHUNK_OVERLAP = 200     # 块间重叠字数，避免把一段话切碎
MAX_CHUNKS = 20         # 上限：20 * 3000 = 6 万字，超过就如实拒绝
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_BATCH = 10      # 单次请求的文本条数上限，DashScope 有批量限制
MEM_SIM_THRESHOLD = 0.65  # 相似度高于此值视为同一条记忆，更新而非新增
MEM_VEC_THRESHOLD = 0.82  # 语义高置信线：直接归档，不问 LLM。只影响成本，不影响正确性
MEM_VEC_RECALL = 0.45    # 语义召回线：之上的中段候选交给 LLM 判定冲突
MEM_MAX = 50              # 记忆上限，超了要先清理
# WMO 天气码 → 中文。不给这张表，模型看到 weather_code=3 只能瞎猜。
WMO = {
    0: "晴", 1: "大部晴朗", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "小毛毛雨", 53: "毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨", 57: "强冻毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷阵雨", 96: "雷阵雨伴小冰雹", 99: "雷阵雨伴大冰雹",
}
#连接数据库
def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果能按列名取值，而不是按下标
    return conn
# 初始化数据库
def _init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS todos (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL DEFAULT 'default',
                text       TEXT    NOT NULL,
                done       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL
            )
            """
        )
        # 老库升级：todos 表补 session_id（按会话隔离）。
        # ALTER ADD COLUMN 不支持 IF NOT EXISTS，只能先 PRAGMA 查再补；
        # 带 NOT NULL 必须同时给 DEFAULT，且现存旧行会被填成这个默认值——
        # 借这个特性一步完成「补列 + 回填」，旧待办自动归入 default 会话。
        tcols = {r[1] for r in conn.execute("PRAGMA table_info(todos)")}
        if "session_id" not in tcols:
            conn.execute(
                "ALTER TABLE todos ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'"
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_todos_session ON todos(session_id, done, id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_name  TEXT    NOT NULL,
                chunk_idx INTEGER NOT NULL,
                text      TEXT    NOT NULL,
                mtime     REAL    NOT NULL,
                UNIQUE(doc_name, chunk_idx)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terms (
                term     TEXT    NOT NULL,
                chunk_id INTEGER NOT NULL,
                tf       INTEGER NOT NULL,
                PRIMARY KEY (term, chunk_id)
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_terms_term ON terms(term)")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT    NOT NULL DEFAULT 'default',
                content       TEXT    NOT NULL,
                created_at    TEXT    NOT NULL,
                updated_at    TEXT    NOT NULL,
                valid_to      TEXT,
                superseded_by INTEGER,
                vec           BLOB
            )
            """
        )
        # 老库升级：memories 表补缺失列（SQLite 不支持 ADD COLUMN IF NOT EXISTS）。
        # user_id 的 DEFAULT 同样兼做旧数据回填：老记忆自动归入 default 用户。
        mcols = {r[ 1 ] for r in conn.execute( "PRAGMA table_info(memories)" )} 
        if "user_id" not in mcols:
            conn.execute( "ALTER TABLE memories ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'" )
        if "valid_to" not in mcols:
            conn.execute( "ALTER TABLE memories ADD COLUMN valid_to TEXT" ) 
        if "superseded_by" not in mcols:
            conn.execute( "ALTER TABLE memories ADD COLUMN superseded_by INTEGER" )
        if "vec" not in mcols:
            conn.execute( "ALTER TABLE memories ADD COLUMN vec BLOB" )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, valid_to)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                payload    TEXT    NOT NULL,
                created_at TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS summaries (
                session_id TEXT PRIMARY KEY,
                fine_summary   TEXT    NOT NULL DEFAULT '',
                coarse_summary TEXT    NOT NULL DEFAULT '',
                updated_at TEXT    NOT NULL
            )
            """
        )
        # 老库迁移：原来只有 summary 一列，迁到 fine_summary，coarse 留空 scols = {r[ 1 ] for r in conn.execute( "PRAGMA table_info(summaries)" )} if "fine_summary" not in scols:
        scols = {r[ 1 ] for r in conn.execute( "PRAGMA table_info(summaries)" )}
        if "fine_summary" not in scols:
            conn.execute( "ALTER TABLE summaries ADD COLUMN fine_summary TEXT NOT NULL DEFAULT ''" )
            conn.execute( "UPDATE summaries SET fine_summary = summary WHERE summary IS NOT NULL" )
        if "coarse_summary" not in scols:   
            conn.execute( "ALTER TABLE summaries ADD COLUMN coarse_summary TEXT NOT NULL DEFAULT ''" )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS archived_messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT    NOT NULL,
                role        TEXT    NOT NULL,
                payload     TEXT    NOT NULL,
                text        TEXT,
                created_at  TEXT    NOT NULL,
                archived_at TEXT    NOT NULL,
                vec         BLOB    
            )
            """
        )
        # 老库补列 
        acols = {r[ 1 ] for r in conn.execute( "PRAGMA table_info(archived_messages)" )}
        if "vec" not in acols:
            conn.execute( "ALTER TABLE archived_messages ADD COLUMN vec BLOB" )
        if "text" not in acols: 
            conn.execute( "ALTER TABLE archived_messages ADD COLUMN text TEXT" )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS terms_archived (
                term     TEXT    NOT NULL,
                msg_id   INTEGER NOT NULL,
                tf       INTEGER NOT NULL,
                PRIMARY KEY (term, msg_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_terms_archived_term ON terms_archived(term)"
        )
        # 老库补 vec 列
        acols = {r[1] for r in conn.execute("PRAGMA table_info(archived_messages)")}
        if "vec" not in acols:
            conn.execute("ALTER TABLE archived_messages ADD COLUMN vec BLOB")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_archived_session ON archived_messages(session_id, id)"
        )

        # 老库升级：chunks 表可能还没有 vec 列（SQLite 不支持 ADD COLUMN IF NOT EXISTS）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        if "vec" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN vec BLOB")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name)")

        # B-5：长期目标表。粒度比 todos 大—— todos 是单步动作，
        # goals 是跨多次对话推进的成果，有状态机和进度日志。
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS goals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id          TEXT    NOT NULL DEFAULT 'default',
                description     TEXT    NOT NULL,
                success_criteria TEXT    NOT NULL,
                status           TEXT    NOT NULL DEFAULT 'pending',
                progress_log     TEXT    NOT NULL DEFAULT '[]',
                created_at       TEXT    NOT NULL,
                updated_at       TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_goals_user_status ON goals(user_id, status)"
        )

        conn.commit()

_init_db()  # 导入时就确保表存在
# 添加待办
def _norm(text: str) -> str:
    """归一化待办文本：去掉所有空白，用于判断是否重复。"""
    return re.sub(r"\s+", "", text)

# 相似度阈值：>= 此值视为疑似重复，跳过。
# 0.85 能抓住「去买蛋糕」vs「买蛋糕」（≈0.86），
# 但不会误杀「买蛋糕」vs「买蛋糕粉」（≈0.86，会误杀 —— 短文本固有限制）。
# 宁可漏检（用户多一条重复待办）也不要误杀（用户丢失待办），所以阈值偏高。
_TODO_SIMILAR_THRESHOLD = 0.85

def _find_similar(text: str, exist: dict[str, dict]) -> str | None:
    """在已存在的归一化文本里找最相似的一条，超过阈值则返回其原始文本。

    exist 的 key 是 _norm 后的文本，value 含 id。
    返回命中条目的归一化 key，或 None。
    """
    key = _norm(text)
    best_key: str | None = None
    best_ratio = 0.0
    for ex_key in exist:
        ratio = difflib.SequenceMatcher(None, key, ex_key).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_key = ex_key
    if best_ratio >= _TODO_SIMILAR_THRESHOLD:
        return best_key
    return None

def add_todos(items: list[str], session_id: str | None = None) -> str:
    """批量添加待办。已存在的条目自动跳过，不重复添加。

    session_id 给程序入口（MCP server 等）显式指定会话；为 None 时回退到
    set_current_session() 注入的当前会话。注意它【不在 TOOLS schema 里】，
    function-calling 路径的模型看不到也传不了，只有 Python 调用方能用。
    """
    sid = session_id or _CURRENT_SESSION
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    added: list[str] = []
    skipped: list[str] = []

    with closing(_connect()) as conn:
        exist = {
            _norm(r["text"]): {"id": r["id"]}
            for r in conn.execute(
                "SELECT id, text FROM todos WHERE session_id = ?",
                (sid,),
            ).fetchall()
        }
        for text in items:
            key = _norm(text)
            if not key:
                continue
            if key in exist:
                skipped.append(f"{text}（已存在 #{exist[key]['id']}）")
                continue
            # 精确匹配未命中 → 试试相似度查重（difflib）
            sim_key = _find_similar(text, exist)
            if sim_key is not None:
                skipped.append(f"{text}（疑似重复 #{exist[sim_key]['id']}：{sim_key}）")
                continue
            cur = conn.execute(
                "INSERT INTO todos (session_id, text, done, created_at) VALUES (?, ?, ?, ?)",
                (sid, text, 0, now),
            )
            new_id = cur.lastrowid
            added.append(f"#{new_id} {text}")
            exist[key] = {"id": new_id}  # 防止同一批次内自己跟自己重复
        conn.commit()

    blocks: list[str] = []
    if added:
        blocks.append(f"已添加 {len(added)} 条：\n" + "\n".join(added))
    if skipped:
        blocks.append(
            f"跳过 {len(skipped)} 条（已有相同或相似内容，未重复添加）：\n"
            + "\n".join(skipped)
        )
    return "\n".join(blocks) if blocks else "没有需要添加的新待办（全部已存在或相似）。"
#标记待办为已完成
def complete_todo(todo_id: int, session_id: str | None = None) -> str:
    """把某条待办标记为已完成。session_id 语义同 add_todos。"""
    sid = session_id or _CURRENT_SESSION
    todo_id = int(todo_id)  # 模型偶尔会传字符串，容错一下
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, text FROM todos WHERE id = ? AND session_id = ?",
            (todo_id, sid),
        ).fetchone()
        if row is None:
            return f"没有编号为 #{todo_id} 的待办。请先调用 list_todos 确认编号。"
        conn.execute(
            "UPDATE todos SET done = 1 WHERE id = ? AND session_id = ?",
            (todo_id, sid),
        )
        conn.commit()
    return f"已完成 #{todo_id} {row['text']}。"

#彻底删除待办
def delete_todo(todo_id: int, session_id: str | None = None) -> str:
    """彻底删除某条待办。session_id 语义同 add_todos。"""
    sid = session_id or _CURRENT_SESSION
    todo_id = int(todo_id)
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, text FROM todos WHERE id = ? AND session_id = ?",
            (todo_id, sid),
        ).fetchone()
        if row is None:
            return f"没有编号为 #{todo_id} 的待办。请先调用 list_todos 确认编号。"
        conn.execute(
            "DELETE FROM todos WHERE id = ? AND session_id = ?",
            (todo_id, sid),
        )
        conn.commit()
    return f"已删除 #{todo_id} {row['text']}。"

# 列出待办
def create_goal(description: str, success_criteria: str) -> str:
    """创建一个长期目标。description 是目标描述，success_criteria 是可验证的达成标准。
    区别于 todos：todos 是单步动作，goals 是跨多次对话推进的成果，有状态机。
    """
    uid = _CURRENT_USER
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect()) as conn:
        cur = conn.execute(
            "INSERT INTO goals (user_id, description, success_criteria, status, "
            "progress_log, created_at, updated_at) VALUES (?, ?, ?, 'pending', '[]', ?, ?)",
            (uid, description, success_criteria, now, now),
        )
        conn.commit()
        gid = cur.lastrowid
    return (
        f"已创建目标 #{gid}\n"
        f"描述：{description}\n"
        f"达成标准：{success_criteria}\n"
        f"状态：pending\n"
        f"下一步：调 advance_goal 记录第一次推进"
    )


def advance_goal(goal_id: int, action: str, result: str, next_step: str) -> str:
    """推进目标。记录本次 action/result，写下 next_step 给未来的自己。\n    next_step 是关键——下次对话 agent 能接上。状态自动流转：pending → in_progress → on_track。"""
    uid = _CURRENT_USER
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT status, progress_log FROM goals WHERE id = ? AND user_id = ?",
            (goal_id, uid),
        ).fetchone()
        if not row:
            return f"找不到目标 #{goal_id}"
        log = json.loads(row["progress_log"] or "[]")
        log.append({
            "ts": now,
            "action": action,
            "result": result,
            "next_step": next_step,
        })
        # 状态自动流转：首次推进 pending→in_progress，第二次开始 in_progress→on_track
        new_status = row["status"]
        if new_status == "pending":
            new_status = "in_progress"
        elif new_status == "in_progress":
            new_status = "on_track"
        conn.execute(
            "UPDATE goals SET progress_log = ?, status = ?, updated_at = ? "
            "WHERE id = ? AND user_id = ?",
            (json.dumps(log, ensure_ascii=False), new_status, now, goal_id, uid),
        )
        conn.commit()
    return (
        f"目标 #{goal_id} 推进已记录\n"
        f"本次行动：{action}\n"
        f"本次结果：{result}\n"
        f"状态：{new_status}\n"
        f"下次该干：{next_step}"
    )


def get_goal(goal_id: int | None = None) -> str:
    """查目标。不传 id 返回所有进行中（pending/in_progress/on_track）的目标；
    传 id 返回单个目标详情含完整进度日志。按 user_id 隔离。"""
    uid = _CURRENT_USER
    with closing(_connect()) as conn:
        if goal_id is None:
            rows = conn.execute(
                "SELECT id, description, status, success_criteria FROM goals "
                "WHERE user_id = ? AND status IN ('pending','in_progress','on_track') "
                "ORDER BY updated_at DESC",
                (uid,),
            ).fetchall()
            if not rows:
                return "当前没有进行中的长期目标。"
            lines = ["【进行中的长期目标】"]
            for r in rows:
                lines.append(
                    f"#{r['id']} [{r['status']}] {r['description']}\n"
                    f"  达成标准：{r['success_criteria']}"
                )
            return "\n".join(lines)
        row = conn.execute(
            "SELECT id, description, success_criteria, status, progress_log, "
            "created_at, updated_at FROM goals WHERE id = ? AND user_id = ?",
            (goal_id, uid),
        ).fetchone()
        if not row:
            return f"找不到目标 #{goal_id}"
        log = json.loads(row["progress_log"] or "[]")
        lines = [
            f"目标 #{row['id']}",
            f"描述：{row['description']}",
            f"达成标准：{row['success_criteria']}",
            f"状态：{row['status']}",
            f"创建于：{row['created_at']}，更新于：{row['updated_at']}",
            f"进度日志（共 {len(log)} 条）：",
        ]
        for i, e in enumerate(log, 1):
            lines.append(
                f"  {i}. [{e['ts']}] {e['action']} → {e['result'][:40]}...\n"
                f"     下次：{e['next_step']}"
            )
        return "\n".join(lines)


def update_goal_status(goal_id: int, status: str, reason: str = "") -> str:
    """显式状态流转：done（完成）/ blocked（卡住）/ abandoned（放弃）。
    pending/in_progress/on_track 状态由 advance_goal 自动管理，不在此处理。
    """
    if status not in ("done", "blocked", "abandoned"):
        return "错误：status 必须是 done/blocked/abandoned"
    uid = _CURRENT_USER
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT status FROM goals WHERE id = ? AND user_id = ?",
            (goal_id, uid),
        ).fetchone()
        if not row:
            return f"找不到目标 #{goal_id}"
        conn.execute(
            "UPDATE goals SET status = ?, updated_at = ? WHERE id = ? AND user_id = ?",
            (status, now, goal_id, uid),
        )
        conn.commit()
    extra = f"（原因：{reason}）" if reason else ""
    return f"目标 #{goal_id} 状态已更新为 {status}{extra}"


def get_goal_next_steps() -> list[dict]:
    """自驱循环专用：取所有进行中目标各自的最新 next_step。

    和 get_goal 的分工：get_goal 返回格式化字符串给模型看；
    这个返回结构化数据，给 orchestrator 的自驱触发器拼触发消息用。
    """
    uid = _CURRENT_USER
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, description, status, success_criteria, progress_log FROM goals "
            "WHERE user_id = ? AND status IN ('pending','in_progress','on_track') "
            "ORDER BY updated_at ASC",
            (uid,),
        ).fetchall()
    out = []
    for r in rows:
        log = json.loads(r["progress_log"] or "[]")
        next_step = log[-1]["next_step"] if log else "（尚无进度记录，先做初始推进）"
        out.append({
            "goal_id": r["id"],
            "description": r["description"],
            "status": r["status"],
            "success_criteria": r["success_criteria"],
            "next_step": next_step,
        })
    return out


def list_todos(session_id: str | None = None) -> str:
    """列出所有待办，未完成在前。session_id 语义同 add_todos。"""
    sid = session_id or _CURRENT_SESSION
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, text, done, created_at FROM todos "
            "WHERE session_id = ? ORDER BY done, id",
            (sid,),
        ).fetchall()

    if not rows:
        return "当前没有待办。"

    undone = [f"- #{r['id']} {r['text']}（{r['created_at']}）" for r in rows if not r["done"]]
    done = [f"- #{r['id']} {r['text']}" for r in rows if r["done"]]

    lines: list[str] = []
    if undone:
        lines.append("未完成：")
        lines += undone
    if done:
        lines.append("已完成：")
        lines += done
    return f"共 {len(rows)} 条，未完成 {len(undone)} 条\n" + "\n".join(lines)
# 查询天气
def get_weather(city: str) -> str:
    """查询城市当前天气，返回中文描述。"""
    with httpx.Client(timeout=10) as client:
        geo = client.get(
            GEO_URL, params={"name": city, "count": 1, "language": "zh"}
        ).json()
        if not geo.get("results"):
            return f"查不到城市「{city}」，请换一个城市名或换个说法。"

        place = geo["results"][0]
        name = place.get("name", city)
        region = " ".join(
            x for x in (place.get("country"), place.get("admin1")) if x
        )
        current = client.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,weather_code,wind_speed_10m,relative_humidity_2m",
                "timezone": "Asia/Shanghai",
            },
        ).json()["current"]

        code = current["weather_code"]
        return (
            f"{name}（{region}）：{WMO.get(code, f'未知天气码 {code}')}，"
            f"气温 {current['temperature_2m']}℃，"
            f"湿度 {current['relative_humidity_2m']}%，"
            f"风速 {current['wind_speed_10m']} km/h，"
            f"数据时间 {current['time']}。"
        )
#拼接文档路径
def _doc_path(filename: str) -> Path:
    """把文件名锁死在 documents/ 内，挡住 ../../ 这类路径穿越。"""
    p = (DOCS_DIR / filename).resolve()
    if not p.is_relative_to(DOCS_DIR.resolve()):
        raise ValueError(f"不允许读取 documents/ 以外的文件：{filename}")
    return p
#读取文档1
def _read_text(path: Path) -> str:
    """读文本。utf-8-sig 能兼容带 BOM 的文件，失败再退到 GBK。"""
    try:
        return path.read_text(encoding="utf-8-sig", errors="ignore")
    except UnicodeDecodeError:
        return path.read_text(encoding="gbk", errors="ignore")


# 格式转换层：把各种文件格式统一转成纯文本，喂给下游的 RAG 索引和摘要。
def _extract_text(path: Path) -> str:
    """按文件后缀分派解析器，统一返回纯文本。"""
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return _read_text(path)
    if suffix == ".csv":
        return _extract_csv(path)
    if suffix == ".xlsx":
        return _extract_excel(path)
    if suffix == ".pdf":
        return _extract_pdf(path)
    if suffix == ".docx":
        return _extract_docx(path)
    return _read_text(path)  # 兜底，理论上 SUPPORTED_SUFFIX 已过滤


def _df_to_markdown(df) -> str:
    """把 pandas DataFrame 转成 Markdown 表格字符串。"""
    if df.empty:
        return ""
    # NaN 转空串，避免表格里出现 nan
    df = df.fillna("")
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def _extract_csv(path: Path) -> str:
    """CSV 转 Markdown 表格。自动尝试 utf-8 / gbk 编码。"""
    import pandas as pd
    for enc in ("utf-8-sig", "gbk"):
        try:
            df = pd.read_csv(path, encoding=enc)
            return _df_to_markdown(df)
        except UnicodeDecodeError:
            continue
    return _df_to_markdown(pd.read_csv(path, encoding="utf-8", errors="ignore"))


def _extract_excel(path: Path) -> str:
    """Excel 多 sheet 转 Markdown 表格，每个 sheet 带标题。"""
    import pandas as pd
    sheets = pd.read_excel(path, sheet_name=None)
    parts = []
    for sheet_name, df in sheets.items():
        table = _df_to_markdown(df)
        if table:
            parts.append(f"## Sheet: {sheet_name}\n\n{table}")
    return "\n\n".join(parts)


def _extract_pdf(path: Path) -> str:
    """PDF 提取文本（逐页拼接，页间用分页标记分隔）。"""
    import pdfplumber
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"--- 第 {i} 页 ---\n{text}")
    return "\n\n".join(pages)


def _extract_docx(path: Path) -> str:
    """Word 提取正文段落文本。"""
    from docx import Document
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
#列出文档
def list_documents() -> str:
    """列出 documents/ 下可读取的文档及字数。"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIX
    )
    if not files:
        return "documents/ 目录下还没有可读取的文档（支持 .md .txt .csv .xlsx .pdf .docx）。"

    lines = []
    for p in files:
        size = len(_extract_text(p))
        hint = "可直接读全文" if size <= FULL_TEXT_LIMIT else "较长，会自动分块摘要"
        lines.append(f"- {p.name}（{size} 字，{hint}）")
    return "可用文档：\n" + "\n".join(lines)
_client: OpenAI | None = None

def _llm() -> OpenAI:
    """工具内部用的模型客户端。懒加载：模块导入时 env 可能还没准备好。"""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
        )
    return _client


MAP_PROMPT = """你是文档摘要助手。把下面这段内容压缩成 3-5 条要点。
要求：
- 每条不超过 40 字，一行一条，以 "- " 开头
- 只保留事实、结论、数字、人名、日期
- 不要写"本节介绍了""这一段讲了"这类套话
- 不要添加原文里没有的内容

内容：
{chunk}"""

REDUCE_PROMPT = """下面是同一篇文档各部分分别提炼出的要点。请把它们合并成一份完整摘要。
要求：
- 输出 4-6 条核心结论，一行一条，以 "- " 开头
- 保留具体数字、人名、日期等关键事实
- 去掉重复内容
- 不要出现"第一部分""本节"这类分块痕迹

各部分要点：
{points}"""

def _split_text(text: str) -> list[str]:
    """按段落切块，尽量不切断语义；块尾保留少量重叠。"""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    cur = ""

    for p in paragraphs:
        if len(p) > CHUNK_SIZE:  # 单个段落就超长，只能硬切
            if cur:
                chunks.append(cur)
                cur = ""
            for i in range(0, len(p), CHUNK_SIZE):
                chunks.append(p[i : i + CHUNK_SIZE])
            continue
        if cur and len(cur) + len(p) + 1 > CHUNK_SIZE:
            chunks.append(cur)
            cur = cur[-CHUNK_OVERLAP:]
        cur = f"{cur}\n{p}".strip() if cur else p

    if cur:
        chunks.append(cur)
    return chunks

def _summarize_long(text: str, filename: str) -> str:
    """长文档 map-reduce：分块 → 每块要点 → 合并成最终摘要。"""
    chunks = _split_text(text)
    if len(chunks) > MAX_CHUNKS:
        return (
            f"文档《{filename}》共 {len(text)} 字，分出 {len(chunks)} 块，"
            f"超出单次处理上限（{MAX_CHUNKS} 块 / 约 {MAX_CHUNKS * CHUNK_SIZE} 字）。"
            f"请让用户把文档拆成几份再分别总结。"
        )

    client = _llm()
    model = os.environ["OPENAI_MODEL"]

    # map：每块单独摘要。注意这里没有传 tools，模型不可能在这里再调工具。
    points: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        print(f"  [分块] {i}/{len(chunks)}（{len(chunk)} 字）")
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": MAP_PROMPT.replace("{chunk}", chunk)}
                ],
                temperature=0.2,
            )
            points.append(resp.choices[0].message.content or "")
        except Exception as e:
            points.append(f"（第 {i} 块摘要失败：{e}）")

    # reduce：把所有要点合并成一份
    joined = "\n".join(points)[:8000]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": REDUCE_PROMPT.replace("{points}", joined)}
            ],
            temperature=0.2,
        )
        summary = resp.choices[0].message.content or ""
    except Exception as e:
        summary = f"（合并失败：{e}）以下为各部分要点：\n{joined}"

    return (
        f"文档《{filename}》共 {len(text)} 字，已自动分成 {len(chunks)} 块摘要"
        f"（原文较长，未全文载入上下文）：\n\n{summary}"
    )

#读取文档2
def read_document(filename: str) -> str:
    """读取文档内容。短文档返回全文；长文档先截断（第 2 步改成分块摘要）。"""
    path = _doc_path(filename)
    if path.suffix.lower() not in SUPPORTED_SUFFIX:
        return f"暂不支持 {path.suffix} 文件，目前支持 .md .txt .csv .xlsx .pdf .docx。"
    if not path.exists():
        return f"没有找到「{filename}」。\n{list_documents()}"

    text = _extract_text(path)
    if not text.strip():
        return f"文档「{path.name}」是空的。"
    if len(text) > FULL_TEXT_LIMIT:
        return _summarize_long(text, path.name)

    return f"文档《{path.name}》全文（{len(text)} 字）：\n\n{text}"
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]+|[a-zA-Z0-9]+")

_embed_available: bool | None = None  # None=未验证 True=可用 False=已失败，别再重试


def _pack_vec(v: Sequence[float]) -> bytes:
    return array("f", v).tobytes()


def _unpack_vec(b: bytes) -> Sequence[float]:
    a = array("f")
    a.frombytes(b)
    return a



def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。几千个 chunk 的规模下，纯 Python 暴力算只要几十毫秒。"""
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _embed(texts: list[str]) -> list[list[float]] | None:
    """批量向量化。失败就熔断并降级 —— 不能因为 embedding 挂了就让检索整体不可用。"""
    global _embed_available
    if _embed_available is False or not texts:
        return None
    try:
        out: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH):
            resp = _llm().embeddings.create(
                model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH]
            )
            # 按 index 排序，别假设返回顺序一定等于输入顺序
            out.extend(d.embedding for d in sorted(resp.data, key=lambda x: x.index))
        _embed_available = True
        return out
    except Exception as e:
        _embed_available = False
        print(f"  [索引] embedding 不可用（{type(e).__name__}: {e}），已降级为纯关键词检索。")
        return None

def _tokenize(text: str) -> list[str]:
    """中文切 bigram（二字滑动），英文数字按词。零依赖的朴素分词。

    「预算情况」→ ['预算', '算情', '情况']。
    bigram 让「预算」这种二字词能被直接命中，同时不依赖 jieba。
    """
    tokens: list[str] = []
    for m in _TOKEN_RE.findall(text):
        if "\u4e00" <= m[0] <= "\u9fff":  # 汉字串
            if len(m) == 1:
                tokens.append(m)
            else:
                tokens.extend(m[i : i + 2] for i in range(len(m) - 1))
        else:
            tokens.append(m.lower())
    return tokens


# ---------- 长期记忆 ----------


def list_memories() -> str:
    """返回当前有效的全部记忆，用于注入系统提示词。不给模型调用，所以不注册进 TOOLS。

    valid_to IS NULL = 还没被新值替代 = 当前真相。
    被归档的旧值物理上还在表里，但这里不返回，免得模型看到两条矛盾的记忆。
    """
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE valid_to IS NULL AND user_id = ? ORDER BY id",
            (_CURRENT_USER,),
        ).fetchall()

    if not rows:
        return "（暂无记忆）"
    return "\n".join(f"- #{r['id']} {r['content']}" for r in rows)

def _row_vec(conn: sqlite3.Connection, r: sqlite3.Row) -> Sequence[float] | None:
    """取一条记忆的存量向量；老记忆没有（列是后加的）就现场补算并回写，老数据逐渐自愈。"""
    if r["vec"] is not None:
        return _unpack_vec(r["vec"])
    vecs = _embed([r["content"]])
    if not vecs:
        return None
    conn.execute("UPDATE memories SET vec = ? WHERE id = ?", (_pack_vec(vecs[0]), r["id"]))
    return vecs[0]

def _archive_memory(
    conn: sqlite3.Connection, old: sqlite3.Row, content: str, now: str, new_vec: Sequence[float] | None
) -> str:
    """归档旧记忆并插入新值：旧值不删，只标失效 + 记下被谁替代。"""
    cur = conn.execute(
        "INSERT INTO memories (user_id, content, created_at, updated_at, vec) VALUES (?, ?, ?, ?, ?)",
        (_CURRENT_USER, content, now, now, _pack_vec(new_vec) if new_vec else None),
    )
    new_id = cur.lastrowid
    conn.execute(
        "UPDATE memories SET valid_to = ?, superseded_by = ? WHERE id = ?",
        (now, new_id, old["id"]),
    )
    conn.commit()
    return f"已更新记忆 #{old['id']} → #{new_id}：{old['content']} → {content}（旧值已归档，可查历史）"

def _judge_memory_conflict(new_content: str, candidates: list[sqlite3.Row]) -> list[int]:
    """让 LLM 判定新事实与哪些旧记忆是同一属性的新值。返回应归档的旧记忆 id；失败返回 []。"""
    lines = "\n".join(f"- id={r['id']}: {r['content']}" for r in candidates)
    prompt = (
        "下面是一条新事实和若干条同一用户的旧事实。\n"
        "判断新事实是否是某条旧事实中同一属性的新值（例如城市变了、职位变了）。\n"
        "只做事实比对：同一属性的新值才算冲突；话题相近但属性不同不算冲突。\n"
        f"旧事实：\n{lines}\n"
        f"新事实：{new_content}\n"
        "只输出 JSON 数组，形如 [3]，元素是发生冲突的旧事实 id；没有冲突输出 []。不要输出其他内容。"
    )
    try:
        resp = _llm().chat.completions.create(
            model=os.environ["OPENAI_MODEL"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        return [int(i) for i in json.loads(raw)]
    except Exception:
        return []

def remember(content: str) -> str:
    """记住一条关于用户的长期事实或偏好，跨会话保留。

    冲突消解：相似度高的旧记忆不直接覆盖 content，而是归档
    （写 valid_to + superseded_by），新值作为新行插入。
    这样既能查"曾经"，又不会让矛盾的两条在当前记忆里并存。

    只记长期稳定的信息：身份、职业、所在地、习惯、偏好、长期项目背景。
    不要记：一次性任务、documents 里能检索到的事实、
    敏感凭证、你的推测。
    """
    content = content.strip()
    if not content:
        return "记忆内容为空，未写入。"
    if len(content) > 200:
        return "这条内容太长，请压缩成一句话（200 字以内）再记。"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    key = _norm(content)
    # 新内容向量一次算好；embedding 熔断时为 None，语义路自动跳过，退回纯字面行为
    vecs = _embed([content])
    new_vec = vecs[0] if vecs else None

    with closing(_connect()) as conn:
        # 只在"当前用户的有效记忆"里查重，别把别的用户或已归档的旧值翻出来比
        rows = conn.execute(
            "SELECT id, content, vec FROM memories WHERE valid_to IS NULL AND user_id = ?",
            (_CURRENT_USER,),
        ).fetchall()

        # 完全一样：跳过
        for r in rows:
            if _norm(r["content"]) == key:
                return f"这条已经记过了（#{r['id']}）：{r['content']}"

        # 双路打分：字面 ratio + 语义余弦，先收集再决策
        scored = []
        for r in rows:
            ratio = difflib.SequenceMatcher(None, key, _norm(r["content"])).ratio()
            old_vec = _row_vec(conn, r)
            sim = _cosine(new_vec, old_vec) if new_vec and old_vec else 0.0
            scored.append((ratio, sim, r))

        # 高置信：字面或语义任一路直接超阈值，归档，不惊动 LLM
        for ratio, sim, r in scored:
            if ratio >= MEM_SIM_THRESHOLD or sim >= MEM_VEC_THRESHOLD:
                return _archive_memory(conn, r, content, now, new_vec)

        # 中置信：相似度只衡量"像不像"，衡量不了"冲不冲突"——判定权交给 LLM
        cands = [r for ratio, sim, r in scored if sim >= MEM_VEC_RECALL]
        if cands:
            hit_ids = _judge_memory_conflict(content, cands)
            for r in cands:
                if r["id"] in hit_ids:
                    return _archive_memory(conn, r, content, now, new_vec)

        if len(rows) >= MEM_MAX:
            return f"记忆已满（{MEM_MAX} 条），请先调用 forget 清理。"

        cur = conn.execute(
            "INSERT INTO memories (user_id, content, created_at, updated_at, vec) VALUES (?, ?, ?, ?, ?)",
            (_CURRENT_USER, content, now, now, _pack_vec(new_vec) if new_vec else None),
        )
        new_id = cur.lastrowid
        conn.commit()

    return f"已记住 #{new_id}：{content}"

def forget(content: str) -> str:
    """删除记忆。content 可以是原话，也可以是其中的关键词。"""
    key = _norm(content)
    if not key:
        return "请说明要删除哪条记忆。"

    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, content FROM memories WHERE user_id = ?",
            (_CURRENT_USER,),
        ).fetchall()
        # 先精确包含，再模糊相似
        hit = [r for r in rows if key in _norm(r["content"])]
        if not hit:
            hit = [
                r
                for r in rows
                if difflib.SequenceMatcher(None, key, _norm(r["content"])).ratio() >= 0.5
            ]
        if not hit:
            return f"没有找到与「{content}」相关的记忆（当前共 {len(rows)} 条）。"

        for r in hit:
            conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (r["id"], _CURRENT_USER),
            )
        conn.commit()

    return "已删除记忆：\n" + "\n".join(f"- #{r['id']} {r['content']}" for r in hit)
# ---------- 对话历史 ----------

def save_message(session_id: str, payload: dict[str, Any]) -> None:
    """存一条消息。payload 是完整 message dict，JSON 原样进出，零信息损失。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, payload, created_at) VALUES (?,?,?,?)",
            (
                session_id,
                payload.get("role", "unknown"),
                json.dumps(payload, ensure_ascii=False),
                now,
            ),
        )
        conn.commit()


def load_recent_messages(session_id: str, limit: int) -> list[dict[str, Any]]:
    """取最近 limit 条，按时间正序返回，方便直接 append 进上下文。"""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT payload FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [json.loads(r["payload"]) for r in reversed(rows)]

def clear_messages(session_id: str) -> None:
    """只清空消息，保留摘要。压缩时用这个 —— 摘要是压缩的产物，不能一起删。"""
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.commit()
def archive_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    """把被压缩掉的消息归档到 archived_messages，并建索引供后续检索式召回。

    索引两路：BM25（terms_archived 表）+ 向量（vec 列），和知识库检索同一套架构。
    只给 user / assistant 消息建索引，tool 消息（工具返回）跳过——内容太长太杂。
    embedding 失败时自动降级为纯 BM25（沿用 _embed 的熔断机制）。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with closing(_connect()) as conn:
        for m in messages:
            role = m.get("role", "unknown")
            # 跳过 system 和 tool 消息，只索引用户/助手的对话内容
            if role not in ("user", "assistant"):
                continue
            content = m.get("content") or ""
            if not content.strip():
                continue

            cur = conn.execute(
                "INSERT INTO archived_messages (session_id, role, payload, text, created_at, archived_at) "
                "VALUES (?,?,?,?,?,?)",
                (
                    session_id,
                    role,
                    json.dumps(m, ensure_ascii=False),
                    content,
                    m.get("created_at", ""),
                    now,
                ),
            )
            msg_id = cur.lastrowid
            if msg_id is None:
                continue

            # BM25 分词索引
            counts: dict[str, int] = {}
            for t in _tokenize(content):
                counts[t] = counts.get(t, 0) + 1
            conn.executemany(
                "INSERT OR REPLACE INTO terms_archived (term, msg_id, tf) VALUES (?,?,?)",
                [(t, msg_id, c) for t, c in counts.items()],
            )

            # 向量索引（失败就降级，不阻塞）
            vecs = _embed([content])
            if vecs and len(vecs) == 1:
                conn.execute(
                    "UPDATE archived_messages SET vec = ? WHERE id = ?",
                    (_pack_vec(vecs[0]), msg_id),
                )
        conn.commit()
def clear_session(session_id: str) -> None:
    """清空某个会话的历史与摘要。"""
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM archived_messages WHERE session_id = ?", (session_id,))
        conn.commit()

def clear_user(user_id: str) -> None:
    """清空某个用户的全部记忆（待办按会话隔离，不归这里管）。"""
    with closing(_connect()) as conn:
        conn.execute("DELETE FROM memories WHERE user_id = ?", (user_id,))
        conn.commit()


def get_summary(session_id: str) -> tuple[str, str]:
    """返回 (细摘要, 粗摘要)。细摘要=近期对话要点，粗摘要=跨多轮的核心事实。"""
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT fine_summary, coarse_summary FROM summaries WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    if not row:
        return "", ""
    return row["fine_summary"] or "", row["coarse_summary"] or ""


def save_summary(session_id: str, fine: str, coarse: str = "") -> None:
    """保存两层摘要。fine=细摘要(近期)，coarse=粗摘要(长期核心事实)。

    注意：旧的 summary 列仍在表里且有 NOT NULL 约束，这里顺手塞个空串占位，
    新代码不再读写它。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO summaries (session_id, fine_summary, coarse_summary, summary, updated_at) "
            "VALUES (?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "fine_summary = excluded.fine_summary, "
            "coarse_summary = excluded.coarse_summary, "
            "summary = '', "
            "updated_at = excluded.updated_at",
            (session_id, fine, coarse, "", now),
        )
        conn.commit()

# 工具的「说明书」：模型靠它决定要不要用、怎么用
TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "查询指定城市的当前实时天气（气温、天气状况、湿度、风速）。"
                "当用户问天气、气温、是否下雨、要不要带伞、穿什么衣服时必须调用它。"
                "禁止凭记忆回答任何天气问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，中文，例如：北京、上海、杭州",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todos",
            "description": (
                "把用户提到的待办事项批量保存到本地数据库，重启后依然存在。"
                "只有当用户明确要求记录、保存、记下来时才调用；"
                "用户只是让你整理、总结、分析时不要调用。"
                "一次调用保存所有条目，不要一条一条地调用。"
                "内容相同的条目会被自动跳过，无需你自己判断是否重复。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "待办事项列表，每条一句话、以动词开头，例如 ['写周报', '买牛奶']",
                    }
                },
                "required": ["items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": (
                "读取已保存的所有待办事项。"
                "当用户问「我有哪些待办」「待办做到哪了」「我之前记了什么」时调用。"
                "不要凭对话记忆回答这类问题。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
        {
        "type": "function",
        "function": {
            "name": "list_documents",
            "description": (
                "列出本地 documents 目录下所有可读取的文档及字数。"
                "当用户问「有哪些文档」「我能总结哪个文件」，"
                "或者你要读文件但不确定文件名时，先调用它。"
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "读取本地文档内容，供你总结、提炼要点、提取待办等。"
                "当用户提到某个具体文件、或让你总结/概括/提炼文件内容时调用。"
                "参数是文件名而不是完整路径；不确定文件名时先调用 list_documents。"
                "只能读 documents 目录下的 .md 和 .txt 文件。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "文件名，含扩展名，例如：会议记录.md",
                    }
                },
                "required": ["filename"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "complete_todo",
            "description": (
                "把某条待办标记为已完成。"
                "当用户说「XX 做完了」「划掉 XX」「第 N 条完成了」时调用。"
                "不要用它删除待办（那是 delete_todo），也不要用它新增待办（那是 add_todos）。"
                "编号必须来自 list_todos 的真实结果，不要凭对话记忆猜测；"
                "手上没有编号时先调用 list_todos。"
                "不要因为在回复里写了「已完成」就认为真的完成了 —— 只有调用本工具才会改变状态。"

            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {
                        "type": "integer",
                        "description": "待办编号，纯数字，例如 3",
                    }
                },
                "required": ["todo_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_todo",
            "description": (
                "彻底删除某条待办，不可恢复。"
                "只有当用户明确说「删掉」「去掉」「不需要了」时才调用；"
                "用户只是说做完了，应该调用 complete_todo。"
                "编号必须来自 list_todos 的真实结果。"
                "不要因为在回复里写了「已删除」就认为真的删除了 —— 只有调用本工具才会改变状态。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todo_id": {
                        "type": "integer",
                        "description": "待办编号，纯数字，例如 3",
                    }
                },
                "required": ["todo_id"],
            },
        },
    },
        {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在个人知识库（documents 目录下所有文档）中检索与问题相关的原文片段。"
                "**凡涉及具体数据、事实、细节的问题——金额、预算、日期、人名、指标、"
                "某句话怎么说的、某个结论是什么——都必须先调用本工具检索，"
                "不要直接回答，也不要因为用户没提「文档」「报告」二字就跳过检索。**"
                "检索到就依据片段回答并注明来自哪个文档；确实检索不到，才能说知识库里没有。"
                "与 read_document 的区别：要总结某个指定文件的整篇内容时用 read_document；"
                "要查找具体信息时用本工具。"
                "返回的是原文片段，只能依据片段回答，片段里没有的不要编造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索问题或关键词，保留关键名词，例如「预算 投入 金额」",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回几条片段，默认 5，最多 10。一般不用改。",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "keyword", "semantic"],
                        "description": (
                            "检索方式。hybrid=关键词+语义融合（默认，最准）；"
                            "keyword=纯关键词；semantic=纯语义。"
                            "只有在用户明确要求「用语义检索」「对比一下两种检索」时才指定，"
                            "平时不要传。"
                        ),
                    },

                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "把关于用户的一条长期事实或偏好写入记忆，跨会话保留。"
                "当用户说「记住…」「以后都…」「别忘了…」，"
                "或主动说出稳定的个人信息（身份、职业、所在地、习惯、偏好）时调用。"
                "**不要记**：一次性任务、documents 里检索得到的内容、"
                "敏感凭证（密码/卡号/证件号）、你的推测——存疑时不记。"
                "内容相近的旧记忆会被自动更新，你不必自己判断是否重复。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要记住的内容，一句话，例如「用户在杭州工作」",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": (
                "删除一条已记住的内容。"
                "当用户说「忘掉…」「别记了」「那个不准」时调用。"
                "参数是原话或其中的关键词，不需要精确的完整句子。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "要删除的记忆原文或关键词，例如「杭州」",
                    }
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retrieve_history",
            "description": (
                "在你之前的聊天历史中检索相关内容。"
                "**当用户问到「我们之前聊过什么」「刚才那个 XX 是多少」"
                "「你还记得我之前说的 XX 吗」，或任何引用了早期对话但"
                "当前上下文里已找不到的内容时，必须先调用本工具。**"
                "不要凭对话记忆回答——早期对话已被压缩，只有检索回来才是真实内容。"
                "与 search_knowledge 的区别：search_knowledge 查 documents 知识库，"
                "本工具查你和用户的聊天历史。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索关键词，保留核心名词，例如「杭州 工作 预算」",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回几条，默认 5，最多 10",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_goal",
            "description": (
                "创建一个长期目标。"
                "当用户说出长期成果类请求（不是单步动作），例如「我想这周读完 X」「我要减抖10斤」时调。\n"
                "与 add_todos 区别：add_todos 是单步动作，create_goal 是跨多次对话推进的成果。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "目标描述，例如：这周读完《活着》",
                    },
                    "success_criteria": {
                        "type": "string",
                        "description": "可验证的达成标准，例如：读完最后一页并给感想",
                    },
                },
                "required": ["description", "success_criteria"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "advance_goal",
            "description": (
                "推进目标。每次对话开头/结尾调一次。\n"
                "next_step 是关键——写下下次该干啥，让未来的自己接得上。\n"
                "状态自动从 pending→in_progress→on_track 流转。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "integer", "description": "目标 id"},
                    "action": {"type": "string", "description": "本次做了什么"},
                    "result": {"type": "string", "description": "本次拿到什么结果"},
                    "next_step": {"type": "string", "description": "下次对话该干啥"},
                },
                "required": ["goal_id", "action", "result", "next_step"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_goal",
            "description": (
                "查长期目标。不传 goal_id 返回所有进行中目标；传 id 返回单个详情含进度日志。\n"
                "本工具是只读的，可并发。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {
                        "type": "integer",
                        "description": "目标 id。不传返回全部进行中目标。",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_goal_status",
            "description": (
                "显式状态流转：done（完成）/ blocked（卡住）/ abandoned（放弃）。\n"
                "pending/in_progress/on_track 由 advance_goal 自动管理，不在此处理。\n"
                "只有达成了 success_criteria 或确实卡住才调。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "goal_id": {"type": "integer", "description": "目标 id"},
                    "status": {"type": "string", "enum": ["done", "blocked", "abandoned"]},
                    "reason": {"type": "string", "description": "状态变更原因，可选"},
                },
                "required": ["goal_id", "status"],
            },
        },
    },
]


# search_knowledge 定义在 rag.py。用懒加载包装，避免 tools ↔ rag 循环导入：
# 只有真正调用时才 import rag，无论 tools 和 rag 谁先被 import 都能工作。
def _search_knowledge_lazy(*args: Any, **kwargs: Any) -> str:
    from rag import search_knowledge
    return search_knowledge(*args, **kwargs)
def _retrieve_history_lazy(*args: Any, **kwargs: Any) -> str:
    from rag import retrieve_history
    return retrieve_history(*args, **kwargs)

# 名字 → 真实函数。调度时用它查表执行。
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "add_todos": add_todos,
    "list_todos": list_todos,
    "list_documents": list_documents,
    "read_document": read_document,
    "complete_todo": complete_todo,
    "delete_todo": delete_todo,
    "search_knowledge": _search_knowledge_lazy,
    "remember": remember,
    "create_goal": create_goal,
    "advance_goal": advance_goal,
    "get_goal": get_goal,
    "update_goal_status": update_goal_status,
    "forget": forget,
    "retrieve_history": _retrieve_history_lazy,
}
# 可以并发执行的工具：只读或纯网络 I/O，彼此无状态冲突。
# 写数据库的工具串行执行，避免 sqlite 锁竞争。
READ_ONLY_TOOLS = frozenset(
    {"get_weather", "list_todos", "list_documents", "read_document", "search_knowledge","retrieve_history", "get_goal"}
)


