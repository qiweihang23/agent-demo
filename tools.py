"""工具定义与实现。

约定：每个工具返回一个「给模型看的字符串」。
不要返回 dict —— 模型要的是自然语言结果，它自己会组织语言。
"""
from __future__ import annotations
from datetime import datetime
import os
import re
from collections.abc import Callable
import sqlite3
from contextlib import closing
from pathlib import Path
import httpx
from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam
load_dotenv()
GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DATA_DIR = Path(__file__).parent / "data"
DB_PATH = DATA_DIR / "assistant.db"
DOCS_DIR = Path(__file__).parent / "documents"
SUPPORTED_SUFFIX = {".md", ".txt"}
FULL_TEXT_LIMIT = 8000  # 超过这个字数就不再返回全文，改走分块摘要
CHUNK_SIZE = 3000       # 每块字符数
CHUNK_OVERLAP = 200     # 块间重叠字数，避免把一段话切碎
MAX_CHUNKS = 20         # 上限：20 * 3000 = 6 万字，超过就如实拒绝
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
                text       TEXT    NOT NULL,
                done       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL
            )
            """
        )
        conn.commit()
_init_db()  # 导入时就确保表存在
# 添加待办
def _norm(text: str) -> str:
    """归一化待办文本：去掉所有空白，用于判断是否重复。"""
    return re.sub(r"\s+", "", text)

def add_todos(items: list[str]) -> str:
    """批量添加待办。已存在的条目自动跳过，不重复添加。"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    added: list[str] = []
    skipped: list[str] = []

    with closing(_connect()) as conn:
        exist = {
            _norm(r["text"]): {"id": r["id"]}
            for r in conn.execute("SELECT id, text FROM todos").fetchall()
        }
        for text in items:
            key = _norm(text)
            if not key:
                continue
            if key in exist:
                skipped.append(f"{text}（已存在 #{exist[key]['id']}）")
                continue
            cur = conn.execute(
                "INSERT INTO todos (text, done, created_at) VALUES (?, ?, ?)",
                (text, 0, now),
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
            f"跳过 {len(skipped)} 条（已有相同内容，未重复添加）：\n"
            + "\n".join(skipped)
        )
    return "\n".join(blocks) if blocks else "没有需要添加的新待办（全部已存在）。"
#标记待办为已完成
def complete_todo(todo_id: int) -> str:
    """把某条待办标记为已完成。"""
    todo_id = int(todo_id)  # 模型偶尔会传字符串，容错一下
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, text FROM todos WHERE id = ?", (todo_id,)
        ).fetchone()
        if row is None:
            return f"没有编号为 #{todo_id} 的待办。请先调用 list_todos 确认编号。"
        conn.execute("UPDATE todos SET done = 1 WHERE id = ?", (todo_id,))
        conn.commit()
    return f"已完成 #{todo_id} {row['text']}。"

#彻底删除待办
def delete_todo(todo_id: int) -> str:
    """彻底删除某条待办。"""
    todo_id = int(todo_id)
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT id, text FROM todos WHERE id = ?", (todo_id,)
        ).fetchone()
        if row is None:
            return f"没有编号为 #{todo_id} 的待办。请先调用 list_todos 确认编号。"
        conn.execute("DELETE FROM todos WHERE id = ?", (todo_id,))
        conn.commit()
    return f"已删除 #{todo_id} {row['text']}。"

# 列出待办
def list_todos() -> str:
    """列出所有待办，未完成在前。"""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, text, done, created_at FROM todos ORDER BY done, id"
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
#列出文档
def list_documents() -> str:
    """列出 documents/ 下可读取的文档及字数。"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in DOCS_DIR.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIX
    )
    if not files:
        return "documents/ 目录下还没有可读取的文档（目前只支持 .md 和 .txt）。"

    lines = []
    for p in files:
        size = len(_read_text(p))
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
        return f"暂不支持 {path.suffix} 文件，目前只支持 .md 和 .txt。"
    if not path.exists():
        return f"没有找到「{filename}」。\n{list_documents()}"

    text = _read_text(path)
    if not text.strip():
        return f"文档「{path.name}」是空的。"
    if len(text) > FULL_TEXT_LIMIT:
        return _summarize_long(text, path.name)

    return f"文档《{path.name}》全文（{len(text)} 字）：\n\n{text}"

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
]


# 名字 → 真实函数。调度时用它查表执行。
TOOL_FUNCTIONS: dict[str, Callable[..., str]] = {
    "get_weather": get_weather,
    "add_todos": add_todos,
    "list_todos": list_todos,
    "list_documents": list_documents,
    "read_document": read_document,
    "complete_todo": complete_todo,
    "delete_todo": delete_todo,
}
# 可以并发执行的工具：只读或纯网络 I/O，彼此无状态冲突。
# 写数据库的工具串行执行，避免 sqlite 锁竞争。
READ_ONLY_TOOLS = frozenset(
    {"get_weather", "list_todos", "list_documents", "read_document"}
)


