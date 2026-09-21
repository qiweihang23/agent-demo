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
KB_CHUNK_SIZE = 500        # 检索块比摘要块小得多：要的是定位精度，不是覆盖面
KB_CHUNK_OVERLAP = 80
KB_TOP_K = 5               # 只把最相关的几条塞进上下文
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_BATCH = 10      # 单次请求的文本条数上限，DashScope 有批量限制
RRF_K = 60            # RRF 系数，越大越"抹平"两路排名差异
VEC_MIN_SCORE = 0.25   # 余弦低于此值视为不相关，不参与融合
KB_MIN_POOL = 5   # 某一路命中少于此数，视为该路对本次查询失效，不参与融合
MEM_SIM_THRESHOLD = 0.75  # 相似度高于此值视为同一条记忆，更新而非新增
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
                text       TEXT    NOT NULL,
                done       INTEGER NOT NULL DEFAULT 0,
                created_at TEXT    NOT NULL
            )
            """
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
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                content    TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL
            )
            """
        )
                # 老库升级：chunks 表可能还没有 vec 列（SQLite 不支持 ADD COLUMN IF NOT EXISTS）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(chunks)")}
        if "vec" not in cols:
            conn.execute("ALTER TABLE chunks ADD COLUMN vec BLOB")

        conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_name)")
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


def _kb_chunks(text: str) -> list[str]:
    """按字符滑窗切块，块间保留重叠，避免把关键句切断。"""
    step = KB_CHUNK_SIZE - KB_CHUNK_OVERLAP
    chunks: list[str] = []
    for i in range(0, len(text), step):
        piece = text[i : i + KB_CHUNK_SIZE].strip()
        if piece:
            chunks.append(piece)
        if i + KB_CHUNK_SIZE >= len(text):
            break
    return chunks


def _drop_doc(conn: sqlite3.Connection, name: str) -> None:
    """清掉某文档的索引（重建前 / 文件已删除时）。"""
    ids = [r[0] for r in conn.execute("SELECT id FROM chunks WHERE doc_name=?", (name,))]
    if not ids:
        return
    conn.execute(f"DELETE FROM terms WHERE chunk_id IN ({','.join('?' * len(ids))})", ids)
    conn.execute("DELETE FROM chunks WHERE doc_name=?", (name,))


def _index_doc(conn: sqlite3.Connection, name: str, text: str, mtime: float) -> int:
    """重建单个文档的索引：先删旧，再插新，最后补向量。"""
    _drop_doc(conn, name)
    chunks = _kb_chunks(text)
    ids: list[int] = []

    for idx, chunk in enumerate(chunks):
        cur = conn.execute(
            "INSERT INTO chunks (doc_name, chunk_idx, text, mtime) VALUES (?,?,?,?)",
            (name, idx, chunk, mtime),
        )
        cid = cur.lastrowid
        if cid is None:  # 插入未返回 rowid，跳过，避免后面拿到 None
            continue
        ids.append(cid)
        counts: dict[str, int] = {}
        for t in _tokenize(chunk):
            counts[t] = counts.get(t, 0) + 1
        conn.executemany(
            "INSERT OR REPLACE INTO terms (term, chunk_id, tf) VALUES (?,?,?)",
            [(t, cid, c) for t, c in counts.items()],
        )

    vecs = _embed(chunks)
    if vecs and len(vecs) == len(ids):
        conn.executemany(
            "UPDATE chunks SET vec=? WHERE id=?",
            [(_pack_vec(v), cid) for v, cid in zip(vecs, ids)],
        )
    return len(ids)



def _ensure_index(conn: sqlite3.Connection) -> None:
    """让索引跟磁盘保持一致：改动过的重建，已删除的清掉。靠 mtime 判断。"""
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    on_disk = {
        p.name: p.stat().st_mtime
        for p in DOCS_DIR.iterdir()
        if p.suffix.lower() in SUPPORTED_SUFFIX
    }
    indexed = {
        r["doc_name"]: r["mtime"]
        for r in conn.execute(
            "SELECT doc_name, MAX(mtime) AS mtime FROM chunks GROUP BY doc_name"
        )
    }

    for name in set(indexed) - set(on_disk):
        _drop_doc(conn, name)

    for name, mtime in on_disk.items():
        if indexed.get(name) != mtime:
            _index_doc(conn, name, _read_text(DOCS_DIR / name), mtime)
        # 老索引（加 vec 列之前建的）没有向量，补齐；embedding 不可用时 _embed 返回 None，直接跳过
    todo = conn.execute("SELECT id, text FROM chunks WHERE vec IS NULL").fetchall()
    if todo:
        vecs = _embed([r["text"] for r in todo])
        if vecs and len(vecs) == len(todo):
            conn.executemany(
                "UPDATE chunks SET vec=? WHERE id=?",
                [(_pack_vec(v), r["id"]) for v, r in zip(vecs, todo)],
            )

    conn.commit()

def _bm25_scores(conn: sqlite3.Connection, qterms: list[str]) -> dict[int, float]:
    """BM25 打分。-- 原 search_knowledge 里的打分逻辑，抽出来好跟向量路平级。"""
    if not qterms:
        return {}
    ph = ",".join("?" * len(qterms))
    total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    df = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT term, COUNT(*) FROM terms WHERE term IN ({ph}) GROUP BY term",
            qterms,
        )
    }
    avgdl = conn.execute("SELECT AVG(LENGTH(text)) FROM chunks").fetchone()[0] or 1.0
    rows = conn.execute(
        f"""
        SELECT c.id, t.term, t.tf, LENGTH(c.text) AS dl
        FROM terms t JOIN chunks c ON c.id = t.chunk_id
        WHERE t.term IN ({ph})
        """,
        qterms,
    ).fetchall()

    k1, b = 1.5, 0.75
    scores: dict[int, float] = {}
    for r in rows:
        d = df.get(r["term"], 0)
        idf = math.log(1 + (total - d + 0.5) / (d + 0.5)) if d else 0.0
        tf, dl = r["tf"], r["dl"] or 1
        scores[r["id"]] = scores.get(r["id"], 0.0) + idf * (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * dl / avgdl)
        )
    return scores


def _vec_scores(conn: sqlite3.Connection, query: str) -> dict[int, float]:
    """向量路：query 整句向量化，与所有 chunk 算余弦。不做分词 —— 这正是它的优势。"""
    rows = conn.execute("SELECT id, vec FROM chunks WHERE vec IS NOT NULL").fetchall()
    if not rows:
        return {}
    q = _embed([query])
    if not q:
        return {}
    qv = q[0]
    return {
        r["id"]: s
        for r in rows
        if (s := _cosine(qv, _unpack_vec(r["vec"]))) >= VEC_MIN_SCORE
    }


def _rrf(pools: list[dict[int, float]], k: int = RRF_K) -> dict[int, float]:
    """RRF 融合：只看名次不看分数，绕开两路量纲不可比的问题。"""
    fused: dict[int, float] = {}
    for scores in pools:
        for rank, cid in enumerate(sorted(scores, key=lambda c: -scores[c]), 1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused

def search_knowledge(query: str, top_k: int = KB_TOP_K, mode: str = "hybrid") -> str:
    """在个人知识库里检索：关键词(BM25) + 语义(embedding)，RRF 融合取 Top-K。

    mode: hybrid(默认，两路融合) / keyword(纯关键词) / semantic(纯语义)。
    """
    top_k = max(1, min(int(top_k), 10))
    mode = (mode or "hybrid").lower()
    if mode not in {"hybrid", "keyword", "semantic"}:
        mode = "hybrid"

    with closing(_connect()) as conn:
        _ensure_index(conn)

        total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        if total == 0:
            return "知识库是空的：documents/ 目录下还没有可索引的 .md / .txt 文档。"

        meta = {
            r["id"]: (r["doc_name"], r["text"])
            for r in conn.execute("SELECT id, doc_name, text FROM chunks")
        }
        qterms = sorted(set(_tokenize(query)))

        bm25 = {} if mode == "semantic" else _bm25_scores(conn, qterms)
        vec = {} if mode == "keyword" else _vec_scores(conn, query)

        if mode == "keyword":
            fused = bm25
        elif mode == "semantic":
            fused = vec
        else:
            # 某一路候选太少 = 它对这个查询基本失效，它的"第 1 名"含金量很低，
            # 参与融合只会污染结果（RRF 会给它 1/(k+1) 的最高档分数）。
            pools = [p for p in (bm25, vec) if len(p) >= KB_MIN_POOL]
            fused = _rrf(pools) if len(pools) > 1 else (bm25 or vec)

        with closing(_connect()) as conn:
            n_vec = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE vec IS NOT NULL"
            ).fetchone()[0]
        print(
            f"  [检索] 模式={mode}｜关键词命中 {len(bm25)} 条，语义候选 {len(vec)} 条"
            f"｜知识库 {len(meta)} 块，其中 {n_vec} 块已向量化"
        )


    if not fused:
        return (
            f"没有找到与「{query}」相关的文档内容。"
            f"可以换个更贴近文档用词的说法再试，或调用 list_documents 确认有哪些文档。"
        )

    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    blocks = []
    for i, (cid, s) in enumerate(ranked, 1):
        doc, text = meta[cid]
        hits = "+".join(
            t for t, pool in (("关键词", bm25), ("语义", vec)) if cid in pool
        ) or "无"
        blocks.append(
            f"[{i}] 来自《{doc}》｜融合分 {s:.4f}｜命中：{hits}\n{text}"
        )

    return (
        f"共 {len(fused)} 个候选片段，以下是最相关的 {len(ranked)} 条：\n\n"
        + "\n\n".join(blocks)
        + "\n\n（以上是知识库中的原文片段，是回答的唯一依据。"
        "片段里没有的信息，请如实说明文档中未提到，不要用自己的知识补充。）"
    )

# ---------- 长期记忆 ----------


def list_memories() -> str:
    """返回全部记忆，用于注入系统提示词。不给模型调用，所以不注册进 TOOLS。"""
    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, content FROM memories ORDER BY id"
        ).fetchall()

    if not rows:
        return "（暂无记忆）"
    return "\n".join(f"- #{r['id']} {r['content']}" for r in rows)


def remember(content: str) -> str:
    """记住一条关于用户的长期事实或偏好，跨会话保留。

    只记长期稳定的信息：身份、职业、所在地、习惯、偏好、长期项目背景。
    不要记：一次性任务（如"查天气"）、documents 里能检索到的事实、
    敏感凭证（密码、卡号、证件号）、你的推测。
    """
    content = content.strip()
    if not content:
        return "记忆内容为空，未写入。"
    if len(content) > 200:
        return "这条内容太长，请压缩成一句话（200 字以内）再记。"

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    key = _norm(content)

    with closing(_connect()) as conn:
        rows = conn.execute("SELECT id, content FROM memories").fetchall()

        # 完全一样：直接跳过
        for r in rows:
            if _norm(r["content"]) == key:
                return f"这条已经记过了（#{r['id']}）：{r['content']}"

        # 高度相似 = 同一件事的新说法，更新旧的，绝不并存两条矛盾的
        for r in rows:
            ratio = difflib.SequenceMatcher(None, key, _norm(r["content"])).ratio()
            if ratio >= MEM_SIM_THRESHOLD:
                conn.execute(
                    "UPDATE memories SET content = ?, updated_at = ? WHERE id = ?",
                    (content, now, r["id"]),
                )
                conn.commit()
                return f"已更新记忆 #{r['id']}：{r['content']} → {content}"

        if len(rows) >= MEM_MAX:
            return f"记忆已满（{MEM_MAX} 条），请先调用 forget 清理。"

        cur = conn.execute(
            "INSERT INTO memories (content, created_at, updated_at) VALUES (?, ?, ?)",
            (content, now, now),
        )
        conn.commit()
        new_id = cur.lastrowid

    return f"已记住 #{new_id}：{content}"


def forget(content: str) -> str:
    """删除记忆。content 可以是原话，也可以是其中的关键词。"""
    key = _norm(content)
    if not key:
        return "请说明要删除哪条记忆。"

    with closing(_connect()) as conn:
        rows = conn.execute("SELECT id, content FROM memories").fetchall()
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
            conn.execute("DELETE FROM memories WHERE id = ?", (r["id"],))
        conn.commit()

    return "已删除记忆：\n" + "\n".join(f"- #{r['id']} {r['content']}" for r in hit)

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
    "search_knowledge": search_knowledge,
    "remember": remember,
    "forget": forget,
}
# 可以并发执行的工具：只读或纯网络 I/O，彼此无状态冲突。
# 写数据库的工具串行执行，避免 sqlite 锁竞争。
READ_ONLY_TOOLS = frozenset(
    {"get_weather", "list_todos", "list_documents", "read_document", "search_knowledge"}
)


