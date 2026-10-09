"""RAG 检索引擎：文档索引 + BM25 + 向量 + RRF 融合。

从 tools.py 拆出来的检索专用模块。embedding 原语（_embed / _tokenize /
_cosine / _pack_vec 等）仍在 tools.py，因为 archive_messages 也要用，
放这里会形成循环依赖。
"""
from __future__ import annotations

import math
import sqlite3
from contextlib import closing

from tools import (
    DOCS_DIR,
    SUPPORTED_SUFFIX,
    _connect,
    _cosine,
    _embed,
    _extract_text,
    _pack_vec,
    _tokenize,
    _unpack_vec,
)

KB_CHUNK_SIZE = 500        # 检索块比摘要块小得多：要的是定位精度，不是覆盖面
KB_CHUNK_OVERLAP = 80
KB_TOP_K = 5               # 只把最相关的几条塞进上下文
RRF_K = 60            # RRF 系数，越大越"抹平"两路排名差异
VEC_MIN_SCORE = 0.25   # 余弦低于此值视为不相关，不参与融合
KB_MIN_POOL = 5   # 某一路命中少于此数，视为该路对本次查询失效，不参与融合


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
            _index_doc(conn, name, _extract_text(DOCS_DIR / name), mtime)
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




def _bm25_scores(
    conn: sqlite3.Connection, qterms: list[str],
    table: str = "chunks", terms_table: str = "terms", id_col: str = "chunk_id", text_col: str = "text",
) -> dict[int, float]:
    """BM25 打分。table=目标表，terms_table=倒排表，id_col=倒排表中关联目标表的列名。"""
    if not qterms:
        return {}
    ph = ",".join("?" * len(qterms))
    total = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    df = {
        r[0]: r[1]
        for r in conn.execute(
            f"SELECT term, COUNT(*) FROM {terms_table} WHERE term IN ({ph}) GROUP BY term",
            qterms,
        )
    }
    avgdl = conn.execute(f"SELECT AVG(LENGTH({text_col})) FROM {table}").fetchone()[0] or 1.0
    rows = conn.execute(
        f"""
        SELECT t.{id_col} AS cid, t.term, t.tf, LENGTH(c.{text_col}) AS dl
        FROM {terms_table} t JOIN {table} c ON c.id = t.{id_col}
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
        scores[r["cid"]] = scores.get(r["cid"], 0.0) + idf * (tf * (k1 + 1)) / (
            tf + k1 * (1 - b + b * dl / avgdl)
        )
    return scores

def _vec_scores(
    conn: sqlite3.Connection, query: str,
    table: str = "chunks",
) -> dict[int, float]:
    """向量路：query 整句向量化，与 table 中所有有 vec 的行算余弦。"""
    rows = conn.execute(f"SELECT id, vec FROM {table} WHERE vec IS NOT NULL").fetchall()
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
def retrieve_history(query: str, top_k: int = 5, mode: str = "hybrid") -> str:
    """在被压缩归档的历史对话中检索与问题相关的片段。

    当用户问到之前聊过、但已不在当前上下文中的内容时调用。
    只检索当前会话的归档消息（user / assistant），返回相关原文片段。
    与 search_knowledge 的区别：search_knowledge 查 documents 知识库，
    本工具查用户自己的聊天历史。
    """
    import tools  # 取当前会话 id，避免循环导入
    session_id = tools._CURRENT_SESSION

    top_k = max(1, min(int(top_k), 10))
    mode = (mode or "hybrid").lower()
    if mode not in {"hybrid", "keyword", "semantic"}:
        mode = "hybrid"

    with closing(_connect()) as conn:
        rows = conn.execute(
            "SELECT id, role, text FROM archived_messages "
            "WHERE session_id = ? AND text IS NOT NULL AND text != ''",
            (session_id,),
        ).fetchall()
        if not rows:
            return "（没有可检索的归档历史——本轮对话还没产生被压缩的早期内容。）"

        meta = {r["id"]: (r["role"], r["text"]) for r in rows}
        qterms = sorted(set(_tokenize(query)))

        # 双路打分，目标表是 archived_messages / terms_archived
        bm25 = {} if mode == "semantic" else _bm25_scores(
            conn, qterms,
            table="archived_messages", terms_table="terms_archived",
            id_col="msg_id", text_col="text",
        )
        vec = {} if mode == "keyword" else _vec_scores(conn, query, table="archived_messages")

        # 只保留属于当前会话的命中（打分函数没按 session 过滤，这里收敛）
        bm25 = {cid: s for cid, s in bm25.items() if cid in meta}
        vec = {cid: s for cid, s in vec.items() if cid in meta}

        if mode == "keyword":
            fused = bm25
        elif mode == "semantic":
            fused = vec
        else:
            pools = [p for p in (bm25, vec) if len(p) >= KB_MIN_POOL]
            fused = _rrf(pools) if len(pools) > 1 else (bm25 or vec)

        print(
            f"  [历史检索] 模式={mode}｜关键词命中 {len(bm25)} 条，语义候选 {len(vec)} 条"
            f"｜归档 {len(meta)} 条"
        )

    if not fused:
        return f"归档历史里没有找到与「{query}」相关的内容。"

    ranked = sorted(fused.items(), key=lambda x: -x[1])[:top_k]
    blocks = []
    for i, (mid, s) in enumerate(ranked, 1):
        role, text = meta[mid]
        who = "用户" if role == "user" else "助手"
        blocks.append(f"[{i}] {who}（相关度 {s:.4f}）\n{text}")

    return (
        f"从归档历史中找到 {len(ranked)} 条相关片段：\n\n"
        + "\n\n".join(blocks)
        + "\n\n（以上是你之前聊过的原文，仅供回答参考。如果和当前问题无关，请忽略。）"
    )