"""服务化入口：把 Agent 的事件流通过 SSE 推给前端。

启动：
    uv run python server.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
import structlog
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import Agent, load_system_prompt
from mcp_server import mcp
from mcp_client import merged_tools, get_shared_hub, close_shared_hub, external_tools_prompt
from tracing import recent_traces, get_trace

# agent.py 默认把日志压到 WARNING（怕污染 CLI）；服务模式下要看 span 流水，提到 INFO
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO)
)
log = structlog.get_logger()

# 先构建一次 streamable HTTP 子应用：副作用是初始化 mcp._session_manager。
# 之后 mcp.session_manager 才能用在 lifespan 里——FastMCP 的 session_manager
# 属性在 streamable_http_app() 被调用前访问会抛 RuntimeError。多次调用会返回
# 不同的 Starlette 实例但共享同一个 session manager，所以挂这一个就够。
_MCP_HTTP_APP = mcp.streamable_http_app()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """托管 MCP 会话管理器的生命周期。

    为什么必须手动接管：streamable_http_app() 自带的 lifespan 只有在它作为
    「顶层 app」被 uvicorn 直接运行时才会触发；现在它被 mount 成子应用，
    Starlette 的 Router.lifespan 只跑顶层 lifespan，不会把 lifespan 事件转发给
    挂载的子应用——session manager 不会自己起来，MCP 调用会挂死。
    所以这里 startup 时进入 run()，shutdown 时退出，手动接管。

    外部 MCP server（飞书/fetch 等）也在启动时连接：get_shared_hub 是同步阻塞
    调用（拉起子进程），用 to_thread 丢到线程池，避免卡住 event loop；
    连不上的 server 会被 MCPToolHub 自动跳过，不影响 Web 服务本身。
    """
    async with mcp.session_manager.run():
        hub = await asyncio.to_thread(get_shared_hub)
        log.info("mcp_hub_ready", external_tools=len(hub.tool_names) if hub else 0)
        try:
            yield
        finally:
            close_shared_hub()


app = FastAPI(title="agent-demo", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def access_timing(request: Request, call_next):  # type: ignore[no-untyped-def]
    """给每个 HTTP 请求记一条耗时日志 + 响应头。

    注意：SSE 接口在 call_next 返回响应头时就计时结束了，body 要流很久——
    流式内部各阶段的真实耗时看 /traces 里的 span，不看这里。
    """
    t0 = time.perf_counter()
    resp = await call_next(request)
    ms = round((time.perf_counter() - t0) * 1000, 1)
    resp.headers["X-Response-Time-ms"] = str(ms)
    log.info("http", method=request.method, path=request.url.path, status=resp.status_code, ms=ms)
    return resp

class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str = "default"
    user_id: str = "default"

# Agent 实例里揣着 self.messages 这份会话状态，所以按 session 缓存实例，
# 不能每个请求新建一次（否则多轮对话全丢）。
_agents: dict[str, Agent] = {}
# 同一个 session 的请求必须串行：messages 列表不是线程安全的，
# 两个请求同时写会互相串话。不同 session 各持一把锁，互不阻塞。
_locks: dict[str, threading.Lock] = {}
_lock_guard = threading.Lock()

def _get_agent(session_id: str, user_id: str) -> tuple[Agent, threading.Lock]:
    """按 session 取（或首次创建）Agent 实例和它的专属锁。"""
    with _lock_guard:
        agent = _agents.get(session_id)
        if agent is None:
            # hub 已在 lifespan 启动；这里把本地 + 外部 MCP 工具一起注入
            tools, tool_functions = merged_tools()
            agent = Agent(
                system_prompt=load_system_prompt(session_id) + external_tools_prompt(),
                session_id=session_id,
                user_id=user_id,
                tools=tools,
                tool_functions=tool_functions,
            )
            _agents[session_id] = agent
            _locks[session_id] = threading.Lock()
        return agent, _locks[session_id]

def _sse_lines(agent: Agent, message: str) -> Iterator[str]:
    """把 chat_stream 的事件 dict 逐个包成 SSE 帧：data: {...}\\n\\n"""
    try:
        for ev in agent.chat_stream(message):
            yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
    except Exception as e:
        # 流走到一半报错（如上游 API 超时）也必须给客户端一个明确收尾，
        # 不能让浏览器那边傻等到超时。
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/traces")
def traces(limit: int = 20) -> dict[str, list[dict[str, object]]]:
    """最近对话的链路摘要（新的在前）。"""
    return {"traces": recent_traces(min(max(limit, 1), 100))}


@app.get("/traces/{trace_id}")
def trace_detail(trace_id: str) -> dict[str, object]:
    """单条链路详情：全部 span 的父子关系、耗时和属性。"""
    detail = get_trace(trace_id)
    if detail is None:
        raise HTTPException(status_code=404, detail="trace 不存在或已被挤出缓冲")
    return detail

@app.post("/chat")
def chat(req: ChatRequest) -> dict[str, str]:
    """非流式接口：等整轮跑完一次性返回，给普通脚本调用用。"""
    agent, lock = _get_agent(req.session_id, req.user_id)
    with lock:
        return {"reply": agent.chat(req.message)}

def _stream_response(agent: Agent, lock: threading.Lock, message: str) -> StreamingResponse:
    def gen() -> Iterator[str]:
        # 锁在生成器内部获取：要等框架真正开始迭代（客户端已连上）才持有，
        # 避免请求排队时白占锁。整轮流式期间同 session 其他请求等待。
        with lock:
            yield from _sse_lines(agent, message)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        # no-cache：防中间代理缓冲；X-Accel-Buffering：关 nginx 缓冲，
        # 否则事件会被攒成一坨再发，流式就退化成非流式了。
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    """正式流式接口，JSON body 传参。"""
    agent, lock = _get_agent(req.session_id, req.user_id)
    return _stream_response(agent, lock, req.message)

@app.get("/chat/stream")
def chat_stream_get(
    message: str, session_id: str = "default", user_id: str = "default"
) -> StreamingResponse:
    """GET 版流式接口：浏览器 EventSource / 地址栏只能发 GET，专供肉眼调试。

    直接在浏览器打开就能看到事件一段段往下刷：
        http://127.0.0.1:8000/chat/stream?message= 查一下杭州天气
    """
    if not message.strip():
        raise HTTPException(status_code=400, detail="message 不能为空")
    agent, lock = _get_agent(session_id, user_id)
    return _stream_response(agent, lock, message)


# MCP over Streamable HTTP：远程客户端（Trae / Claude / 其他 agent）经
# http://127.0.0.1:8000/mcp/mcp 调用 todos/weather 等工具。必须挂在 StaticFiles（/）
# 之前，否则 /mcp/... 会被静态文件兜底吞掉。路径是「挂载点 /mcp」+「子应用内部
# 路由 /mcp」= /mcp/mcp，看着重复但这是避免 trailing-slash 307 重定向的标准挂法
# （POST 请求不会被重定向截断，MCP 客户端更稳）。
app.mount("/mcp", _MCP_HTTP_APP)

# 前端页面必须最后挂载：Starlette 按注册顺序匹配，放这里不会挡住上面的
# /health、/chat、/traces 等 API；html=True 让访问 / 时直接返回 index.html。
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="web")

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000)