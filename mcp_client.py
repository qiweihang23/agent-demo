"""MCP Client 适配层：连接外部 MCP Server，把它们的工具变成 Agent 可直接注入的工具。

这是「接生态」的一半：mcp_server.py 是把我们的工具暴露给别人（Server），
本文件反过来——去连别人的 Server，把外部工具拉进我们的 Agent（Client）。

为什么需要后台线程 + event loop：
  MCP Python SDK 的 ClientSession 全是 async，而 Agent 调用工具是同步函数。
  ClientSession 还必须在「创建它的那个 event loop」里使用，不能每次调用临时
  asyncio.run()（loop 不同，连接直接报废）。所以开一个守护线程跑一条持久
  event loop，长连接和 session 都活在里面；同步侧用 run_coroutine_threadsafe
  把调用投递进去，再阻塞等结果。对 Agent 而言，外部工具和本地工具调用方式
  完全一样，没有 async 概念。

数据流：
  Agent.chat()  ──同步调用──►  callable(kwargs)
                                   │ run_coroutine_threadsafe
                                   ▼
                       后台 loop：ClientSession.call_tool
                                   │ JSON-RPC（stdio 子进程 / HTTP）
                                   ▼
                            外部 MCP Server
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "mcp_servers.json"
CONNECT_TIMEOUT = 30.0   # 拉起子进程 + initialize 的上限
CALL_TIMEOUT = 60.0      # 单次工具调用上限，防止外部 server 挂死拖垮 Agent

_NAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize(name: str) -> str:
    """OpenAI 工具名只允许 [A-Za-z0-9_-]，server/tool 名统一清洗。"""
    return _NAME_RE.sub("_", name)


def default_config() -> dict[str, dict[str, Any]]:
    """零配置默认：用 stdio 自连本项目的 mcp_server.py，用来验证 client 链路。

    接真实生态时在 mcp_servers.json 里加条目即可，格式：
    {
      "some-server": {"transport": "stdio", "command": ["npx", "-y", "..."]},
      "remote-api":  {"transport": "http",  "url": "http://host/mcp"}
    }
    """
    return {
        "agent-demo": {
            "transport": "stdio",
            "command": [sys.executable, str(ROOT / "mcp_server.py")],
        }
    }


def load_config() -> dict[str, dict[str, Any]]:
    """读 mcp_servers.json；不存在就用内置默认（stdio 自连）。"""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return default_config()


class _ServerConn:
    """一条到某个 MCP Server 的长连接及其上下文管理器。

    长连接不能用 async with（出作用域就断开），所以手动管理 __aenter__/__aexit__：
    start 时进入并保活，close 时逆序退出。
    """

    def __init__(self, name: str, cfg: dict[str, Any]):
        self.name = name
        self.cfg = cfg
        self.session: ClientSession | None = None
        self.tool_index: dict[str, Any] = {}  # 原始工具名 -> mcp Tool（含描述/schema）
        self._transport_cm = None   # stdio_client / streamablehttp_client 的 CM
        self._session_cm = None     # ClientSession 的 CM

    async def open(self) -> list:
        """建立连接、initialize、拉取工具清单，返回 mcp Tool 列表。"""
        transport = self.cfg.get("transport", "stdio")
        if transport == "stdio":
            command = self.cfg["command"]
            params = StdioServerParameters(
                command=command[0],
                args=command[1:],
                # 显式继承当前环境：子进程可能需要 API key 等环境变量
                env=os.environ.copy(),
                cwd=str(self.cfg.get("cwd", ROOT)),
            )
            # 把子进程 stderr 丢弃，避免它的日志混进终端。
            # 注意必须给真实文件描述符（DEVNULL），StringIO 没有 fileno() 会报错。
            self._transport_cm = stdio_client(params, errlog=subprocess.DEVNULL)
        elif transport == "http":
            self._transport_cm = streamablehttp_client(self.cfg["url"])
        else:
            raise ValueError(f"[{self.name}] 不支持的传输类型：{transport}")

        rw = await self._transport_cm.__aenter__()
        read, write = rw[0], rw[1]
        self._session_cm = ClientSession(read, write)
        self.session = await self._session_cm.__aenter__()
        await self.session.initialize()

        result = await self.session.list_tools()
        tools = list(result.tools)
        self.tool_index = {t.name: t for t in tools}
        return tools

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        assert self.session is not None
        result = await self.session.call_tool(tool_name, arguments or {})
        texts = [c.text for c in result.content if isinstance(c, types.TextContent)]
        text = "\n".join(texts).strip()
        if result.isError:
            return f"[MCP 工具 {self.name}/{tool_name} 执行失败] {text}"
        return text or "(工具执行成功但没有文本返回)"

    async def close(self) -> None:
        # 逆序释放：先关 session，再关传输（断子进程）
        if self._session_cm is not None:
            await self._session_cm.__aexit__(None, None, None)
        if self._transport_cm is not None:
            await self._transport_cm.__aexit__(None, None, None)


class MCPToolHub:
    """连接配置里的全部 MCP Server，把外部工具适配成 Agent 可注入的形态。

    用法：
        hub = MCPToolHub()
        hub.start()
        agent = Agent(
            tools=TOOLS + hub.openai_tools(),
            tool_functions={**TOOL_FUNCTIONS, **hub.callables()},
        )
        ...
        hub.close()
    """

    def __init__(self, config: dict[str, dict[str, Any]] | None = None):
        self._config = config if config is not None else load_config()
        self._conns: dict[str, _ServerConn] = {}
        # 暴露给模型的名字 -> (连接, server 上的原始工具名)
        self._tool_map: dict[str, tuple[_ServerConn, str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    # ---- 后台 event loop ----
    def _run_in_loop(self, coro, timeout: float):
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)

    def start(self) -> "MCPToolHub":
        """启动后台 loop，连接所有 server 并发现工具。任一 server 失败不影响其他。"""
        # Windows 上只有 ProactorEventLoop 支持子进程 stdio，
        # SelectorEventLoop 会在拉起 stdio server 时抛 "fileno" 错误。
        if sys.platform == "win32":
            self._loop = asyncio.ProactorEventLoop()
        else:
            self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, name="mcp-client-loop", daemon=True
        )
        self._thread.start()

        for name, cfg in self._config.items():
            try:
                conn = _ServerConn(name, cfg)
                tools = self._run_in_loop(conn.open(), CONNECT_TIMEOUT)
            except Exception as e:
                print(f"  [MCP] 连接 {name} 失败，已跳过：{e}")
                continue
            self._conns[name] = conn
            for tool in tools:
                exposed = f"mcp_{_sanitize(name)}__{_sanitize(tool.name)}"
                self._tool_map[exposed] = (conn, tool.name)
            print(f"  [MCP] 已连接 {name}，发现 {len(tools)} 个工具：{[t.name for t in tools]}")
        return self

    # ---- 给 Agent 的两样东西 ----
    def openai_tools(self) -> list[dict[str, Any]]:
        """外部工具的 OpenAI function-calling schema，可直接拼进 TOOLS。"""
        result = []
        for exposed, (conn, remote_name) in self._tool_map.items():
            # 找到原始 Tool 对象拿描述/schema
            tool = self._raw_tool(conn, remote_name)
            schema = dict(tool.inputSchema or {})
            schema.setdefault("type", "object")
            result.append({
                "type": "function",
                "function": {
                    "name": exposed,
                    "description": f"[外部工具·MCP服务 {conn.name}] {tool.description or ''}",
                    "parameters": schema,
                },
            })
        return result

    def callables(self) -> dict[str, Any]:
        """名字 -> 同步可调用函数，可直接合并进 TOOL_FUNCTIONS。"""
        return {name: self._make_callable(name) for name in self._tool_map}

    def _raw_tool(self, conn: _ServerConn, remote_name: str):
        # 工具清单只在 open 时取过一次，这里从 session 缓存不划算，
        # 直接让 conn 记住 list_tools 的结果。
        return conn.tool_index[remote_name]

    def _make_callable(self, exposed_name: str):
        conn, remote_name = self._tool_map[exposed_name]

        def call(**kwargs) -> str:
            return self._run_in_loop(conn.call(remote_name, kwargs), CALL_TIMEOUT)

        call.__name__ = exposed_name
        return call

    @property
    def tool_names(self) -> list[str]:
        return list(self._tool_map)

    def close(self) -> None:
        """关闭全部连接，停掉后台 loop。"""
        async def _shutdown():
            for conn in self._conns.values():
                try:
                    await conn.close()
                except Exception:
                    pass

        if self._loop is not None and self._loop.is_running():
            try:
                self._run_in_loop(_shutdown(), CONNECT_TIMEOUT)
            except Exception:
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._conns.clear()
        self._tool_map.clear()


# --------------------------------------------------------------------------
# 进程级单例：CLI / server 共用，外部 MCP 连接整个进程只建一次
# --------------------------------------------------------------------------
_shared_hub: "MCPToolHub | None" = None
_hub_lock = threading.Lock()


def get_shared_hub() -> "MCPToolHub | None":
    """惰性启动并连接配置里的全部 MCP server，进程内只连一次。

    - 环境变量 DISABLE_MCP=1：直接返回 None，退回纯本地工具（断网/排障用）。
    - 所有 server 都连不上（一个外部工具都没拿到）：返回 None，调用方无需特判。
    """
    global _shared_hub
    if os.environ.get("DISABLE_MCP") == "1":
        return None
    with _hub_lock:
        if _shared_hub is None:
            hub = MCPToolHub().start()
            if hub.tool_names:
                _shared_hub = hub
            else:
                # 一个外部工具都没拿到：没必要保留后台线程，关掉
                hub.close()
        return _shared_hub


def close_shared_hub() -> None:
    """进程退出时调用：关闭全部外部 MCP 连接和后台 loop。"""
    global _shared_hub
    with _hub_lock:
        if _shared_hub is not None:
            _shared_hub.close()
            _shared_hub = None


def merged_tools() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """返回 (openai_tools, tool_functions)：本地工具在前、外部 MCP 工具在后。

    无可用外部 server 时原样返回本地那套，CLI / server 行为与接线前一致。
    """
    # 局部导入避免 mcp_client ↔ tools 之间产生不必要的顶层耦合
    from tools import TOOL_FUNCTIONS, TOOLS

    hub = get_shared_hub()
    if hub is None:
        return TOOLS, TOOL_FUNCTIONS
    return (
        TOOLS + hub.openai_tools(),
        {**TOOL_FUNCTIONS, **hub.callables()},
    )


def external_tools_prompt() -> str:
    """生成注入 system prompt 的「外部 MCP 工具清单」文本块。

    光把工具塞进 tools 参数还不够——生产 system prompt 没提过这些能力，
    模型会把陌生工具当摆设。这里把每个外部工具的名字和用途列出来，明确告知
    「你确实能做这些事」。无外部工具时返回空串，对原有提示词零影响。
    """
    hub = get_shared_hub()
    if hub is None:
        return ""
    lines = [
        "",
        "## 你可调用的外部工具（MCP 生态）",
        "除本地工具外，你还接入了以下外部系统的能力。用户相关需求请直接调用对应工具，",
        "不要因为不在本地工具列表里就拒绝或声称做不到：",
    ]
    for schema in hub.openai_tools():
        fn = schema["function"]
        # 去掉适配层加的统一前缀，让描述读起来更自然
        desc = fn["description"].replace("[外部工具·MCP服务 ", "（").replace("]", "）", 1)
        lines.append(f"- {fn['name']}：{desc}")
    return "\n".join(lines)
