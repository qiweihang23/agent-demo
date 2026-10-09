"""MCP over HTTP 验证：作为远程客户端连到 server.py 暴露的 streamable HTTP 端点，
走完整 JSON-RPC over HTTP 流程：initialize -> tools/list -> tools/call，并复测会话隔离。

前提：先在另一个终端把服务跑起来
    uv run python server.py
然后本脚本：
    uv run python mcp_http_check.py
"""
from __future__ import annotations

import asyncio
import re

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MCP_URL = "http://127.0.0.1:8000/mcp/mcp"

async def main() -> None:
    # streamable_http_client 返回 (read, write, get_session_id) 三元组。
    # 第三个是回调，能拿到服务端分配的 Mcp-Session-Id；这里不需要用，
    # ClientSession 内部会自动带上该 header 维持会话。
    async with streamable_http_client(MCP_URL) as (read_stream, write_stream, _get_session_id):
        async with ClientSession(read_stream, write_stream) as session:
            # 1) 握手：交换协议版本与能力（HTTP 模式下服务端会分配 Mcp-Session-Id）
            await session.initialize()

            # 2) tools/list：和 stdio 一样，服务端自报家门
            listed = await session.list_tools()
            print("=== 服务端暴露的工具 ===")
            for t in listed.tools:
                print(f"- {t.name}: {t.description.splitlines()[0]}")

            # 3) tools/call：天气（只读，无副作用，适合冒烟）
            print("\n=== 调用 weather(city=杭州) ===")
            r1 = await session.call_tool("weather", {"city": "杭州"})
            print(r1.content[0].text)

            # 4) 会话隔离验证：和 mcp_check.py 同样的用例，但这次是跨进程跨网络
            test_sid = "mcp-http-test"
            print(f"\n=== add_todos → 会话 {test_sid} ===")
            r_add = await session.call_tool(
                "add_todos",
                {"items": ["MCP-HTTP 隔离验证项"], "session_id": test_sid},
            )
            added_text = r_add.content[0].text
            print(added_text)

            print(f"\n=== todos(session={test_sid})：应能看到上一条 ===")
            print((await session.call_tool("todos", {"session_id": test_sid})).content[0].text)

            print("\n=== todos(session=default)：不应出现 MCP-HTTP 隔离验证项 ===")
            print((await session.call_tool("todos", {})).content[0].text)

            # 清场：抠出 #编号 删除，脚本可重复跑
            m = re.search(r"#(\d+)", added_text)
            if m:
                tid = int(m.group(1))
                print(f"\n=== delete_todo #{tid}（清场）===")
                r_del = await session.call_tool(
                    "delete_todo", {"todo_id": tid, "session_id": test_sid}
                )
                print(r_del.content[0].text)

if __name__ == "__main__":
    asyncio.run(main())