"""MCP 最小闭环验证：作为客户端拉起 mcp_server.py 子进程，
走完整的 JSON-RPC over stdio 流程：initialize -> tools/list -> tools/call。

跑法：
    uv run python mcp_check.py
"""
from __future__ import annotations

import asyncio
import re

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main() -> None:
    # 描述"怎么把子进程拉起来"。Windows 下 uv.exe 在 PATH 里可直接用
    params = StdioServerParameters(
        command="uv",
        args=["run", "python", "mcp_server.py"],
        # 子进程继承本项目 .env（tools.py 里天气工具不需要 key，但保持习惯）
        env=None,
    )

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1) 握手：交换协议版本与能力
            await session.initialize()

            # 2) tools/list：服务端自报家门，看看协议层自动生成了什么 schema
            listed = await session.list_tools()
            print("=== 服务端暴露的工具 ===")
            for t in listed.tools:
                print(f"- {t.name}: {t.description.splitlines()[0]}")
                print(f"  输入 schema: {t.inputSchema}")

            # 3) tools/call：天气（无状态工具）
            print("\n=== 调用 weather(city=杭州) ===")
            r1 = await session.call_tool("weather", {"city": "杭州"})
            print(r1.content[0].text)

            # 4) 会话隔离验证：显式 session_id 必须真正决定数据归属
            test_sid = "mcp-test"
            print(f"\n=== add_todos → 会话 {test_sid} ===")
            r_add = await session.call_tool(
                "add_todos",
                {"items": ["MCP隔离验证项"], "session_id": test_sid},
            )
            added_text = r_add.content[0].text
            print(added_text)

            print(f"\n=== todos(session={test_sid})：应能看到上一条 ===")
            print((await session.call_tool("todos", {"session_id": test_sid})).content[0].text)

            print("\n=== todos(session=default)：不应出现 MCP隔离验证项 ===")
            print((await session.call_tool("todos", {})).content[0].text)

            # 清场：从 add 的返回文本里抠出 #编号 删除，脚本可重复跑
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