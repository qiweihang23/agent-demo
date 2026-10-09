"""验证飞书 MCP 包装层：lark-cli 命令经我们自建的 MCP server 被 Agent 调用。

链路：
  Agent ──MCP(stdio)──► mcp_lark_server.py ──subprocess──► lark-cli ──HTTPS──► 飞书开放平台
                         凭据靠 LARKSUITE_CLI_* 环境变量沿子进程链透传，不落盘。

    uv run python mcp_lark_check.py
"""
from __future__ import annotations

from agent import Agent
from mcp_client import MCPToolHub


def main() -> None:
    hub = MCPToolHub()  # 读 mcp_servers.json：agent-demo + fetch + lark
    hub.start()
    try:
        print("\n========== 全部外部工具 ==========")
        for name in hub.tool_names:
            print(" -", name)

        lark_tools = {n: f for n, f in hub.callables().items() if n.startswith("mcp_lark__")}
        assert lark_tools, "没有发现 lark 工具，检查上面的连接日志"

        print("\n========== 直连：今日日程 ==========")
        print(lark_tools["mcp_lark__lark_agenda"]())

        print("\n========== 直连：我的任务 ==========")
        print(lark_tools["mcp_lark__lark_my_tasks"]())

        print("\n========== 直连：搜索群聊（关键词：测试） ==========")
        print(lark_tools["mcp_lark__lark_search_chat"](query="测试"))

        print("\n========== Agent 端到端：自主查日程 ==========")
        agent = Agent(
            system_prompt=(
                "你是飞书助手。查日程/任务/群聊时调用 mcp_lark__ 开头的工具，"
                "拿到结果后用中文简洁汇报；没有数据就如实说没有，不要编造。"
            ),
            tools=hub.openai_tools(),
            tool_functions=hub.callables(),
            enable_claim_check=False,
            persist_messages=False,
        )
        print(agent.chat("我今天飞书日历上有什么安排？"))
    finally:
        hub.close()
        print("\n[hub 已关闭]")


if __name__ == "__main__":
    main()
