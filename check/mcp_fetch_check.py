"""验证「真实第三方 MCP Server」即插即用：官方 mcp-server-fetch。

和自连测试的区别：fetch 是外部社区维护的 server，我们一行适配代码都没改，
只在 mcp_servers.json 里加了配置条目，Agent 就获得了「抓取网页」的新能力。

    uv run python mcp_fetch_check.py
"""
from __future__ import annotations

from agent import Agent
from mcp_client import MCPToolHub

URL = "https://example.com"


def main() -> None:
    hub = MCPToolHub()  # 读 mcp_servers.json：agent-demo + fetch 两个 server
    hub.start()
    try:
        print("\n========== 多 server 聚合后的全部外部工具 ==========")
        for name in hub.tool_names:
            print(" -", name)

        # ---------- 直连 fetch ----------
        print("\n========== 直连第三方 fetch 工具抓取网页 ==========")
        fetch_fn = hub.callables()["mcp_fetch__fetch"]
        result = fetch_fn(url=URL, max_length=600)
        print(result[:600])

        # ---------- Agent 端到端：自主用外部 fetch 回答 ----------
        print("\n========== Agent 自主调用 fetch 回答 ==========")
        agent = Agent(
            system_prompt=(
                "你是助手。需要网页内容时，调用名字以 mcp_fetch__ 开头的外部工具，"
                "拿到内容后用中文简要概括，不要罗列原始 HTML。"
            ),
            tools=hub.openai_tools(),
            tool_functions=hub.callables(),
            enable_claim_check=False,
            persist_messages=False,
        )
        print(agent.chat(f"帮我抓取 {URL} 并用一句话概括它是什么页面。"))
    finally:
        hub.close()
        print("\n[hub 已关闭]")


if __name__ == "__main__":
    main()
