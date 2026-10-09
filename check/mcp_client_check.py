"""验证 MCP Client 适配层：外部 MCP Server 的工具能被发现、直调，并被 Agent 选用。

阶段 A（直连）：不经过模型，直接调用 hub.callables() 里的同步函数，
               证明「后台 async session → 同步调用」这条桥是通的。
阶段 B（端到端）：把外部工具注入一个全新 Agent，只给它外部工具，
               问天气，看模型是否真的选择并调用了 MCP 工具。

    uv run python mcp_client_check.py
"""
from __future__ import annotations

from agent import Agent
from mcp_client import MCPToolHub


def main() -> None:
    hub = MCPToolHub()  # 默认配置：stdio 自连本项目 mcp_server.py
    hub.start()

    try:
        # ---------- 阶段 A：直连验证 ----------
        print("\n========== 阶段 A：直连外部 MCP 工具 ==========")
        print("发现的外部工具：", hub.tool_names)

        calls = hub.callables()
        weather = calls["mcp_agent-demo__weather"]
        add_todos = calls["mcp_agent-demo__add_todos"]
        list_todos = calls["mcp_agent-demo__todos"]

        print("\n--- 调用 weather(city='杭州') ---")
        print(weather(city="杭州"))

        print("\n--- 调用 add_todos（经 MCP 写入本地库） ---")
        print(add_todos(items=["MCP接入测试：写周报"], session_id="mcp-check"))

        print("\n--- 调用 todos（读回，验证写入确实经过外部 server） ---")
        print(list_todos(session_id="mcp-check"))

        # ---------- 阶段 B：Agent 端到端 ----------
        print("\n========== 阶段 B：Agent 自主选用外部 MCP 工具 ==========")
        agent = Agent(
            system_prompt=(
                "你是一个助手。只能使用提供的工具回答问题；"
                "涉及天气、待办时必须调用名字以 mcp_ 开头的外部工具。"
            ),
            tools=hub.openai_tools(),
            tool_functions=hub.callables(),
            enable_claim_check=False,
            persist_messages=False,
        )
        reply = agent.chat("杭州现在天气怎么样？")
        print("Agent 回复：", reply)

    finally:
        hub.close()
        print("\n[hub 已关闭，子进程已终止]")


if __name__ == "__main__":
    main()
