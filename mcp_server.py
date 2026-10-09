"""MCP Server：把 agent-demo 的能力按 Model Context Protocol 暴露出去。

分层理解这个文件：
- 业务逻辑层：tools.py 里的 get_weather / list_todos（一行没改，直接复用）
- 协议适配层：本文件。@mcp.tool() 把普通 Python 函数翻译成 MCP 工具，
  函数签名 -> 输入 schema，docstring -> 工具描述，返回值 -> TextContent
- 传输层：默认 stdio，但同一份工具也能改走 streamable HTTP。
  · stdio：本地客户端（Trae / Claude Desktop）拉起本子进程，双方通过
    stdin/stdout 收发 JSON-RPC（stderr 仍可打日志，不影响协议）。
  · HTTP：不在这里启动，由 server.py 把 mcp.streamable_http_app() 挂到
    FastAPI 的 /mcp 路径，远程客户端经 http://host/mcp/mcp 调用。
  两种传输共用同一套 @mcp.tool() 工具，区别只在「消息怎么搬」。

stdio 模式由客户端拉起子进程，不是自己 python 跑：
    uv run python mcp_server.py
HTTP 模式不在这里启动，随主服务一起跑：
    uv run python server.py   # MCP 端点在 http://127.0.0.1:8000/mcp/mcp
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

# 别名导入：MCP 包装函数想保留 add_todos 等工具名，直接同名导入会在函数体里
# 调到自己（无限递归），所以底层实现统一加 _ 前缀
from tools import (
    add_todos as _add_todos,
    complete_todo as _complete_todo,
    delete_todo as _delete_todo,
    get_weather,
    list_todos as _list_todos,
)

# server 名字会出现在客户端的工具列表里，相当于你的"产品名"
mcp = FastMCP("agent-demo")


@mcp.tool()
def weather(city: str) -> str:
    """查询指定城市的当前实时天气（气温、天气状况、湿度、风速）。

    当用户问天气、气温、是否下雨、要不要带伞时使用。
    city 传中文城市名，例如：北京、上海、杭州。
    """
    # 协议层不写任何业务逻辑，只是转调 —— 这就是"适配层"该有的薄度
    return get_weather(city)


@mcp.tool()
def todos(session_id: str = "default") -> str:
    """读取某个会话下已保存的全部待办（未完成在前，含编号和创建时间）。

    当用户问「我有哪些待办」「待办做到哪了」「之前记了什么」时使用。
    session_id 标识会话，不同会话的待办完全互相隔离：用户明确提到会话或
    项目名（如「工作会话」「work」）时传对应标识，没有说明就传 "default"。
    """
    # 不调 set_current_session：参数自足，不碰全局变量，并发调用也不会串会话
    return _list_todos(session_id)


@mcp.tool()
def add_todos(items: list[str], session_id: str = "default") -> str:
    """把待办事项批量保存到指定会话，重启后仍在，内容重复会自动跳过。

    仅当用户明确要求「记录/保存/加个待办」时使用；一次调用传全部条目，
    不要一条一调。每条一句话、以动词开头，例如 ["写周报", "买牛奶"]。
    session_id 语义同 todos。
    """
    return _add_todos(items, session_id)


@mcp.tool()
def complete_todo(todo_id: int, session_id: str = "default") -> str:
    """把指定会话里某条待办标记为已完成。todo_id 是列表中 # 后的数字编号。

    编号不确定时先调 todos 查，不要猜。session_id 语义同 todos。
    """
    return _complete_todo(todo_id, session_id)


@mcp.tool()
def delete_todo(todo_id: int, session_id: str = "default") -> str:
    """彻底删除指定会话里某条待办（不可恢复）。todo_id 是列表中 # 后的编号。

    仅在用户明确要求删除时使用。session_id 语义同 todos。
    """
    return _delete_todo(todo_id, session_id)

if __name__ == "__main__":
    # stdio 模式：生命周期完全由父进程（MCP 客户端）管理
    mcp.run()