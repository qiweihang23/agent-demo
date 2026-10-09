"""飞书 MCP 包装层：把 lark-cli 的精选命令变成标准 MCP 工具。

为什么需要这一层：
  Trae 内置的 lark-cli 是「命令行工具」，不是 MCP server——它没有 stdio 上的
  JSON-RPC 接口。我们在外面包一个极薄的 FastMCP server：MCP 工具 → 拼命令行
  参数 → subprocess 调 lark-cli → 把它的 JSON 输出回传给 MCP 客户端。
  这样项目既有的 MCPToolHub 就能像连 fetch server 一样连飞书，client 零改动。

认证怎么透传：
  lark-cli 本身没有独立配置文件，它靠三个环境变量拿凭据：
  LARKSUITE_CLI_USER_ACCESS_TOKEN / LARKSUITE_CLI_BRAND / LARKSUITE_CLI_APP_ID。
  这些变量由宿主（Trae）在授权后注入。本进程从父进程继承环境，subprocess 又
  从本进程继承，所以只要启动我们的 agent 的环境里有这三个变量，整条子进程链
  都有凭据，无需在任何文件里落 token。

    uv run python mcp_lark_server.py   # stdio 模式，由 MCP 客户端拉起
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# 默认用 Trae 插件自带的 lark-cli；也可用环境变量 LARK_CLI_PATH 覆盖
_DEFAULT_CLI = (
    Path.home()
    / ".trae-cn/plugins/trae-remote-official/lark/1.0.5/bin/lark-cli.exe"
)
LARK_CLI = os.environ.get("LARK_CLI_PATH", str(_DEFAULT_CLI))
TIMEOUT = 30

mcp = FastMCP("lark-wrapped")


def _run(args: list[str]) -> str:
    """执行一条 lark-cli 命令，返回它的 JSON 输出。出错也返回字符串，交给模型纠偏。"""
    try:
        proc = subprocess.run(
            [LARK_CLI, *args],
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            encoding="utf-8",
        )
    except FileNotFoundError:
        return f"[配置错误] 找不到 lark-cli：{LARK_CLI}，可用环境变量 LARK_CLI_PATH 指定"
    except subprocess.TimeoutExpired:
        return f"[超时] lark-cli 执行超过 {TIMEOUT}s：{' '.join(args)}"

    out = (proc.stdout or "").strip()
    err = (proc.stderr or "").strip()
    # lark-cli 正常结果在 stdout（JSON）；失败信息有时走 stderr
    return out or err or "(lark-cli 没有返回内容)"


def _compact(raw: str, limit: int = 3000) -> str:
    """lark-cli 输出是 JSON；能解析就提取 data，失败再截断原文，省 token。"""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw[:limit]
    if isinstance(payload, dict) and payload.get("ok") is False:
        return f"[飞书接口报错] {json.dumps(payload.get('error', payload), ensure_ascii=False)}"
    return json.dumps(payload.get("data", payload), ensure_ascii=False, indent=2)[:limit]


@mcp.tool()
def lark_agenda() -> str:
    """查询我今天的飞书日历日程安排。无参数。返回今天的会议/日程列表（可能为空）。"""
    return _compact(_run(["calendar", "+agenda"]))


@mcp.tool()
def lark_my_tasks() -> str:
    """查询分配给我的飞书任务清单。无参数。返回我负责的未完成任务。"""
    return _compact(_run(["task", "+get-my-tasks"]))


@mcp.tool()
def lark_search_chat(query: str) -> str:
    """按关键词搜索我可见的飞书群聊，用来在发消息前查到准确的 chat_id（oc_ 开头）。

    query：群名称关键词，例如「项目周会」。
    发群消息前必须先搜出 chat_id，不要凭空猜。
    """
    return _compact(_run(["im", "+chat-search", "--query", query]))


@mcp.tool()
def lark_send_message(text: str, chat_id: str = "", user_id: str = "") -> str:
    """以你的身份向飞书群或个人发送一条纯文本消息。写操作，会真实发出。

    text：消息正文。
    chat_id：群聊 ID（oc_ 开头），与 user_id 二选一；群 ID 不确定时先调 lark_search_chat。
    user_id：个人 open_id（ou_ 开头），与 chat_id 二选一。
    """
    if not chat_id and not user_id:
        return "必须提供 chat_id 或 user_id 其中一个作为接收方。"
    args = ["im", "+messages-send", "--text", text, "--yes"]
    args += ["--chat-id", chat_id] if chat_id else ["--user-id", user_id]
    return _compact(_run(args))


if __name__ == "__main__":
    mcp.run()
