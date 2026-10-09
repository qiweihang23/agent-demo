"""手搓的轻量链路追踪：不引入 OpenTelemetry，只做本项目够用的三件事——

1. 一次对话 = 一条 trace，里面每个阶段（整轮 / LLM 调用 / 工具执行）= 一个 span，
   span 有父子关系，各自记录耗时和关键属性；

2. span 结束时打一条 structlog 结构化日志；

3. 最近 N 条 trace 留在内存环形缓冲里，供 server 的 /traces 接口回看。

并发模型说明：

- Trace 对象和父 span id 全部显式当参数传，不使用 contextvars。
  原因：SSE 同步生成器会被 anyio.to_thread 一次次放进「全新复制的 context」里
  推进，contextvar 的 token 在某次 next() 里 set、在另一次 next() 里 reset 会抛
  "Token was created in a different Context"。显式传参对生成器挂起/换线程免疫。

- 工具在线程池里并发执行：Trace._add 自带 Lock，多线程同时开 span 安全，
  父 span id 由调用方显式传入，不需要 bind_ctx 之类的上下文恢复。
"""
from __future__ import annotations

import contextlib
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import structlog

log = structlog.get_logger()

MAX_TRACES = 200          # 内存里最多留多少条，多了丢最老的
PREVIEW_LIMIT = 200       # 工具参数/结果只记前 200 字，避免一条 span 撑爆内存

def preview(value: Any, limit: int = PREVIEW_LIMIT) -> str:
    """把任意值压成短字符串用于展示。"""
    s = value if isinstance(value, str) else repr(value)
    return s if len(s) <= limit else s[:limit] + "…"

@dataclass
class Span:
    span_id: int
    parent_id: int | None
    name: str
    start_wall: float                 # time.time()，给人看的时间戳
    start_perf: float                 # perf_counter，用来算耗时
    attrs: dict[str, Any] = field(default_factory=dict)
    end_perf: float | None = None
    error: str | None = None

    def set_attr(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    @property
    def duration_ms(self) -> float | None:
        if self.end_perf is None:
            return None
        return round((self.end_perf - self.start_perf) * 1000, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "start": time.strftime("%H:%M:%S", time.localtime(self.start_wall)),
            "duration_ms": self.duration_ms,
            "attrs": self.attrs,
            "error": self.error,
        }

class Trace:
    def __init__(self, name: str, attrs: dict[str, Any]):
        self.trace_id = uuid.uuid4().hex[:12]
        self._lock = Lock()
        self._next_id = 1
        self.start_wall = time.time()
        self.end_perf: float | None = None
        self.spans: list[Span] = []
        self.root = self._add(None, name, attrs)

    def _add(self, parent: int | None, name: str, attrs: dict[str, Any]) -> Span:
        with self._lock:
            sp = Span(
                span_id=self._next_id,
                parent_id=parent,
                name=name,
                start_wall=time.time(),
                start_perf=time.perf_counter(),
                attrs=dict(attrs),
            )
            self._next_id += 1
            self.spans.append(sp)
        return sp

    def finish(self) -> None:
        """收尾根 span 并打日志。正常/异常路径都应恰好调一次。"""
        self.root.end_perf = time.perf_counter()
        self.end_perf = self.root.end_perf
        _emit(self, self.root)

    def summary(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "name": self.root.name,
            "start": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.start_wall)
            ),
            "duration_ms": self.root.duration_ms,
            "spans": len(self.spans),
            "error": self.root.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "start": time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(self.start_wall)
            ),
            "root": self.root.to_dict(),
            "spans": [s.to_dict() for s in self.spans],
        }

# 最近的 trace 环形缓冲。服务长期运行也不会无限涨内存。
_recent: deque[Trace] = deque(maxlen=MAX_TRACES)
_recent_lock = Lock()

def recent_traces(limit: int = 20) -> list[dict[str, Any]]:
    with _recent_lock:
        return [t.summary() for t in list(_recent)[-limit:][::-1]]

def get_trace(trace_id: str) -> dict[str, Any] | None:
    with _recent_lock:
        for t in _recent:
            if t.trace_id == trace_id:
                return t.to_dict()
    return None

def _emit(trace: Trace, span: Span) -> None:
    """span 结束时打一条结构化日志。trace 直接当参数传，不读任何隐式上下文。"""
    log.info(
        "span",
        trace_id=trace.trace_id,
        span=span.name,
        ms=span.duration_ms,
        error=span.error,
        **{f"attr_{k}": v for k, v in span.attrs.items()},
    )

@contextlib.contextmanager
def start_trace(name: str, **attrs: Any):
    """开启一条 trace（根 span）。with 块结束自动收尾，异常记到根 span 上。

    纯对象、零 contextvars：可以安全套在会被挂起/换线程推进的生成器外层。
    """
    trace = Trace(name, attrs)
    with _recent_lock:
        _recent.append(trace)
    try:
        yield trace
    except BaseException as e:  # noqa: BLE001 - 追踪层要原样放行所有异常
        trace.root.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        trace.finish()

@contextlib.contextmanager
def trace_span(
    trace: Trace | None,
    name: str,
    parent_id: int | None = None,
    **attrs: Any,
):
    """开一个子 span。trace 为 None 时降级为空操作，业务代码无需感知追踪层。

    父 span 必须显式传入（不再有「当前 span」这种隐式状态），
    因此跨生成器边界和线程池都不会串。
    """
    if trace is None:
        yield _NULL_SPAN
        return
    span = trace._add(parent_id, name, attrs)
    try:
        yield span
    except BaseException as e:  # noqa: BLE001
        span.error = f"{type(e).__name__}: {e}"
        raise
    finally:
        span.end_perf = time.perf_counter()
        _emit(trace, span)

class _NullSpan:
    """没开 trace 时的占位对象，set_attr 静默丢弃。"""

    def set_attr(self, key: str, value: Any) -> None:
        pass

_NULL_SPAN = _NullSpan()
