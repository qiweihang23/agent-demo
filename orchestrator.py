"""多 agent 编排：supervisor + 两个专精 worker（B-1 handoff 模式）。

核心思想：supervisor 的「工具」不是业务工具，而是「把任务派给某 worker」。
  - weather_worker：只看 get_weather，专精天气查询
  - todo_worker：只看 add_todos/list_todos/complete_todo/delete_todo，专精待办
  - supervisor：只看 handoff_to_weather / handoff_to_todo，决策派给谁、汇总回复

这一步教：多 agent 边界、handoff、专精化（工具子集）、supervisor 的分发职责。
不教：显式规划（B-2）、自我评估（B-3）、失败回溯（B-4）、长期目标（B-5）。

跑法：
    uv run python orchestrator_check.py
或代码里：
    from orchestrator import build_orchestrator
    sup = build_orchestrator(session_id="orch")
    for ev in sup.chat_stream("查杭州天气..."):
        ...
"""
from __future__ import annotations

import threading
from datetime import datetime

from agent import Agent
from tools import (
    TOOLS,
    get_weather,
    add_todos,
    list_todos,
    complete_todo,
    delete_todo,
    create_goal,
    advance_goal,
    get_goal,
    update_goal_status,
)

# ---------- worker 工具子集 ----------
# 专精化 = 模型只看到自己领域的工具，就不会跨界调用。这是「角色边界」的物理实现。
WEATHER_TOOLS = [t for t in TOOLS if t["function"]["name"] == "get_weather"]
TODO_TOOLS = [
    t for t in TOOLS
    if t["function"]["name"] in {"add_todos", "list_todos", "complete_todo", "delete_todo"}
]

WEATHER_FUNCS = {"get_weather": get_weather}
TODO_FUNCS = {
    "add_todos": add_todos,
    "list_todos": list_todos,
    "complete_todo": complete_todo,
    "delete_todo": delete_todo,
}

# ---------- worker prompts ----------
# 固定 prompt，不重载（worker 不需要 {memories}/{documents} 注入）。
# 明确告诉它「只做自己专精的事，不要跨界、不要派任务」—— 收紧角色边界。
WEATHER_PROMPT = """你是天气查询专员。你只能调用 get_weather 工具查城市天气。
拿到任务后查完天气，把结果原样返回给上游，不要做任何额外的事：
- 不要管待办、记忆、文档
- 不要给上游派任务
- 简洁返回天气事实即可。"""

TODO_PROMPT = """你是待办管理专员。你只能调用 add_todos / list_todos / complete_todo / delete_todo。
拿到任务后按需调工具完成，把结果原样返回给上游：
- 不要查天气、不要管记忆、文档
- 不要给上游派任务
- 简洁返回操作结果即可。"""


# ---------- PlanState（B-2 显式规划）----------
# L3 的核心特征：计划不再是「模型脑子里的临时推理」，
# 而是一个可观察、可追踪、跨步骤持久的数据结构。
class PlanState:
    """显式计划对象。make_plan 后填充，每次 handoff 后更新进度。"""
    def __init__(self) -> None:
        self.steps: list[dict] = []
        self.task_evaluation: str | None = None  # 整体任务评估结果

    def make_plan(self, steps: list[str]) -> str:
        self.steps = [
            {
                "idx": i + 1,
                "description": s,
                "status": "pending",
                "result": None,
                "evaluation": None,     # pass / needs_retry / None
                "retry_count": 0,
                "feedback": None,        # 评估给的修正建议
            }
            for i, s in enumerate(steps)
        ]
        self.task_evaluation = None
        lines = [f"  {s['idx']}. {s['description']} [{s['status']}]" for s in self.steps]
        return f"已制定 {len(self.steps)} 步计划：\n" + "\n".join(lines)

    def mark_done(self, step_idx: int, result: str) -> str:
        if not self.steps:
            return "错误：尚未制定计划，请先调 make_plan。"
        if not (1 <= step_idx <= len(self.steps)):
            return f"错误：step_idx {step_idx} 越界（计划共 {len(self.steps)} 步）。"
        step = self.steps[step_idx - 1]
        step["status"] = "done"
        step["result"] = result[:200]
        return f"步骤 {step_idx} 已完成：{step['description'][:40]}"

    def record_evaluation(
        self, step_idx: int, evaluation: str, feedback: str
    ) -> str:
        """记录某步评估结果。pass=可进下一步，needs_retry=需重做。"""
        if not (1 <= step_idx <= len(self.steps)):
            return f"错误：step_idx {step_idx} 越界。"
        step = self.steps[step_idx - 1]
        step["evaluation"] = evaluation
        step["feedback"] = feedback
        return f"步骤 {step_idx} 评估={evaluation}" + (
            f"，建议：{feedback[:80]}" if feedback else ""
        )

    def increment_retry(self, step_idx: int) -> int:
        """步骤重试计数+1，返回当前重试次数。"""
        step = self.steps[step_idx - 1]
        step["retry_count"] += 1
        step["status"] = "retried"
        return step["retry_count"]

    def mark_failed(self, step_idx: int, reason: str) -> str:
        """超过重试上限，标记步骤失败。"""
        step = self.steps[step_idx - 1]
        step["status"] = "failed"
        step["evaluation"] = "failed"
        step["feedback"] = reason
        return f"步骤 {step_idx} 失败：{reason[:80]}"

    def update_plan(self, action: str, step_idx: int, new_description: str = "") -> str:
        """动态改计划：insert_after / delete / replace。

        执行中发现原计划不合适，可以改 —— 这是「失败回溯」的核心：
        不是原地重做某步，而是调整计划本身。
        """
        if not self.steps:
            return "错误：尚未制定计划。"
        if not (1 <= step_idx <= len(self.steps)):
            return f"错误：step_idx {step_idx} 越界。"
        if action == "delete":
            removed = self.steps.pop(step_idx - 1)
            self._renumber()
            return f"已删除原步骤 {step_idx}：{removed['description'][:40]}"
        if action == "replace":
            if not new_description:
                return "错误：replace 需要填 new_description。"
            step = self.steps[step_idx - 1]
            old = step["description"]
            step["description"] = new_description
            step["status"] = "pending"
            step["evaluation"] = None
            step["retry_count"] = 0
            return f"步骤 {step_idx} 描述已更新：{old[:30]} → {new_description[:30]}"
        if action == "insert_after":
            if not new_description:
                return "错误：insert_after 需要填 new_description。"
            self.steps.insert(step_idx, {
                "idx": step_idx + 1,  # 临时，下面 _renumber 会修正
                "description": new_description,
                "status": "pending",
                "result": None,
                "evaluation": None,
                "retry_count": 0,
                "feedback": None,
            })
            self._renumber()
            return f"已在步骤 {step_idx} 后插入新步骤：{new_description[:40]}"
        return f"错误：未知 action={action}（可用：insert_after/delete/replace）"

    def _renumber(self) -> None:
        """删除/插入后重排序号。"""
        for i, s in enumerate(self.steps):
            s["idx"] = i + 1

    def decide_after_failure(
        self, step_idx: int, decision: str, new_description: str = ""
    ) -> str:
        """某步 failed 后必须调，显式选择如何应对 —— 不能沉默继续。

        decision 取 abort（放弃整个任务）/ skip（跳过该步继续）/ revise（改计划重做）。
        revise 时 new_description 填新步骤描述。
        """
        if decision not in ("abort", "skip", "revise"):
            return "错误：decision 必须是 abort/skip/revise。"
        if decision == "abort":
            self.task_evaluation = "aborted"
            for s in self.steps:
                if s["status"] == "pending":
                    s["status"] = "skipped"
            return "任务已放弃（abort），后续 pending 步骤标记为 skipped。"
        if decision == "skip":
            return f"步骤 {step_idx} 失败已跳过，继续执行后续 pending 步骤。"
        if decision == "revise":
            if not new_description:
                return "错误：revise 需要填 new_description。"
            return self.update_plan("replace", step_idx, new_description)
        return ""

    def all_done(self) -> bool:
        return bool(self.steps) and all(
            s["status"] == "done" and s["evaluation"] == "pass"
            for s in self.steps
        )

    def format(self) -> str:
        if not self.steps:
            return "（尚未制定计划）"
        lines = []
        for s in self.steps:
            line = f"  {s['idx']}. {s['description']} [{s['status']}]"
            if s.get("evaluation"):
                line += f" eval={s['evaluation']}"
            if s.get("retry_count"):
                line += f" retry={s['retry_count']}"
            if s.get("result"):
                line += f" → {s['result'][:50]}..."
            lines.append(line)
        if self.task_evaluation:
            lines.append(f"整体评估：{self.task_evaluation}")
        return "\n".join(lines)


# ---------- handoff 工具 ----------
# supervisor 的「工具」不是业务工具，而是「把任务派给某 worker」。
# handoff_to_weather(task) → weather_worker.chat(task) → 返回 worker 的最终回复字符串。
# 这样 supervisor 的决策权 = 「派给谁」，而不是「直接调啥工具」—— 这就是 handoff 模式。
def _make_handoff_tool(worker: Agent, name: str, description: str):
    """把一个 worker 包成一个可被 supervisor 调用的工具函数。"""
    def _handoff(task: str) -> str:
        # worker.chat 是同步封装，跑完整个事件流只取最终回复。
        # supervisor 视角：这就是个「调一下、拿结果」的普通工具。
        return worker.chat(task)
    _handoff.__name__ = name
    _handoff.__doc__ = description
    return _handoff


def _make_planned_handoff(
    worker: Agent, plan: "PlanState", name: str, description: str
):
    """带进度追踪的 handoff：执行后把结果记到 plan 对应步骤上。

    和 B-1 的 _make_handoff_tool 区别：多一个 step_idx 参数，
    把「这次 handoff 对应计划的第几步」显式化 —— 这就是进度追踪。
    """
    def _handoff(task: str, step_idx: int = 0) -> str:
        result = worker.chat(task)
        if not plan.steps:
            # B-5 遗留修复：单步目标推进不需要计划。活已干完，返回成功结果
            # 而不是报错，避免模型进入“补计划 → 重做一遍”的返工循环
            return result
        mark = plan.mark_done(step_idx, result)
        return f"{result}\n\n[{mark}]"
    _handoff.__name__ = name
    _handoff.__doc__ = description
    return _handoff


def _handoff_tool_schema(name: str, description: str) -> dict:
    """构造 OpenAI function-calling 格式的工具说明书。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "要交给该专员完成的子任务描述，目标要写清楚。",
                    }
                },
                "required": ["task"],
            },
        },
    }


# ---------- supervisor prompt ----------
# supervisor 也不重载 system prompt（fixed），但需要知道当前日期。
SUPERVISOR_PROMPT_TMPL = """你是任务分发主管（supervisor）。今天是 {date}。

你**不能**直接调用 get_weather / add_todos / list_todos 等业务工具。
你只能调用两个 handoff 工具：
- handoff_to_weather(task)：把任务交给天气专员
- handoff_to_todo(task)：把任务交给待办专员

工作方式：
1. 看用户请求，判断要派给谁（可能要派多次，比如先查天气再写待办）
2. 用 handoff 工具把子任务派出去，拿到结果
3. 拿到所有结果后，汇总成给用户的最终回复

注意：
- 派出去的 task 要写清楚目标，让专员能独立完成
- 不要自己猜天气、不要自己改待办 —— 必须通过 handoff
- 多步任务按顺序派，前一步的结果可以喂给下一步的 task 描述"""


# ---------- B-2 planning supervisor prompt ----------
PLANNING_SUPERVISOR_PROMPT_TMPL = """你是带规划能力的任务分发主管（supervisor）。今天是 {date}。

工作流程（必须严格按顺序，不能跳过任何阶段）：

【阶段 1：规划】
先调用 make_plan(steps) 把用户任务拆成子步骤。要求：
- 每步写清楚目标（让专员能独立完成）
- 步骤要可执行、可验证
- 多步任务按依赖顺序排（比如先查天气、再根据结果写待办）

【阶段 2：执行】
按计划顺序逐个调用 handoff 工具，每次传 step_idx 表示在执行第几步：
- handoff_to_weather(task, step_idx)：把某步交给天气专员
- handoff_to_todo(task, step_idx)：把某步交给待办专员
前一步的结果可以在下一步的 task 描述里引用。

【阶段 3：汇总】
所有步骤 done 后，把各步结果汇总成给用户的最终回复。

你**不能**直接调用 get_weather / add_todos 等业务工具。
只能调用 make_plan / handoff_to_weather / handoff_to_todo。

关键区别（vs 不规划的 supervisor）：
- 必须先 make_plan 再 handoff，不能跳过规划
- 每次 handoff 必须带 step_idx，把执行和计划绑起来
- 这样计划是显式的、可追踪的，不是脑子里的临时想法"""


# ---------- 通用装配层（B 阶段收尾重构）----------
# 重构原则：装配样板（worker/handoff/闭包/schema/Agent 构造）只写一遍，
# 各 build_X 函数只保留「选哪个 prompt + 挂哪些工具」的差异部分。

def _today() -> str:
    now = datetime.now()
    return f"{now:%Y-%m-%d} 星期{'一二三四五六日'[now.weekday()]}"


def _tool_schema(
    name: str, description: str, properties: dict, required: list[str]
) -> dict:
    """OpenAI function-calling 工具说明书。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object", "properties": properties, "required": required
            },
        },
    }


def _handoff_schema(name: str, description: str, planned: bool = False) -> dict:
    """handoff 工具说明书；planned=True 时多带 step_idx（B-2+ 进度追踪）。"""
    props = {
        "task": {
            "type": "string",
            "description": "要交给该专员完成的子任务描述，目标要写清楚。",
        }
    }
    required = ["task"]
    if planned:
        props["step_idx"] = {
            "type": "integer", "description": "正在执行计划的第几步（从 1 开始）。"
        }
        required.append("step_idx")
    return _tool_schema(name, description, props, required)


HANDOFF_DESC_W = "把一个需要查天气的子任务交给天气专员，返回天气事实。"
HANDOFF_DESC_T = "把一个待办管理子任务（添加/列出/完成/删除待办）交给待办专员，返回操作结果。"
PLANNED_HANDOFF_DESC_W = "执行某步：把任务交给天气专员。step_idx 标记计划第几步。"
PLANNED_HANDOFF_DESC_T = "执行某步：把任务交给待办专员。step_idx 标记计划第几步。"


def _build_workers(
    model: str | None, session_id: str, user_id: str
) -> tuple[Agent, Agent]:
    """两个专精 worker：固定 prompt、不落库、开 claim 校验。"""
    common = dict(
        model=model, session_id=session_id, user_id=user_id,
        reload_system_prompt=False, enable_claim_check=True, persist_messages=False,
    )
    return (
        Agent(system_prompt=WEATHER_PROMPT, tools=WEATHER_TOOLS,
              tool_functions=WEATHER_FUNCS, **common),
        Agent(system_prompt=TODO_PROMPT, tools=TODO_TOOLS,
              tool_functions=TODO_FUNCS, **common),
    )


def _build_planned_handoffs(
    model: str | None, session_id: str, user_id: str, plan: "PlanState"
) -> tuple:
    """planned 版 handoff 对（B-2+）：执行后自动 mark_done 到计划步骤。"""
    weather_worker, todo_worker = _build_workers(model, session_id, user_id)
    return (
        _make_planned_handoff(
            weather_worker, plan, "handoff_to_weather", PLANNED_HANDOFF_DESC_W
        ),
        _make_planned_handoff(
            todo_worker, plan, "handoff_to_todo", PLANNED_HANDOFF_DESC_T
        ),
    )


def _build_supervisor(
    model: str | None, session_id: str, user_id: str,
    prompt: str, tools: list, funcs: dict,
) -> Agent:
    """supervisor：主会话持久化、固定 prompt、关 claim 校验（handoff 会误伤）。"""
    return Agent(
        model=model, system_prompt=prompt, session_id=session_id,
        user_id=user_id, tools=tools, tool_functions=funcs,
        reload_system_prompt=False, enable_claim_check=False, persist_messages=True,
    )


MAX_STEP_RETRY = 2  # 每步最多重试 2 次，防死循环


def _make_plan_toolkit(plan: "PlanState") -> dict:
    """计划五件套工具（B-3+），闭包共享同一个 plan 对象。"""
    def make_plan(steps: list[str]) -> str:
        return plan.make_plan(steps)
    make_plan.__doc__ = "制定任务执行计划。必须先调此工具。"

    def evaluate_step(step_idx: int, evaluation: str, feedback: str = "") -> str:
        if evaluation not in ("pass", "needs_retry"):
            return "错误：evaluation 必须是 'pass' 或 'needs_retry'。"
        if evaluation == "needs_retry":
            retries = plan.increment_retry(step_idx)
            if retries > MAX_STEP_RETRY:
                return plan.mark_failed(
                    step_idx, f"重试 {MAX_STEP_RETRY} 次仍未通过：{feedback}"
                )
        return plan.record_evaluation(step_idx, evaluation, feedback)
    evaluate_step.__doc__ = (
        "评估某步结果。evaluation 取 'pass'/'needs_retry'，后者填 feedback。"
        "超重试上限自动标 failed。"
    )

    def evaluate_task(evaluation: str, feedback: str = "") -> str:
        plan.task_evaluation = evaluation
        return f"整体评估={evaluation}" + (
            f"，建议：{feedback[:80]}" if feedback else ""
        )
    evaluate_task.__doc__ = "全部步骤处理完后调用，评估整体任务达成度。"

    def update_plan(action: str, step_idx: int, new_description: str = "") -> str:
        return plan.update_plan(action, step_idx, new_description)
    update_plan.__doc__ = (
        "动态改计划。action='insert_after'/'delete'/'replace'。执行中发现计划不合适时调。"
    )

    def decide_after_failure(step_idx: int, decision: str, new_description: str = "") -> str:
        return plan.decide_after_failure(step_idx, decision, new_description)
    decide_after_failure.__doc__ = "某步 failed 后必须调。decision='abort'/'skip'/'revise'。"

    return {
        "make_plan": make_plan, "evaluate_step": evaluate_step,
        "evaluate_task": evaluate_task, "update_plan": update_plan,
        "decide_after_failure": decide_after_failure,
    }


def _plan_tool_schemas() -> list[dict]:
    """计划五件套 + planned handoff 的 schema（B-2~B-5 共用一份）。"""
    return [
        _tool_schema(
            "make_plan", "制定任务执行计划。必须先调此工具。",
            {"steps": {"type": "array", "items": {"type": "string"},
                       "description": "子任务描述列表，按执行顺序排。"}},
            ["steps"],
        ),
        _handoff_schema("handoff_to_weather", PLANNED_HANDOFF_DESC_W, planned=True),
        _handoff_schema("handoff_to_todo", PLANNED_HANDOFF_DESC_T, planned=True),
        _tool_schema(
            "evaluate_step",
            "评估某步结果。evaluation 取 'pass'/'needs_retry'，后者填 feedback。超重试上限自动标 failed。",
            {"step_idx": {"type": "integer", "description": "评估第几步。"},
             "evaluation": {"type": "string", "enum": ["pass", "needs_retry"]},
             "feedback": {"type": "string", "description": "修正建议。"}},
            ["step_idx", "evaluation"],
        ),
        _tool_schema(
            "evaluate_task", "全部步骤处理完后调用，评估整体任务达成度。",
            {"evaluation": {"type": "string", "enum": ["pass", "needs_retry"]},
             "feedback": {"type": "string"}},
            ["evaluation"],
        ),
        _tool_schema(
            "update_plan", "动态改计划。action='insert_after'/'delete'/'replace'。",
            {"action": {"type": "string", "enum": ["insert_after", "delete", "replace"]},
             "step_idx": {"type": "integer"}, "new_description": {"type": "string"}},
            ["action", "step_idx"],
        ),
        _tool_schema(
            "decide_after_failure",
            "某步 failed 后必须调。decision='abort'/'skip'/'revise'。",
            {"step_idx": {"type": "integer"},
             "decision": {"type": "string", "enum": ["abort", "skip", "revise"]},
             "new_description": {"type": "string"}},
            ["step_idx", "decision"],
        ),
    ]


def _goal_tool_schemas() -> list[dict]:
    """B-5 goal 四件套 schema。"""
    return [
        _tool_schema(
            "create_goal", "创建长期目标。跨多次对话推进的成果。",
            {"description": {"type": "string"},
             "success_criteria": {"type": "string", "description": "可验证的达成标准。"}},
            ["description", "success_criteria"],
        ),
        _tool_schema(
            "advance_goal",
            "推进目标。next_step 必须是尚未完成的下一个动作（刚做完的不要写）。状态自动 pending→in_progress→on_track。",
            {"goal_id": {"type": "integer"}, "action": {"type": "string"},
             "result": {"type": "string"}, "next_step": {"type": "string"}},
            ["goal_id", "action", "result", "next_step"],
        ),
        _tool_schema(
            "get_goal", "查长期目标。不传 goal_id 返回所有进行中目标。",
            {"goal_id": {"type": "integer"}}, [],
        ),
        _tool_schema(
            "update_goal_status",
            "状态流转 done/blocked/abandoned。pending/in_progress/on_track 自动管理。",
            {"goal_id": {"type": "integer"},
             "status": {"type": "string", "enum": ["done", "blocked", "abandoned"]},
             "reason": {"type": "string"}},
            ["goal_id", "status"],
        ),
    ]


GOAL_FUNCS = {
    "create_goal": create_goal, "advance_goal": advance_goal,
    "get_goal": get_goal, "update_goal_status": update_goal_status,
}


def build_orchestrator(
    model: str | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> Agent:
    """构建 B-1 supervisor：内部装配两个 worker，handoff 逻辑对调用方透明。

    worker 用主 session_id（业务工具作用到真实数据）但不落库，
    工具结果回灌主会话，对话本身不污染主历史。
    """
    supervisor_prompt = SUPERVISOR_PROMPT_TMPL.replace("{date}", _today())
    weather_worker, todo_worker = _build_workers(model, session_id, user_id)

    weather_handoff = _make_handoff_tool(weather_worker, "handoff_to_weather", HANDOFF_DESC_W)
    todo_handoff = _make_handoff_tool(todo_worker, "handoff_to_todo", HANDOFF_DESC_T)
    return _build_supervisor(
        model, session_id, user_id, supervisor_prompt,
        [
            _handoff_schema("handoff_to_weather", HANDOFF_DESC_W),
            _handoff_schema("handoff_to_todo", HANDOFF_DESC_T),
        ],
        {"handoff_to_weather": weather_handoff, "handoff_to_todo": todo_handoff},
    )


# ---------- B-2：带规划能力的 orchestrator ----------
def build_planning_orchestrator(
    model: str | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> tuple[Agent, PlanState]:
    """构建带规划能力的 supervisor，返回 (supervisor, plan_state)。

    plan_state 是外部可读的：跑完 chat_stream 后调 plan.format() 看进度。
    这就是「计划可观察」—— L3 的核心特征，L2 做不到。
    """
    plan = PlanState()
    supervisor_prompt = PLANNING_SUPERVISOR_PROMPT_TMPL.replace("{date}", _today())
    weather_worker, todo_worker = _build_workers(model, session_id, user_id)
    weather_handoff = _make_planned_handoff(
        weather_worker, plan, "handoff_to_weather",
        "把某步交给天气专员，返回天气事实。step_idx 标记计划第几步。",
    )
    todo_handoff = _make_planned_handoff(
        todo_worker, plan, "handoff_to_todo",
        "把某步交给待办专员，返回操作结果。step_idx 标记计划第几步。",
    )
    toolkit = _make_plan_toolkit(plan)

    sup_tools = [
        _tool_schema(
            "make_plan",
            "制定任务执行计划。steps 是子任务描述列表，按执行顺序排。必须先调此工具再调 handoff。",
            {"steps": {"type": "array", "items": {"type": "string"},
                       "description": "子任务描述列表，按执行顺序排。"}},
            ["steps"],
        ),
        _handoff_schema("handoff_to_weather", weather_handoff.__doc__, planned=True),
        _handoff_schema("handoff_to_todo", todo_handoff.__doc__, planned=True),
    ]
    sup_funcs = {
        "make_plan": toolkit["make_plan"],
        "handoff_to_weather": weather_handoff,
        "handoff_to_todo": todo_handoff,
    }
    supervisor = _build_supervisor(model, session_id, user_id, supervisor_prompt, sup_tools, sup_funcs)
    return supervisor, plan


# ---------- B-5 goal-oriented supervisor prompt ----------
GOAL_SUPERVISOR_PROMPT_TMPL = """你是带长期目标记忆的任务主管。今天是 {date}。

你多了「目标」这层抽象，比 todos 高一级：
- todos = 单步动作（今天读 50 页）
- goals = 跨多次对话推进的成果（这周读完《活着》）

当前用户进行中的长期目标：
{active_goals}
重要：
你只有两个 worker，能力边界如下：
- weather_worker：只能查城市天气
- todo_worker：只能增删改查待办
制定 plan 时，每一步必须能被这两个 worker 之一执行。
如果任务需要其他能力（如查书籍信息、上网搜索），不要制定那一步
——改用常识推断或直接告诉用户"我无法查询该信息"，不要硬派给 worker。
可用工具：
- make_plan / handoff_to_weather / handoff_to_todo / evaluate_step / evaluate_task / update_plan / decide_after_failure（同 B-4）
- create_goal(description, success_criteria)：创建长期目标
- advance_goal(goal_id, action, result, next_step)：推进目标。next_step 必须是**尚未完成**的下一个动作，刚做完的事不要写进去，否则下轮自驱会白跑一趟
- get_goal(goal_id=None)：查目标当前状态
- update_goal_status(goal_id, status, reason)：状态流转 done/blocked/abandoned

工作方式（按任务类型二选一，不要混用）：

【A. 目标推进 —— 多数消息属于这类】
next_step 通常是“加一条待办”“记录一次进度”这类单步动作，
一个 worker 一步就能完成：直接 handoff 执行 → 调 advance_goal
记录 action/result 并写新 next_step。单步动作不需要 make_plan，
完成后也不要补计划。只有真正的多步任务（≥3 步）才走
make_plan → handoff(step_idx) 流程。

【B. 临时任务（与目标无关，如查天气）】
一步能完成就直接 handoff，多步才 make_plan。

其他规则：
- 达成 success_criteria 才调 update_goal_status(done)
- 回复简洁：做不到的事一句话说明，不要长篇解释能力边界
- 同一动作成功过就不要重复执行

这一步的核心认知：
- 用户不用每次重新解释背景 —— 你自己知道要推进什么
- next_step 是给“未来的自己”的提示，必须写清楚
- goal 的状态不是“做完就 done”，而是要对照 success_criteria 评估"""


def build_goal_oriented_orchestrator(
    model: str | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> tuple[Agent, PlanState]:
    """B-5 goal-oriented orchestrator。

    和 B-4 区别：
    - 调用方启动时会调 get_goal 拿当前用户进行中目标，注入 prompt
    - supervisor 有 create/advance/get/update_goal_status 四个新工具
    - 跨对话能接续推进同一个 goal（数据在 goals 表里）
    """
    from tools import get_goal, set_current_user

    # 注入进行中目标 —— supervisor 启动时就知道用户有哪些 goal 在跑
    set_current_user(user_id)
    active_goals = get_goal()  # 返回进行中目标的格式化字符串

    plan = PlanState()
    supervisor_prompt = GOAL_SUPERVISOR_PROMPT_TMPL.replace(
        "{date}", _today()
    ).replace("{active_goals}", active_goals)

    weather_worker, todo_worker = _build_workers(model, session_id, user_id)
    weather_handoff = _make_planned_handoff(
        weather_worker, plan, "handoff_to_weather", PLANNED_HANDOFF_DESC_W,
    )
    todo_handoff = _make_planned_handoff(
        todo_worker, plan, "handoff_to_todo", PLANNED_HANDOFF_DESC_T,
    )
    toolkit = _make_plan_toolkit(plan)

    sup_tools = _plan_tool_schemas() + _goal_tool_schemas()
    sup_funcs = {
        **toolkit,
        "handoff_to_weather": weather_handoff,
        "handoff_to_todo": todo_handoff,
        **GOAL_FUNCS,
    }
    supervisor = _build_supervisor(model, session_id, user_id, supervisor_prompt, sup_tools, sup_funcs)
    return supervisor, plan


# ---------- B-4 resilient supervisor prompt ----------
RESILIENT_SUPERVISOR_PROMPT_TMPL = """你是带规划 + 自我评估 + 失败回溯能力的任务分发主管。今天是 {date}。

可用工具：
- make_plan(steps)：制定计划
- handoff_to_weather / handoff_to_todo(task, step_idx)：执行某步
- evaluate_step(step_idx, evaluation, feedback)：评估某步
- evaluate_task(evaluation, feedback)：整体评估
- update_plan(action, step_idx, new_description)：动态改计划
  · action='insert_after'：在某步后插入新步
  · action='delete'：删除某步
  · action='replace'：替换某步描述（状态重置）
- decide_after_failure(step_idx, decision, new_description)：某步 failed 后必须调
  · decision='abort'：放弃整个任务
  · decision='skip'：跳过该步继续
  · decision='revise'：改计划重做（需填 new_description）

关键规则：
1. 某步 evaluate_step 返回 needs_retry 超重试上限 → 该步会被自动标记 failed
2. **某步 failed 后，必须调 decide_after_failure，不能沉默继续下一步**
   - 这是核心：失败要显式决策，不能糊弄过去
3. 执行中发现原计划描述与实际不符，可以主动调 update_plan 修正
   - 例：查天气发现不下雨，原计划下一步是“加'带伞'待办” → 可调 update_plan('replace', 2, "加'不用带伞'待办")
4. 改完计划后，从第一个 pending 步骤继续执行
5. 全部处理完后仍要调 evaluate_task 做整体评估

你**不能**直接调业务工具。

这一步的核心认知：
- 计划不是冻结的，是活的 —— 执行一步拿到新信息后，要重新评估后续计划是否还合适
- 失败不是“原地重做”，而是“回到计划层修正”
- 失败后不能沉默，必须显式选择如何应对"""


# ---------- B-4 resilient orchestrator ----------
def build_resilient_orchestrator(
    model: str | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> tuple[Agent, PlanState]:
    """B-4 失败回溯 orchestrator。

    和 B-3 区别：
    - 加 update_plan 工具：执行中可动态改计划
    - 加 decide_after_failure 工具：failed 后必须显式决策
    - plan.format() 能看到步骤被 insert/delete/replace 的痕迹
    """
    plan = PlanState()
    supervisor_prompt = RESILIENT_SUPERVISOR_PROMPT_TMPL.replace("{date}", _today())
    weather_worker, todo_worker = _build_workers(model, session_id, user_id)
    weather_handoff = _make_planned_handoff(
        weather_worker, plan, "handoff_to_weather", PLANNED_HANDOFF_DESC_W,
    )
    todo_handoff = _make_planned_handoff(
        todo_worker, plan, "handoff_to_todo", PLANNED_HANDOFF_DESC_T,
    )
    toolkit = _make_plan_toolkit(plan)

    sup_tools = _plan_tool_schemas()
    sup_funcs = {
        **toolkit,
        "handoff_to_weather": weather_handoff,
        "handoff_to_todo": todo_handoff,
    }
    supervisor = _build_supervisor(model, session_id, user_id, supervisor_prompt, sup_tools, sup_funcs)
    return supervisor, plan


# ---------- B-3 self-evaluating supervisor prompt ----------
SELF_EVAL_SUPERVISOR_PROMPT_TMPL = """你是带规划 + 自我评估能力的任务分发主管（supervisor）。今天是 {date}。

工作流程（必须严格按顺序）：

【阶段 1：规划】
调 make_plan(steps) 制定计划。每步要可执行、可验证。

【阶段 2：执行 + 步骤评估】
对每个步骤：
  1) 调 handoff_to_weather / handoff_to_todo(task, step_idx) 执行该步
  2) **立刻**调 evaluate_step(step_idx, evaluation, feedback) 评估结果
     - evaluation 取 'pass' 或 'needs_retry'
     - feedback 填修正建议（needs_retry 时必填，pass 可留空）
  3) 若 needs_retry：调 handoff(task_修正后, step_idx) 重做该步，再调 evaluate_step 重评
     - 最多重试 2 次，仍不过则该步标记 failed，继续下一步
  4) 若 pass：进入下一步

【阶段 3：任务整体评估】
所有步骤处理完后，调 evaluate_task(evaluation, feedback)：
  - evaluation 取 'pass' 或 'needs_retry'
  - 若 needs_retry：哪步不行就重做哪步（同样有重试上限）
  - 若 pass：汇总最终回复

你**不能**直接调业务工具。可用工具：make_plan / handoff_to_weather / handoff_to_todo / evaluate_step / evaluate_task。

关键认知：
- 评估和执行是两个独立动作，不能合并
- 每步必须调 evaluate_step，不能跳过
- 评估是你的检查员帽子，执行是你的执行者帽子，必须分离思考"""


# ---------- B-3 self-evaluating orchestrator ----------


def build_self_evaluating_orchestrator(
    model: str | None = None,
    session_id: str = "default",
    user_id: str = "default",
) -> tuple[Agent, PlanState]:
    """B-3 自我评估 orchestrator。

    和 B-2 区别：
    - supervisor 每次 handoff 后必须调 evaluate_step 评估
    - needs_retry 会重做（有上限），这是 self-critique 的核心
    - 全部 done 后调 evaluate_task 整体评估
    - plan.format() 能看到每步 evaluation/retry_count，L2 做不到
    """
    plan = PlanState()
    supervisor_prompt = SELF_EVAL_SUPERVISOR_PROMPT_TMPL.replace("{date}", _today())
    weather_worker, todo_worker = _build_workers(model, session_id, user_id)
    weather_handoff = _make_planned_handoff(
        weather_worker, plan, "handoff_to_weather", PLANNED_HANDOFF_DESC_W,
    )
    todo_handoff = _make_planned_handoff(
        todo_worker, plan, "handoff_to_todo", PLANNED_HANDOFF_DESC_T,
    )
    toolkit = _make_plan_toolkit(plan)

    # B-3 只用计划三件套（make_plan/evaluate_step/evaluate_task），不含 update_plan/decide_after_failure
    sup_tools = _plan_tool_schemas()[:5]
    sup_funcs = {
        "make_plan": toolkit["make_plan"],
        "handoff_to_weather": weather_handoff,
        "handoff_to_todo": todo_handoff,
        "evaluate_step": toolkit["evaluate_step"],
        "evaluate_task": toolkit["evaluate_task"],
    }
    supervisor = _build_supervisor(model, session_id, user_id, supervisor_prompt, sup_tools, sup_funcs)
    return supervisor, plan


# ---------- B-6 自驱循环（self-driven loop）----------
# B-1~B-5 全是「反应式」：用户发消息 → agent 干活。
# B-6 补上「主动式」：没人发消息，agent 自己醒来干活。
# 关键认知：「用户消息」这个槽位不一定要人来填 —— 目标的 next_step 也可以填。
# 这就是 L4 的核心：agent 和自己的记忆对话，循环推进长期目标。

PROACTIVE_TRIGGER_TMPL = """【系统自驱触发】本轮没有用户输入，是定时调度器唤醒你自主工作。今天是 {date}。

当前用户进行中的目标及各自的 next_step：
{goal_lines}

自驱规则：
1. 逐个评估每个目标：next_step 能否产生**实质推进**（添加新待办、查天气、记录新进度）？
   - 能做 → 直接 handoff 执行（单步动作不需要 make_plan），完成后调 advance_goal
     记录并写下新的 next_step；只有 ≥3 步的多步任务才 make_plan
   - 只是"检查某待办完成没有""等用户反馈""等时间到来" → 本轮跳过该目标：
     不调任何工具、不写 advance_goal。检查不算推进——状态无变化就留日志，
     每轮自驱都会刷一条垃圾日志，把进度日志变成噪音
2. 绝不伪造用户回复：需要用户反馈/确认才能继续的目标一律跳过
3. 若某目标实际已达成 success_criteria → 调 update_goal_status(goal_id, "done")
4. 所有目标都跳过时，直接回复"本轮自驱无可推进项，等待用户"，不调任何工具
5. 本轮最多推进 2 个目标，防止无限自我对话
6. 日期只以上方"今天是 {date}"为准，禁止自行推断时间流逝"""


def run_proactive_round(
    session_id: str,
    user_id: str,
    model: str | None = None,
    verbose: bool = False,
) -> str | None:
    """自驱一轮：无用户输入，由调度器触发 supervisor 按 next_step 推进目标。

    流程：读 goals 表拿 next_step → 拼触发消息 → 走完整 chat_stream。
    返回最终回复；没有进行中目标时返回 None（不浪费 LLM 调用）。
    """
    from tools import set_current_user, get_goal_next_steps

    set_current_user(user_id)
    goals = get_goal_next_steps()
    if not goals:
        return None

    lines = []
    for g in goals:
        lines.append(
            f"- 目标 #{g['goal_id']} [{g['status']}] {g['description']}\n"
            f"  达成标准：{g['success_criteria']}\n"
            f"  next_step：{g['next_step']}"
        )
    # 日期锚点：防模型从 next_step 里的未来日期自行脑补"时间已流逝"
    trigger = (
        PROACTIVE_TRIGGER_TMPL
        .replace("{date}", _today())
        .replace("{goal_lines}", "\n".join(lines))
    )

    # 每轮重建 supervisor：B-5 的 active_goals 注入 prompt 是构建时快照，
    # 重建才能拿到最新目标状态。会话历史在库里，重建不丢上下文。
    sup, _ = build_goal_oriented_orchestrator(
        session_id=session_id, user_id=user_id, model=model
    )

    reply = ""
    for ev in sup.chat_stream(trigger):
        if verbose:
            if ev["type"] == "text":
                print(ev["content"], end="", flush=True)
            elif ev["type"] == "tool_result":
                print(f"\n  [{ev['name']}] {ev['result'][:100]}")
            elif ev["type"] == "final":
                reply = ev["reply"]
                print(f"\n  [trace] {ev['trace_id']}")
        elif ev["type"] == "final":
            reply = ev["reply"]
    return reply


# ---------- 自驱调度器：后台线程定时触发 ----------
# 生产里可换 APScheduler / celery beat；这里用最朴素的 daemon 线程。
# 重点看解耦：调度器只管「喊人」，干活的是 supervisor —— 触发源不同，执行路径不变。
_SELF_DRIVE_TARGETS: list[dict] = []
_self_drive_stop = threading.Event()


def register_self_drive(
    session_id: str, user_id: str, model: str | None = None
) -> None:
    """把一个 (session, user) 注册为自驱对象：调度器每个周期为它跑一轮。"""
    _SELF_DRIVE_TARGETS.append(
        {"session_id": session_id, "user_id": user_id, "model": model}
    )


def start_self_drive(interval_sec: int = 300) -> threading.Thread:
    """启动自驱调度器（daemon 线程）：每 interval_sec 对所有注册目标各跑一轮。"""
    def _loop() -> None:
        # wait(interval) 比 time.sleep 好：stop 置位后立刻返回，线程马上退出
        while not _self_drive_stop.wait(interval_sec):
            for t in list(_SELF_DRIVE_TARGETS):
                try:
                    run_proactive_round(
                        t["session_id"], t["user_id"], t.get("model")
                    )
                except Exception as e:  # 调度器不能被单轮异常打死
                    print(f"[self-drive] {t['user_id']} 本轮异常：{e}")

    thread = threading.Thread(target=_loop, daemon=True, name="self-drive")
    thread.start()
    return thread


def stop_self_drive() -> None:
    """停调度器：置位事件，wait() 立刻返回，线程退出。"""
    _self_drive_stop.set()