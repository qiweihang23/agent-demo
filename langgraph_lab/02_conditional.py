from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    current: int        # 当前步骤编号（默认覆盖语义，正好做"指针"）
    retries: int        # 已重试次数
    verdict: str        # 评估结论：pass / retry

def execute(state: State) -> dict:
    print(f"执行步骤 {state['current']}（第 {state['retries'] + 1} 次尝试）")
    # 模拟：第 1 步第一次必失败、重试才通过；第 2 步直接通过
    if state["current"] == 1 and state["retries"] == 0:
        return {"verdict": "retry"}
    return {"verdict": "pass"}

def evaluate(state: State) -> dict:
    print(f"评估结论 = {state['verdict']}")
    return {}   # 纯"路由节点"：不改状态，只负责决定去哪

# 条件边要用的"路由函数"：读状态，返回下一个节点的名字
def route_after_eval(state: State) -> str:
    # verdict == "retry" 时返回 "execute"（往回跳重做）
    if state["verdict"] == "retry":
        return "reset_retry"
    # 否则分两种：current == 1 时去 "step2"，current == 2 时去 END
    if state["current"] == 1:
        return "step2"
    return END
    # TODO①：verdict == "retry" 时返回 "execute"（往回跳重做）
    #        否则分两种：current == 1 时去 "step2"，current == 2 时去 END
    #   提示：END 这个终点要从 langgraph.graph import，直接 return END
    pass

def goto_step2(state: State) -> dict:
    return {"current": 2, "retries": 0, "verdict": ""}

def reset_retry(state: State) -> dict:
    # 回到 execute 重做前，重试次数 +1
    return {"retries": state["retries"] + 1}

builder = StateGraph(State)

builder.add_node("execute", execute)
builder.add_node("evaluate", evaluate)
builder.add_node("step2", goto_step2)
builder.add_node("reset_retry", reset_retry)

builder.add_edge(START, "execute")
builder.add_edge("execute", "evaluate")
builder.add_conditional_edges("evaluate", route_after_eval)
# TODO②：用 add_conditional_edges 给 evaluate 连条件边
#   语法：builder.add_conditional_edges("evaluate", 路由函数名)
#   （这里路由函数直接 return 节点名/END，可以不传路由表；先体会最简形式）

# step2 之后也要执行→评估，复用同一对节点
builder.add_edge("step2", "execute")
# reset_retry 做完回到 execute
builder.add_edge("reset_retry", "execute")

graph = builder.compile()
result = graph.invoke({"current": 1, "retries": 0, "verdict": ""})
print("最终状态 =", result)