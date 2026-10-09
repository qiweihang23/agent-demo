from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    current: int
    retries: int
    verdict: str

def execute(state: State) -> dict:
    print(f"执行步骤 {state['current']}（第 {state['retries'] + 1} 次尝试）")
    if state["current"] == 1 and state["retries"] == 0:
        return {"verdict": "retry"}
    return {"verdict": "pass"}

def evaluate(state: State) -> dict:
    print(f"评估结论 = {state['verdict']}")
    return {}

def route_after_eval(state: State):
    if state["verdict"] == "retry":
        return "reset_retry"
    if state["current"] == 1:
        return "step2"
    return END

def goto_step2(state: State) -> dict:
    return {"current": 2, "retries": 0, "verdict": ""}

def reset_retry(state: State) -> dict:
    return {"retries": state["retries"] + 1}

# 关键：checkpointer 和"在哪个节点前暂停"都变成参数，图本身不变
def build(checkpointer=None, interrupt_before=None):
    b = StateGraph(State)
    b.add_node("execute", execute)
    b.add_node("evaluate", evaluate)
    b.add_node("step2", goto_step2)
    b.add_node("reset_retry", reset_retry)
    b.add_edge(START, "execute")
    b.add_edge("execute", "evaluate")
    b.add_conditional_edges("evaluate", route_after_eval)
    b.add_edge("step2", "execute")
    b.add_edge("reset_retry", "execute")
    return b.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)