from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
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
# 唯一变化①：编译时把 checkpointer 传进去 checkpointer = InMemorySaver()
checkpointer = InMemorySaver()
graph = builder. compile (checkpointer=checkpointer) 
# 唯一变化②：invoke 时带一个 config，用 thread_id 标识"这是哪一条执行线" 
config = RunnableConfig(configurable={"thread_id": "thread-1" })
result = graph.invoke({"current": 1, "retries": 0, "verdict": ""}, config)  
print("最终状态 =", result)
# 1) 读这条 thread 当前的快照 final = graph.get_state(config) print ( "当前 values =" , final.values) print ( "下一步 next =" , final. next ) # 图跑完了这里是空元组 # 2) 拉出这条 thread 的全部历史快照（每个节点跑完都存了一份） 
print ( "\n--- 历史快照（新→旧）---" ) 
for snap in graph.get_state_history(config): 
    print (f"current= {snap.values.get('current')} , " 
           f"verdict= {snap.values.get('verdict')} , " 
           f"next= { list (snap. next )} " )