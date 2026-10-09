from typing import Annotated
from operator import add
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

# ① 定义 State：图里流动的"数据表"
class State(TypedDict):
    messages: Annotated[list[str], add]   # ② add = 节点返回的列表会被"追加"，而不是覆盖

# ③ 节点：普通函数，读 state，return 一个"更新补丁"
def node_a(state: State) -> dict:
    print("node_a 看到当前 messages =", state["messages"])
    return {"messages": ["你好"]}

def node_b(state: State) -> dict:
    print("node_b 看到当前 messages =", state["messages"])
    return {"messages": ["世界"]}

# ④ 搭图：注册节点 → 连边 → 编译
builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_edge(START, "a")   # 入口 → a
builder.add_edge("a", "b")     # a → b
builder.add_edge("b", END)     # b → 结束

graph = builder.compile()

# ⑤ 运行，传入初始状态
result = graph.invoke({"messages": []})
print("最终状态 =", result)