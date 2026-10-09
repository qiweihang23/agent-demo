from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from lab_graph import build

DB = "langgraph_lab/checkpoints.db"
config: RunnableConfig = {"configurable": {"thread_id": "job-1"}}

with SqliteSaver.from_conn_string(DB) as saver:
    graph = build(saver)                       # 注意：不再设 interrupt_before
    snap = graph.get_state(config)
    print("【进程B】读到旧档 → current =",
          snap.values["current"], " next =", list(snap.next))
    result = graph.invoke(None, config)       # 关键：传 None = 不带新输入，从断点继续
    print("【进程B】续跑完成 =", result)