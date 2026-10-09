from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from lab_graph import build

DB = "langgraph_lab/checkpoints.db"
config: RunnableConfig = {"configurable": {"thread_id": "job-1"}}

with SqliteSaver.from_conn_string(DB) as saver:   # with 退出会自动建表并关闭连接
    graph = build(saver, interrupt_before=["step2"])
    graph.invoke({"current": 1, "retries": 0, "verdict": ""}, config)
    snap = graph.get_state(config)
    print("【进程A】即将退出 → current =",
          snap.values["current"], " next =", list(snap.next))