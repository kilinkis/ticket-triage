"""Draw a chain's structure. ASCII output requires grandalf."""

from langchain_core.runnables import RunnablePassthrough

from triage.chains import build_analysis_chain, build_reply_chain
from triage.model import get_model

model = get_model()
pipeline = build_analysis_chain(model) | RunnablePassthrough.assign(reply=build_reply_chain(model))

graph = pipeline.get_graph()
graph.print_ascii()

print("\nMermaid version, which renders on GitHub:\n")
print(graph.draw_mermaid())
