"""Model parameters, messages, chat prompt templates and few-shot prompts."""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

from triage.data import CRASH_TICKET
from triage.model import get_model

# 1. Parameters are set when the model is created.
model = get_model(temperature=0)
short_model = get_model(temperature=0, max_tokens=5)

# 2. A chat model takes a list of messages and returns an AIMessage.
messages = [
    SystemMessage("You are a friendly support agent. Write a short reply to the customer."),
    HumanMessage(CRASH_TICKET),
]
print("Full answer:     ", model.invoke(messages).content)
print("max_tokens=5:    ", short_model.invoke(messages).content)

# 3. A template fills in variables and produces those same messages.
template = ChatPromptTemplate.from_messages(
    [
        ("system", "You triage tickets. Classify the ticket into one of: {categories}."),
        ("human", "{ticket}"),
    ]
)
prompt_value = template.invoke({"categories": "billing, technical, other", "ticket": CRASH_TICKET})
print("\nFormatted template:")
for message in prompt_value.to_messages():
    print(f"  {message.type}: {message.content}")

# 4. Few-shot prompting shows the model examples of the answer you want.
example_prompt = ChatPromptTemplate.from_messages([("human", "{ticket}"), ("ai", "{category}")])
few_shot = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=[
        {"ticket": "Where is my invoice?", "category": "billing"},
        {"ticket": "The page shows error 500.", "category": "technical"},
    ],
)
few_shot_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You triage tickets. Classify the ticket into one category."),
        few_shot,
        ("human", "{ticket}"),
    ]
)
print("\nFew-shot prompt:")
for message in few_shot_template.invoke({"ticket": CRASH_TICKET}).to_messages():
    print(f"  {message.type}: {message.content}")
print("\nFew-shot answer:", model.invoke(few_shot_template.invoke({"ticket": CRASH_TICKET})).content)
