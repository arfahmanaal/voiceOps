from src.llm import chat

SYSTEM = (
    "You are ZentroMart's AI support assistant for an Indian online store that sells groceries and electronics. "
    "In your first reply, tell the customer you are an AI assistant. "
    "Answer questions about orders, delivery, returns, refunds, cancellations and warranty. "
    "Reply in the customer's language, in two or three short sentences. "
    "Never promise refunds, compensation or exceptions."
)


def answer(question):
    reply = chat(SYSTEM, [{"role": "user", "content": question}])
    return {"reply": reply, "retrieved": [], "facts": []}