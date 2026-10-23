from src.llm import client

for model in sorted(client.models.list().data, key=lambda m: m.id):
    print(model.id)