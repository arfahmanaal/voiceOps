import os

import numpy as np

from src.kb import load_sections

EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")


def default_embed():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    prefix = {"passage": "passage: ", "query": "query: "}

    def embed(texts, kind):
        return model.encode(
            [prefix[kind] + text for text in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    return embed


class Retriever:
    def __init__(self, embed=None):
        self.chunks = load_sections()
        self.embed = embed or default_embed()
        passages = [f"{c['doc_title']} - {c['title']}\n{c['text']}" for c in self.chunks]
        self.matrix = self.embed(passages, "passage")

    def search(self, query, k=5):
        scores = self.matrix @ self.embed([query], "query")[0]
        order = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in order]