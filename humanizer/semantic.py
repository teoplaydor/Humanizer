"""Optional meaning-preservation checks via a multilingual sentence-embedding model."""
import os

DEFAULT_EMBED = os.environ.get("HUMANIZER_EMBED", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


class Embedder:
    def __init__(self, name: str = DEFAULT_EMBED, device: str | None = None):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self._cache = {}

    def embed(self, texts):
        missing = [t for t in texts if t not in self._cache]
        if missing:
            vecs = self.model.encode(missing, normalize_embeddings=True, convert_to_numpy=True)
            for t, v in zip(missing, vecs):
                self._cache[t] = v
        return [self._cache[t] for t in texts]

    def sim(self, a: str, b: str) -> float:
        va, vb = self.embed([a, b])
        return float((va * vb).sum())

    def sims_to(self, ref: str, cands):
        vr, *vc = self.embed([ref, *cands])
        return [float((vr * v).sum()) for v in vc]

    def text_sim(self, a: str, b: str) -> float:
        """Similarity of two long texts: cosine of mean sentence embeddings."""
        from .textutil import split_sentences

        def mean_vec(t):
            sents = [t[s:e] for s, e in split_sentences(t)] or [t]
            vs = self.embed(sents)
            m = sum(vs) / len(vs)
            return m / ((m * m).sum() ** 0.5 + 1e-9)

        return float((mean_vec(a) * mean_vec(b)).sum())


def load_embedder(name: str | None):
    if name in (None, "", "none"):
        return None
    try:
        return Embedder(name)
    except Exception as e:  # model not downloadable etc.
        print(f"[humanizer] embedder '{name}' unavailable ({type(e).__name__}); meaning checks fall back to LM-only.")
        return None
