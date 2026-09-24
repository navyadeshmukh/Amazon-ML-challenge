"""Optional dense embeddings (pretrained multilingual sentence encoder, used AS-IS - no fine-tuning).

Each unique text is embedded once (not once per pair), so cost = #unique records, not #pairs.
Licences (verify on the model card before submitting):
  e5small / e5base -> intfloat/multilingual-e5-*      (MIT)
  bgem3            -> BAAI/bge-m3                     (MIT)
  minilm           -> sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0)
'hash' is an offline char-n-gram stand-in used only to test the plumbing without downloads.
"""
import numpy as np
import pandas as pd

MODELS = {
    "e5small": ("intfloat/multilingual-e5-small", "query: "),
    "e5base": ("intfloat/multilingual-e5-base", "query: "),
    "bgem3": ("BAAI/bge-m3", ""),
    "minilm": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", ""),
}


class Embedder:
    def __init__(self, fn, prefix=""):
        self.fn, self.prefix = fn, prefix

    def encode(self, texts):
        out = self.fn([self.prefix + t for t in texts])
        out = np.asarray(out, dtype=np.float32)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


def load_embedder(name, device=None, batch_size=128):
    if name in (None, "", "none"):
        return None
    if name == "hash":
        from sklearn.feature_extraction.text import HashingVectorizer
        hv = HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4), n_features=512,
                               alternate_sign=False, norm="l2")
        return Embedder(lambda t: hv.transform(t).toarray())
    model_id, prefix = MODELS[name]
    from sentence_transformers import SentenceTransformer
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = SentenceTransformer(model_id, device=device)
    m.max_seq_length = 64
    print(f"      embedder {model_id} on {device}")
    return Embedder(lambda t: m.encode(t, batch_size=batch_size, convert_to_numpy=True,
                                       normalize_embeddings=True, show_progress_bar=False), prefix)


def embed_views(emb: Embedder, df: pd.DataFrame):
    """Return {'name','addr','full'} -> float32 (n, d) arrays."""
    nm = df["business_name"].fillna("").astype(str)
    ad = df["business_address"].fillna("").astype(str)
    texts = {"name": nm, "addr": ad, "full": nm + " | " + ad}
    out = {}
    for v, s in texts.items():
        codes, uniq = pd.factorize(s)
        e = emb.encode(list(uniq))
        out[v] = e[codes]
    return out
