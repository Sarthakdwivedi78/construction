import json
import math
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())

class Index:
    def __init__(self, hybrid=False, path=ROOT / "knowledge.json", dense_only=False, dense_weight=0.5, reranker_model=None):
        if not 0 < dense_weight < 1:
            raise ValueError("Dense weight must be between zero and one.")
        self.dense_only = dense_only
        self.dense_weight = dense_weight
        hybrid = hybrid or dense_only
        self.rows = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        ids = [row["hazard_id"] for row in self.rows]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("The corpus must contain unique provision IDs.")
        self.texts = [" ".join(str(value) for key, value in row.items() if key != "effective_from") for row in self.rows]
        self.docs = [Counter(tokens(text)) for text in self.texts]
        self.df = Counter(term for doc in self.docs for term in doc)
        self.avg = sum(map(len, map(tokens, self.texts))) / len(self.rows)
        self.encoder = None
        self.reranker = None
        if reranker_model:
            if not hybrid or dense_only:
                raise ValueError("Cross-encoder reranking requires hybrid retrieval.")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(reranker_model, device="cpu")
        if hybrid:
            import faiss
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(os.environ.get("SAFETY_BGE_MODEL", "BAAI/bge-small-en-v1.5"), device="cpu")
            matrix = self.encoder.encode(self.texts, normalize_embeddings=True)
            self.dense = faiss.IndexFlatIP(matrix.shape[1])
            self.dense.add(matrix)

    def save_dense(self, path):
        import faiss
        if self.encoder is None:
            raise ValueError("Build a dense index before exporting.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.dense, str(path))
        path.with_suffix(".json").write_text(
            json.dumps([row["hazard_id"] for row in self.rows], indent=2), encoding="utf-8")

    def search(self, query, category, authority="Unknown", k=4, state="", inspection_date=""):
        jurisdictions = {"India", authority, "Historical Central"}
        if authority == "State" and state.strip():
            jurisdictions.add(state.strip())
        eligible = {i for i, row in enumerate(self.rows)
                    if row["hazard_category"] == category
                    and row["jurisdiction"] in jurisdictions
                    and (not inspection_date or row.get("effective_from", "") <= inspection_date)}
        def bm25(i):
            doc = self.docs[i]
            length = sum(doc.values())
            return sum(math.log1p((len(self.rows)-self.df[t]+0.5)/(self.df[t]+0.5))
                       * doc[t]*2.2/(doc[t]+1.2*(0.25+0.75*length/self.avg))
                       for t in set(tokens(query)) if doc[t])
        lexical = sorted(eligible, key=lambda i: (-bm25(i), i))
        rankings = [] if self.dense_only else [(lexical, 1 - self.dense_weight if self.encoder else 1)]
        if self.encoder is not None:
            vector = self.encoder.encode(
                ["Represent this sentence for searching relevant passages: " + query],
                normalize_embeddings=True)
            _, ids = self.dense.search(vector, len(self.rows))
            rankings.append(([int(i) for i in ids[0] if i in eligible], 1 if self.dense_only else self.dense_weight))
        scores = Counter()
        for ranking, weight in rankings:
            for rank, i in enumerate(ranking, 1):
                scores[i] += weight / (60 + rank)
        candidates = [i for i, _ in scores.most_common(max(k, 12) if self.reranker else k)]
        if self.reranker and candidates:
            relevances = self.reranker.predict([(query, self.texts[i]) for i in candidates])
            candidates = [i for i, _ in sorted(zip(candidates, relevances),
                                                key=lambda pair: float(pair[1]), reverse=True)]
        return [self.rows[i] for i in candidates[:k]]
