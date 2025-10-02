# scripts/build_index.py
import os, json, re, argparse, ast
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

def coalesce(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
        if v not in (None, "", float("nan")):
            return v
    return None

def parse_attrs(x):
    if isinstance(x, dict):
        return x
    if isinstance(x, str) and x.strip():
        try:
            s = x.strip()
            if s.startswith("{") and "'" in s and '"' not in s:
                s = s.replace("'", '"')
            return json.loads(s)
        except Exception:
            try:
                return ast.literal_eval(x)
            except Exception:
                return {}
    return {}

def flatten_attrs(d):
    if not isinstance(d, dict):
        return ""
    items = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(map(str, v))
        items.append(f"{k}: {v}")
    return ". ".join(items)

def normalize_text(s):
    if s is None: return ""
    s = str(s)
    s = re.sub(r"<br\s*/?>", " ", s, flags=re.I)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def fuse_text(row):
    attrs = flatten_attrs(row.get("p_attributes_parsed", {}))
    parts = [
        str(row.get("brand","")),
        str(row.get("name","")),
        f"Color: {row.get('colour','')}",
        f"Price: ₹{row.get('price','')}",
        f"Avg rating: {row.get('avg_rating','')} (count {row.get('ratingCount','')})",
        normalize_text(row.get("description","")),
        attrs
    ]
    return ". ".join([p for p in parts if p and str(p) != "nan"])

def main(args):
    os.makedirs(args.artifacts, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    keep = ['p_id','name','brand','price','colour','img','ratingCount','avg_rating','description','p_attributes']
    for c in keep:
        if c not in df.columns:
            df[c] = None

    df['p_attributes_parsed'] = df['p_attributes'].apply(parse_attrs)
    df['colour'] = df['colour'].astype(str).str.title()
    df['brand'] = df['brand'].astype(str).str.title()
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['avg_rating'] = pd.to_numeric(df['avg_rating'], errors='coerce')
    df['ratingCount'] = pd.to_numeric(df['ratingCount'], errors='coerce')

    df['doc_text'] = df.apply(fuse_text, axis=1)
    df = df[df['doc_text'].str.len() > 0].reset_index(drop=True)

    p_id_num = pd.to_numeric(df['p_id'], errors='coerce')

    current_max = int(pd.Series(p_id_num).dropna().max() or 0)
    miss = p_id_num.isna().sum()
    if miss > 0:
        p_id_num.loc[p_id_num.isna()] = np.arange(current_max + 1, current_max + 1 + miss)

    df['p_id'] = p_id_num.astype('int64')

    meta_cols = ['p_id','name','brand','price','colour','img','ratingCount','avg_rating']
    meta = df[meta_cols].copy()
    meta.to_parquet(os.path.join(args.artifacts, "meta.parquet"))

    row_map = pd.DataFrame({"row_id": np.arange(len(df), dtype=np.int64), "p_id": df['p_id'].astype('int64')})
    row_map.to_parquet(os.path.join(args.artifacts, "row_map.parquet"))

    tfidf = TfidfVectorizer(min_df=3, max_df=0.6, ngram_range=(1,2))
    X = tfidf.fit_transform(df['doc_text'].tolist())
    joblib.dump(tfidf, os.path.join(args.artifacts, "tfidf.joblib"))
    sparse.save_npz(os.path.join(args.artifacts, "tfidf.npz"), X)

    faiss_index_path = os.path.join(args.artifacts, "faiss.index")
    try:
        import faiss
        from sentence_transformers import SentenceTransformer

        model_name = args.model_name or "sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name)
        embs = model.encode(
            df['doc_text'].tolist(),
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embs = np.asarray(embs, dtype="float32")
        faiss_index = faiss.IndexFlatIP(embs.shape[1])
        faiss_index.add(embs)
        faiss.write_index(faiss_index, faiss_index_path)
        print(f"Saved FAISS index to {faiss_index_path}")
    except Exception as e:
        print("Skipping FAISS (dense embeddings) – reason:", e)

    print(f"Indexed {len(df)} products")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, required=True, help="Path to Fashion Dataset CSV")
    parser.add_argument("--artifacts", type=str, default="artifacts", help="Output dir for indices")
    parser.add_argument("--model_name", type=str, default=None)
    args = parser.parse_args()
    main(args)
