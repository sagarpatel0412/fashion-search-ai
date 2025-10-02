# app.py
import os, json
import numpy as np
import pandas as pd
import streamlit as st
import faiss
try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except Exception as e:
    _SBERT_OK = False
    _SBERT_ERR = str(e)
import joblib
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DATA_CSV = os.environ.get("DATA_CSV", "data/Fashion Dataset.csv")

st.set_page_config(page_title="Fashion Search AI", layout="wide")
st.title("🧠🔎 Fashion Search AI")
st.caption("Hybrid (Embeddings + TF-IDF) product search over your Myntra dataset.")

required_files = ["meta.parquet", "row_map.parquet", "tfidf.joblib", "tfidf.npz"]
missing = [f for f in required_files if not os.path.exists(os.path.join(ARTIFACTS, f))]
if missing:
    st.error("Search artifacts not found.")
    st.code(f'python scripts/build_index.py --input_csv "{DATA_CSV}" --artifacts "{ARTIFACTS}"')
    st.stop()

if not _SBERT_OK:
    st.info(
        "Running in **TF-IDF only** mode. To enable semantic search:\n"
        "```bash\npip install 'sentence-transformers>=2.7,<3.0'\n```"
        f"\n\n(Import error: `{_SBERT_ERR}`)"
    )

@st.cache_resource
def load_artifacts():
    meta = pd.read_parquet(os.path.join(ARTIFACTS, "meta.parquet"))
    row_map = pd.read_parquet(os.path.join(ARTIFACTS, "row_map.parquet"))
    tfidf = joblib.load(os.path.join(ARTIFACTS, "tfidf.joblib"))
    X = sparse.load_npz(os.path.join(ARTIFACTS, "tfidf.npz"))

    encoder = None
    faiss_index = None
    if _SBERT_OK and faiss is not None:
        try:
            encoder = SentenceTransformer(MODEL_NAME)
            faiss_index = faiss.read_index(os.path.join(ARTIFACTS, "faiss.index"))
        except Exception as e:
            encoder, faiss_index = None, None

    meta["p_id"] = pd.to_numeric(meta["p_id"], errors="coerce").astype("Int64")

    row_map["p_id"] = pd.to_numeric(row_map["p_id"], errors="coerce").astype("Int64")

    return meta, row_map, tfidf, X, faiss_index, encoder

def rrf(ranks, k=60):
    """Reciprocal Rank Fusion for lists of IDs."""
    scores = {}
    for rank_list in ranks:
        for r, pid in enumerate(rank_list):
            scores[int(pid)] = scores.get(int(pid), 0.0) + 1.0 / (k + r + 1.0)
    return sorted(scores.items(), key=lambda x: -x[1])

def hybrid_search(query, k_dense=200, k_final=40, filters=None):
    meta, row_map, tfidf, X, faiss_index, encoder = load_artifacts()

    qX = tfidf.transform([query])
    sims = cosine_similarity(qX, X).ravel()
    top_kw_idx = np.argpartition(-sims, min(k_dense, sims.size-1))[:k_dense]
    pids_kw = row_map.iloc[top_kw_idx]["p_id"].dropna().astype("int64").tolist()

    pids_dense = []
    if encoder is not None and faiss_index is not None:
        qv = encoder.encode([query], normalize_embeddings=True)
        D, I = faiss_index.search(np.asarray(qv, dtype="float32"), min(k_dense, faiss_index.ntotal))
        row_ids_dense = I[0].tolist()
        row_ids_dense = [rid for rid in row_ids_dense if rid is not None and rid >= 0]
        if row_ids_dense:
            pids_dense = row_map.iloc[row_ids_dense]["p_id"].dropna().astype("int64").tolist()

    fused = rrf([lst for lst in [pids_dense, pids_kw] if lst])
    candidate_pids = [pid for pid, _ in fused][:max(k_dense, k_final)]

    meta_pid = meta["p_id"]
    mask = meta_pid.isin(pd.Series(candidate_pids, dtype="int64"))
    df = meta[mask].copy()

    if filters:
        if filters.get("brands"):
            df = df[df["brand"].isin(filters["brands"])]
        if filters.get("colours"):
            df = df[df["colour"].isin(filters["colours"])]
        if filters.get("price"):
            lo, hi = filters["price"]
            df = df[(pd.to_numeric(df["price"], errors="coerce").fillna(0) >= lo) &
                    (pd.to_numeric(df["price"], errors="coerce").fillna(0) <= hi)]
        if filters.get("rating"):
            df = df[(pd.to_numeric(df["avg_rating"], errors="coerce").fillna(0) >= filters["rating"])]

    order = {int(pid): i for i, pid in enumerate(candidate_pids)}
    df["__ord"] = df["p_id"].map(order)
    df = df.sort_values("__ord", na_position="last").head(k_final).drop(columns="__ord")

    return df

meta, row_map, tfidf, X, faiss_index, encoder = load_artifacts()

with st.sidebar:
    st.header("Filters")
    brands = (
        meta["brand"].dropna().astype(str).str.title().value_counts().head(30).index.tolist()
        if "brand" in meta.columns else []
    )
    brand_sel = st.multiselect("Brand", sorted(brands), default=[])
    colours = (
        meta["colour"].dropna().astype(str).str.title().unique().tolist()
        if "colour" in meta.columns else []
    )
    colour_sel = st.multiselect("Colour", sorted(colours), default=[])
    pr_series = pd.to_numeric(meta.get("price", pd.Series(dtype=float)), errors="coerce")
    pr_min = int(np.nanmin(pr_series)) if pr_series.notna().any() else 0
    pr_max = int(np.nanmax(pr_series)) if pr_series.notna().any() else 10000
    pr_max = max(pr_max, 1000)
    price_sel = st.slider("Price Range (₹)", min_value=0, max_value=pr_max, value=(pr_min, pr_max), step=50)
    rating_sel = st.slider("Min Avg Rating", 0.0, 5.0, 0.0, 0.1)
    k_final = st.slider("Results to show", 10, 100, 40, 5)
    st.markdown("---")
    st.caption('Try: "black straight kurta under 1500", "red floral dress", "men running shoes blue".')

q = st.text_input("Search what you want ✨", value="black straight kurta under 1500")
if st.button("Search") or q:
    flt = {
        "brands": brand_sel or None,
        "colours": colour_sel or None,
        "price": price_sel,
        "rating": rating_sel if rating_sel > 0 else None,
    }
    with st.spinner("Searching..."):
        results = hybrid_search(q, k_dense=200, k_final=k_final, filters=flt)

    st.subheader(f"Top {len(results)} results")
    for _, row in results.reset_index(drop=True).iterrows():
        c1, c2, c3 = st.columns([1, 3, 2])
        with c1:
            img = row.get("img")
            if isinstance(img, str) and img.startswith("http"):
                st.image(img, use_container_width=True)
        with c2:
            st.markdown(f"**{row.get('brand','')} — {row.get('name','')}**")
            price_val = row.get("price")
            price_txt = f"₹{int(price_val):,}" if pd.notna(price_val) else "—"
            st.markdown(f"Colour: {row.get('colour','-')}  \nPrice: {price_txt}")
        with c3:
            avg_r = row.get("avg_rating")
            rc = row.get("ratingCount")
            try:
                avg_r = float(avg_r) if pd.notna(avg_r) else np.nan
            except Exception:
                avg_r = np.nan
            try:
                rc = int(rc) if pd.notna(rc) else 0
            except Exception:
                rc = 0
            st.markdown(f"⭐ {avg_r:.2f} ({rc} ratings)" if pd.notna(avg_r) else "⭐ —")
            if isinstance(img, str) and img.startswith("http"):
                st.markdown(f"[Open Image]({img})")
