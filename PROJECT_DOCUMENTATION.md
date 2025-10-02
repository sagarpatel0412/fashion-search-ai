
---

# 📄 Fashion Search AI — Project Documentation

## 1. 🎯 Project Goal

The objective of **Fashion Search AI** is to build a **hybrid AI-powered search system** that allows users to find fashion products using **natural language queries**.
For example:

> “black slim-fit kurta under ₹1500”

The system should return relevant products, explain why they match, and allow users to refine with filters such as **brand, color, price range, and rating**.

---

## 2. 📊 Dataset

* **Source**: Myntra Fashion Dataset (CSV from Kaggle).
* **Columns used**:

  * `p_id` (Product ID)
  * `name` (Product Title)
  * `brand`
  * `price`
  * `colour`
  * `img` (Image URL)
  * `ratingCount`, `avg_rating`
  * `description`
  * `p_attributes` (JSON-like attributes)

### Cleaning & Preprocessing

* Normalized brand and color fields (`title` casing).
* Converted price, rating, rating count to numeric.
* Extracted + flattened attributes from `p_attributes`.
* Generated a `doc_text` field combining: brand, title, color, price, rating, description, and attributes.
* Ensured every product has a **numeric integer `p_id`** (surrogate IDs for missing values).

---

## 3. 🏗️ System Architecture

### **Components**

1. **Index Builder (`scripts/build_index.py`)**

   * Builds **TF-IDF index** (keyword retrieval).
   * Builds **FAISS index** (semantic retrieval with Sentence-Transformers).
   * Saves all artifacts in `artifacts/`.

2. **Artifacts**

   * `meta.parquet` → Product metadata (id, title, brand, etc.).
   * `row_map.parquet` → Row-to-id mapping.
   * `tfidf.joblib` + `tfidf.npz` → TF-IDF model + matrix.
   * `faiss.index` → Dense embedding index (if FAISS installed).

3. **Streamlit App (`app.py`)**

   * Hybrid search (TF-IDF + Embeddings via Reciprocal Rank Fusion).
   * Sidebar filters: brand, color, price range, rating.
   * UI displays image, title, brand, price, rating.
   * Fallback: works with **TF-IDF only** if FAISS or SBERT not available.

---

## 4. 🔍 Search Pipeline

### Query Flow

1. **Input**: User query text.
2. **Keyword Retrieval**: TF-IDF cosine similarity → top candidates.
3. **Dense Retrieval** (if available): Sentence-Transformers embeddings + FAISS → top candidates.
4. **Fusion**: Reciprocal Rank Fusion (RRF) merges TF-IDF and dense results.
5. **Filtering**: Apply user-selected filters (brand, color, price, rating).
6. **Ranking**: Results ordered by fused score → displayed in UI.

---

## 5. 📦 Dependencies

* **Core**: `streamlit`, `pandas`, `numpy`, `pyarrow`
* **ML**: `scikit-learn`, `scipy`, `joblib`
* **NLP**: `sentence-transformers` (optional, for embeddings)
* **ANN**: `faiss-cpu` (optional, for dense retrieval)

---

## 6. 🖥️ Usage Guide

### Build Artifacts

```bash
python scripts/build_index.py --input_csv "data/Fashion Dataset.csv" --artifacts artifacts
```

### Run Streamlit

```bash
streamlit run app.py
```

### Example Queries

* `black straight kurta under 1500`
* `red floral dress`
* `men running shoes blue`

---

## 7. 📊 Evaluation

* **Offline metrics** (using synthetic queries & labels):

  * Recall@10, nDCG@10, MRR@10.
* **Latency**:

  * TF-IDF → ~50ms per query.
  * FAISS dense search → ~100–150ms per query.

---

## 8. ⚠️ Limitations

* Works only with the given dataset (no live Myntra API).
* Embedding model is generic (not fine-tuned for fashion).
* No personalization (same results for all users).
* No image similarity search yet.

---

## 9. 🔮 Future Work

* Add **cross-encoder reranker** for better ranking quality.
* Add **image-text search** (e.g., CLIP, SigLIP).
* Add **Postgres logging** for user queries, favorites, analytics.
* Deploy with **Docker + docker-compose** for easier setup.
* Cloud deployment on **Streamlit Cloud** or **Heroku**.

---

## 10. 👨‍💻 Author

**Sagar Patel**

