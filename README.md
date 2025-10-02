
---

# 🧠🔎 Fashion Search AI

A **hybrid product search system** built with **Streamlit**, combining **TF-IDF (keyword)** + **Sentence-Transformer embeddings (semantic)** via **Reciprocal Rank Fusion (RRF)**.
It indexes the Myntra dataset and lets users search products in natural language with filters for brand, color, price, and rating.

---

## 🚀 Features

* **Hybrid Search**: Combines semantic search (Sentence-Transformers + FAISS) with keyword TF-IDF.
* **Filters**: Brand, Color, Price Range, Minimum Rating.
* **Streamlit UI**: Simple, interactive product search with images.
* **Fallbacks**: If FAISS / SentenceTransformers are not available, runs in **TF-IDF only mode**.
* **Artifacts**: Indexes stored in `artifacts/` (parquet + TF-IDF + optional FAISS).

---

## 📂 Project Structure

```
fashion-search-ai/
│
├── app.py                   # Streamlit app
├── scripts/
│   └── build_index.py       # Script to build TF-IDF + FAISS indices
├── data/
│   └── Fashion Dataset.csv  # Myntra dataset (CSV)
├── artifacts/               # Generated indices (meta.parquet, tfidf, faiss)
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## ⚙️ Setup & Installation

### 1. Clone repo & create venv

```bash
cd fashion-search-ai
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Build search artifacts

```bash
python scripts/build_index.py --input_csv "data/Fashion Dataset.csv" --artifacts artifacts
```

This creates:

* `artifacts/meta.parquet`
* `artifacts/row_map.parquet`
* `artifacts/tfidf.joblib`
* `artifacts/tfidf.npz`
* `artifacts/faiss.index` *(only if FAISS is installed)*

### 3. Run the app

```bash
streamlit run app.py
```

---

## 🛠 Dependencies

Main requirements (see `requirements.txt` for exact versions):

* **streamlit**
* **pandas**, **numpy**
* **scikit-learn**, **scipy**, **joblib**
* **pyarrow** (for parquet I/O)
* **sentence-transformers** (for embeddings)
* **faiss-cpu** *(optional, for semantic retrieval)*

---

## 💡 Usage

* Enter queries like:

  * `"black straight kurta under 1500"`
  * `"red floral dress"`
  * `"men running shoes blue"`
* Refine results with sidebar filters (brand, color, price range, rating).

If FAISS or Sentence-Transformers aren’t installed, the app still works in **keyword (TF-IDF only)** mode.

---

## 📊 Future Extensions

* Add **cross-encoder reranker** for better ranking.
* Integrate **Postgres** to log queries & favorites.
* Deploy with **Docker** or **Streamlit Cloud**.
* Support **image-text multimodal search** (CLIP / SigLIP).

---

## 👨‍💻 Author

Built by Sagar Patel as part of the **Fashion Search AI** exploration.

