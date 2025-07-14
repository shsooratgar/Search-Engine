# Persian News Information Retrieval System

This project is a multi-phase Information Retrieval (IR) system for Persian news articles. It includes:
- **Phase 1**: Boolean retrieval using positional indexing.
- **Phase 2**: Ranked retrieval using tf-idf weighting and champion lists.
- **Phase 3**: Semantic search and document classification using Word2Vec embeddings and clustering.

---

## 🗂 Project Structure

**Phase 1**: Implements Boolean IR using positional indexing for exact and phrase search.
**Phase 2**: Implements Ranked IR using TF-IDF weighting and Champion Lists for efficient scoring.
**Phase 2 Extended**: Compares Word2Vec models on embedding-based ranking.
**Phase 3**:  Implements:
  - Document embedding using pretrained Word2Vec.
  - Clustering using vector similarity.
  - Query-based semantic retrieval within clusters.
  - Classification of unlabeled news using KNN over embeddings.

---

## 🧠 Dependencies

Install required Python packages:

```bash
pip install pandas hazm numpy gensim matplotlib
```

You also need the following data files:
- `IR00_3_11k News.xlsx`, `IR00_3_17k News.xlsx`, `IR00_3_20k News.xlsx`
- `IR1_7k_news.xlsx`
- A pretrained Word2Vec model: `w2v_150k_hazm_300_v2.model`

---

## 🚀 Running the Code

### Phase 1: Boolean Retrieval

```bash
python main.py
```
- Input a single or multi-word Persian query.
- Returns top 10 matching documents using positional index.

---

### Phase 2: Ranked Retrieval

```bash
python "main (1).py"
```
- Input a query.
- Returns documents ranked by cosine similarity using TF-IDF and Champion Lists.

---

### Phase 2 Extended: Model Comparison

```bash
python "main (2).py"
```
- Compares results from a trained Word2Vec model vs a given model.
- Outputs top results for each model for a given query.

---

### Phase 3: Semantic Clustering and Classification

```bash
python phase3.py
```

- **Input**: One of:
  - `"1"` → Run document clustering and semantic retrieval.
    - Then enter a query to retrieve top-k results from top-b clusters.
  - `"2"` → Run document classification using KNN on embeddings.
    - Then enter a query with a category tag (e.g., `... cat : sport`) to search within that class.

> *Note: Replace `phase3.py` with your actual filename for this part.*

---

## 📊 Techniques Used

- **Hazm** for Persian text normalization and stemming
- **TF-IDF** and **Positional Indexing**
- **Champion Lists** for faster ranking
- **Word2Vec Embeddings** (pretrained and trained)
- **Clustering** using centroid similarity
- **KNN Classification** over document vectors

---

## 📁 Data

- News articles in Persian from `.xlsx` files
- Each document contains: `title`, `content`, `url`, and (for some) `topic`

---

## 🧪 Example Usage

**Query (Boolean search)**:
```
تحریم‌های آمریکا
```

**Query (Semantic search)**:
```
نفت قیمت بالا cat : economy
```

---

## 📌 Notes

- All processing assumes Persian input and uses Hazm tools.
- Be sure to load or train the Word2Vec model before semantic features.
- You can adjust constants like `k`, `K`, `cent_count`, and `epochs` to fine-tune results.

---

## 📝 Authors

- Developed as part of an Information Retrieval course project.
