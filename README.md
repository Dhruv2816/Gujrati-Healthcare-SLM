# Gujarati Healthcare AI Assistant (GraphRAG Edition)

A localized Small Language Model (SLM) for the Gujarati language, heavily focused on providing safe and grounded healthcare information. It leverages a modern **GraphRAG** architecture—combining a Neo4j Knowledge Graph, ChromaDB semantic vector search, and a Redis caching layer—to answer medical queries effectively.

## 🛠️ Architecture

* **LLM:** Qwen 2.5 3B-Instruct (QLoRA fine-tuned on a custom Gujarati medical dataset).
* **Knowledge Graph:** Neo4j (Entity relationships: `HAS_SYMPTOM`, `TREATED_BY`, etc., extracted automatically from Medical text).
* **Vector DB:** ChromaDB (Semantic search over chunked medical PDFs using `multilingual-e5-large`).
* **Caching:** Redis (Caches exact queries and full responses for near-instant retrieval).
* **UI:** Streamlit dashboard & CLI.

---

## 🚀 Complete Pipeline Setup Guide (Start-to-End)

Follow these exact steps to build your production-grade Gujarati Healthcare AI from scratch:

### Phase 1: Environment Setup
1. **Prerequisites**: Ensure you have Python 3.10+, Docker, and Docker Compose installed.
2. **Setup Infrastructure**:
   ```bash
   docker-compose up -d
   ```
   *(This starts Neo4j on port 7474/7687 and Redis on port 6379)*
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

### Phase 2: Data Collection & Preparation
1. **Raw Data**: Run `notebooks/01_hf_download.ipynb` to download generic EN-GU parallel health corpuses.
2. **Cleaning**: Run `notebooks/02_data_collection_cleaning.ipynb` to filter out non-medical noise and structure the raw data.
3. **Medical Textbooks**: Purchase or acquire medical textbooks (e.g., Davidson's Medicine) in PDF format and place them in `data/books/`.

### Phase 3: GraphRAG Ingestion (Knowledge Graph + Vector DB)
1. **Knowledge Graph Extraction**: Run `notebooks/03_knowledge_graph.ipynb`.
   * This extracts single sentences from your PDFs, identifies medical entities using bilingual Python dictionaries (`src/kg/entity_extractor.py`), and establishes rich weighted relationships in Neo4j.
2. **Verify Neo4j Graph**: Check if your graph was built correctly by running:
   ```bash
   python scripts/verify_kg.py
   ```
3. **Vector Ingestion**: Run `python main.py --ingest-books` to chunk the PDFs and store dense semantic embeddings into ChromaDB.

### Phase 4: Golden Dataset Generation
1. **Grounded QA Creation**: Run `notebooks/04_dataset_creation.ipynb`.
   * **Crucial Step**: This script will query your newly built Neo4j Knowledge Graph to automatically synthesize thousands of fact-grounded Q&A pairs (e.g., extracting symptoms for a specific disease directly from the graph).
   * It also formats translation and summarization pairs, saving the final splits as `data/train.jsonl` and `data/test.jsonl`.
2. **Verify Dataset**: 
   ```bash
   python scripts/sample_dataset.py
   ```

### Phase 5: Local Fine-Tuning (16GB+ GPU Required)
1. **Train Model**: Run `notebooks/05_qlora_finetune.ipynb`.
   * We use Unsloth with RSLoRA (Rank 64) and Sequence Packing for hyper-efficient fine-tuning on a 16GB GPU.
   * This fine-tunes `Qwen/Qwen2.5-3B-Instruct` specifically on your `train.jsonl` Gujarati healthcare data.
2. **Export Adapter**: The notebook will save the LoRA adapter weights to `models/qwen_gu_health_lora/`.

### Phase 6: Run Inference!
1. **Start Ollama Engine**: Ensure Ollama is running locally and the base target model is available (`ollama run qwen2.5:3b`).
2. **Run Streamlit App**:
   ```bash
   streamlit run app/streamlit_app.py
   ```
3. **Run Terminal Check**:
   ```bash
   python test_rag.py
   ```

---

## 📁 Project Structure

```text
.
├── notebooks/                  # Pipeline: 01 to 05 (Data -> Graph -> Dataset -> Finetune)
├── scripts/                    # Utilities: Patchers and Verification scripts
├── src/                        # Modular GraphRAG backend package
│   ├── kg/                     # Neo4j client and Bilingual entity extractor
│   ├── vectordb/               # ChromaDB client 
│   ├── cache/                  # Redis TTL Cache client
│   ├── retriever/              # Hybrid Retrieval (Graph + Vector + Cache + Emergency)
│   └── pipeline/               # End-to-end Inference pipeline (Ollama Wrapper)
├── app/
│   └── streamlit_app.py        # Streamlit web interface
├── data/
│   └── books/                  # Drop medical PDF textbooks here!
├── docker-compose.yml          # Infrastructure setup
└── requirements.txt            # Python dependencies
```

---
**Disclaimer:** *For educational purposes only. Always consult a certified medical professional for healthcare advice.*
