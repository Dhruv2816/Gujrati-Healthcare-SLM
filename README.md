# Gujarati Healthcare AI Assistant (GraphRAG Edition)

A localized Small Language Model (SLM) for the Gujarati language, heavily focused on providing safe and grounded healthcare information. It leverages a modern **GraphRAG** architecture—combining a Neo4j Knowledge Graph, ChromaDB semantic vector search, and a Redis caching layer—to answer medical queries effectively.

## 🛠️ Architecture

* **LLM:** Qwen 2.5 3B (QLoRA fine-tuned on a custom Gujarati medical dataset).
* **Knowledge Graph:** Neo4j (Entities: Diseases, Symptoms, Drugs, Treatments extracted automatically from Medical text).
* **Vector DB:** ChromaDB (Semantic search over chunked medical PDFs using `multilingual-e5-large`).
* **Caching:** Redis (Caches exact queries and full responses for near-instant retrieval).
* **UI:** Premium Streamlit dashboard.

## 📁 Project Structure

```text
.
├── notebooks/                  # Step-by-step Jupyter notebooks for data processing, fine-tuning, and evaluation.
├── src/                        # Modular GraphRAG package
│   ├── config.py               # Central environment config (.env)
│   ├── kg/                     # Neo4j client and spaCy-based medical entity extractor
│   ├── vectordb/               # ChromaDB client for PDF ingestion and vector search
│   ├── cache/                  # Redis client for TTL-based caching
│   ├── retriever/              # Hybrid retrieval orchestration (Graph + Vector + Cache)
│   └── pipeline/               # End-to-end inference LLM pipeline
├── app/
│   └── streamlit_app.py        # Streamlit web interface
├── data/
│   └── books/                  # Drop medical PDF textbooks here for extraction
├── models/                     # Saved LoRA adapter weights go here
├── main.py                     # CLI Entry point for backend tasks
├── docker-compose.yml          # Infrastructure setup (Neo4j, Redis)
└── requirements.txt            # Python dependencies
```

## 🚀 Setup Instructions

### 1. Prerequisites
Ensure you have Docker and Docker Compose installed for handling the Neo4j and Redis infrastructure. Python 3.10+ is recommended.

### 2. Infrastructure Setup (Neo4j & Redis)
Start the background services:
```bash
docker-compose up -d
```
*Neo4j Browser will be available at http://localhost:7474.*

### 3. Environment Variables
Your `.env` file should look like this (add your Hugging Face Token):
```env
HF_TOKEN=your_huggingface_token
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=gujarati_health_neo4j
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=gujarati_health_redis
```

### 4. Install Python Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 5. Data Ingestion (GraphRAG)
Place your medical PDF books (e.g., Harrison's, Davidson's, Oxford Handbook) into the `data/books/` directory. Then ingest them into ChromaDB and Neo4j with one command:
```bash
python main.py --ingest-books
```
Check if the ingestion was successful:
```bash
python main.py --stats
```

### 6. Run the Application
Launch the elegant Streamlit UI to interact with your Medical Assistant:
```bash
streamlit run app/streamlit_app.py
```

Or query directly from your CLI terminal:
```bash
python main.py --query "ડાયાબિટીઝ ના લક્ષણો વિશે જણાવો?" --top-k 5
```

## 🧠 Fine-Tuning Pipeline
To train/fine-tune the model on your own lab computer or cloud GPU:
1. Run `notebooks/02_data_collection_cleaning.ipynb` and `notebooks/04_dataset_creation.ipynb` to build your training dataset (`data/train.jsonl`).
2. Run `notebooks/05_qlora_finetune.ipynb` on a machine with a capable GPU (e.g., 16GB VRAM) to learn Gujarati medical terminology and generate LoRA weights.
3. Move the weights to the `models/qwen_gu_health_lora` directory. The pipeline (`src/pipeline/inference.py`) will automatically detect and load your fine-tuned model instead of the base model.

---
**Disclaimer:** *For educational purposes only. Always consult a certified medical professional for healthcare advice.*
