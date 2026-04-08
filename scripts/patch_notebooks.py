import json
import os

NOTEBOOK_03 = 'notebooks/03_knowledge_graph.ipynb'
NOTEBOOK_05 = 'notebooks/05_qlora_finetune.ipynb'

def patch_nb_03():
    with open(NOTEBOOK_03, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Completely replace the cell doing naive extraction with one using our new src.kg modules
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "edges_symptom_disease = defaultdict(int)" in "".join(cell.get('source', [])):
            cell['source'] = [
                "import os\n",
                "import re\n",
                "from src.kg.entity_extractor import extract_entities\n",
                "from src.kg.neo4j_client import Neo4jClient\n",
                "\n",
                "# Use NLTK or regex for single sentence splitting (sliding window)\n",
                "sentences = [s.strip() for s in re.split(r'[\\.\\n]', full_corpus) if len(s.strip()) > 15]\n",
                "\n",
                "edges_to_write = []\n",
                "for i, sentence in enumerate(sentences):\n",
                "    entities = extract_entities(sentence)\n",
                "    \n",
                "    # Build rich edges based on extracted types\n",
                "    # Diseases to Symptoms\n",
                "    for d in entities.diseases:\n",
                "        for s in entities.symptoms:\n",
                "            edges_to_write.append({\n",
                "                'source': d, 'source_type': 'Disease',\n",
                "                'target': s, 'target_type': 'Symptom',\n",
                "                'rel_type': 'HAS_SYMPTOM', 'confidence': 0.8, 'doc_id': f'sent_{i}'\n",
                "            })\n",
                "    # Diseases to Treatments (Drugs vs Treatments map to TREATED_BY/MANAGED_BY inside client)\n",
                "    for d in entities.diseases:\n",
                "        for t in entities.drugs + entities.treatments:\n",
                "            edges_to_write.append({\n",
                "                'source': d, 'source_type': 'Disease',\n",
                "                'target': t, 'target_type': 'Treatment',\n",
                "                'confidence': 0.9, 'doc_id': f'sent_{i}'\n",
                "            })\n",
                "\n",
                "print(f\"Generated {len(edges_to_write)} rich edges from sentences.\")\n"
            ]

        if cell['cell_type'] == 'code' and "import networkx" in "".join(cell.get('source', [])):
            cell['source'] = [
                "import os\n",
                "from dotenv import load_dotenv\n",
                "load_dotenv()\n",
                "neo4j_pass = os.environ.get('NEO4J_PASSWORD', 'password')\n",
                "neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')\n",
                "\n",
                "client = Neo4jClient(neo4j_uri, 'neo4j', neo4j_pass)\n",
                "client.setup_schema()\n",
                "\n",
                "print(\"Writing batch to Neo4j...\")\n",
                "result = client.write_graph_batch(edges_to_write)\n",
                "print(f\"Processed relationships: {result.get('processed')}\")\n"
            ]

    with open(NOTEBOOK_03, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print(f"Patched {NOTEBOOK_03}")

def patch_nb_05():
    with open(NOTEBOOK_05, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        source_text = "".join(cell.get('source', []))
        if cell['cell_type'] == 'code' and "Qwen/Qwen2.5-2B-Instruct" in source_text:
            new_source = []
            for line in cell['source']:
                if "Qwen/Qwen2.5-2B-Instruct" in line:
                    new_source.append(line.replace("Qwen/Qwen2.5-2B-Instruct", "Qwen/Qwen2.5-3B-Instruct"))
                else:
                    new_source.append(line)
            cell['source'] = new_source

        if cell['cell_type'] == 'code' and "r=16" in source_text and "use_rslora=False" in source_text:
            new_source = []
            for line in cell['source']:
                if "r=16" in line:
                    new_source.append(line.replace("r=16", "r=64"))
                elif "use_rslora=False" in line:
                    new_source.append(line.replace("use_rslora=False", "use_rslora=True"))
                else:
                    new_source.append(line)
            cell['source'] = new_source

        if cell['cell_type'] == 'code' and "packing=False" in source_text:
            new_source = []
            for line in cell['source']:
                if "packing=False" in line:
                    new_source.append(line.replace("packing=False", "packing=True"))
                elif "num_train_epochs=2" in line:
                    new_source.append(line.replace("num_train_epochs=2", "num_train_epochs=4"))
                else:
                    new_source.append(line)
            cell['source'] = new_source

    with open(NOTEBOOK_05, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print(f"Patched {NOTEBOOK_05}")

if __name__ == '__main__':
    try:
        patch_nb_03()
        patch_nb_05()
        print("Success.")
    except Exception as e:
        print(f"Error: {e}")
