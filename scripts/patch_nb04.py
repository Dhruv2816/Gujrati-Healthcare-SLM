import json

NOTEBOOK_04 = 'notebooks/04_dataset_creation.ipynb'

def patch_nb_04():
    with open(NOTEBOOK_04, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        source_text = "".join(cell.get('source', []))
        if cell['cell_type'] == 'code' and "offline KG is deprecated" in source_text:
            cell['source'] = [
                "import os\n",
                "import random\n",
                "from src.kg.neo4j_client import Neo4jClient\n",
                "from dotenv import load_dotenv\n",
                "\n",
                "load_dotenv()\n",
                "neo4j_pass = os.environ.get('NEO4J_PASSWORD', 'password')\n",
                "neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')\n",
                "\n",
                "client = Neo4jClient(neo4j_uri, 'neo4j', neo4j_pass)\n",
                "all_diseases = client.query(\"MATCH (d:Entity {type: 'Disease'}) RETURN d.name as name\")\n",
                "disease_names = [d['name'] for d in all_diseases]\n",
                "\n",
                "# KG-Grounded QA Generation\n",
                "QA_TEMPLATES = [\n",
                "    (\"મને {disease} વિશે જણાવો.\", \"તેના લક્ષણો અને સારવાર અંગેની માહિતી નીચે મુજબ છે:\\n{context}\"),\n",
                "    (\"{disease} ના લક્ષણો શું છે?\", \"તેના સામાન્ય લક્ષણો છે:\\n{context}\"),\n",
                "    (\"શું {disease} માટે કોઈ ઈલાજ છે?\", \"હા, તેની સારવાર આ રીતે થઈ શકે છે:\\n{context}\")\n",
                "]\n",
                "\n",
                "for disease in tqdm(disease_names, desc='KG QA'):\n",
                "    context = client.get_kg_context([disease])\n",
                "    if context:\n",
                "        template = random.choice(QA_TEMPLATES)\n",
                "        qa_examples.append({\n",
                "            'task': 'qa_kg',\n",
                "            'instruction': template[0].format(disease=disease.capitalize()),\n",
                "            'output': template[1].format(context=context) + ' ' + get_safety_sentence()\n",
                "        })\n",
                "\n",
                "print(f'✅ Total QA examples (KG Grounded): {len(qa_examples):,}')\n"
            ]

    with open(NOTEBOOK_04, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4)
        print("Patched notebook 04 successfully.")

if __name__ == '__main__':
    patch_nb_04()
