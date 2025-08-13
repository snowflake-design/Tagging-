#!/usr/bin/env python3

import json
import time
import logging
import base64
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# LangChain components for Stage 1
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Visualization library for Stage 4
from pyvis.network import Network
from bs4 import BeautifulSoup

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    """
    Mocks a blocking LLM API call. It returns a pre-defined JSON string
    based on keywords in the prompt to simulate the final hierarchical logic.
    """
    logging.info("Simulating blocking LLM API call...")
    time.sleep(0.5)

    # --- MOCK FOR STAGE 2: TOPIC TAGGING ---
    if "main_topic" in prompt and "Artificial intelligence" in prompt:
        return json.dumps({
            "main_topic": "AI's Industrial Transformation",
            "summary": "AI, ML, and NLP are rapidly advancing and transforming industries worldwide.",
            "tags": ["AI Transformation", "Machine Learning", "NLP", "Transformer Models", "Computer Vision"]
        })
    if "main_topic" in prompt and "ethical considerations" in prompt:
        return json.dumps({
            "main_topic": "AI Advancements and Challenges",
            "summary": "The deployment of AI raises significant ethical concerns like algorithmic bias and data privacy.",
            "tags": ["Ethical AI", "Algorithmic Bias", "Data Privacy", "Regulatory Frameworks", "Unfair Outcomes"]
        })
    if "main_topic" in prompt and "Climate change" in prompt:
        return json.dumps({
            "main_topic": "Climate Change Impacts",
            "summary": "Climate change is a pressing global challenge with severe environmental consequences.",
            "tags": ["Climate Change", "Global Crisis", "Rising Temperatures", "Extreme Weather", "Sea Level Rise"]
        })
    if "main_topic" in prompt and "Renewable energy" in prompt:
        return json.dumps({
            "main_topic": "Renewable Energy Solutions",
            "summary": "Renewable energy technologies like solar and wind are key solutions to the climate crisis.",
            "tags": ["Renewable Energy", "Solar & Wind", "Energy Storage", "Cost-Effective", "Smart Grid"]
        })

    # --- MOCK FOR STAGE 3: HIERARCHICAL SYNTHESIS ---
    if "You are an expert Information Architect" in prompt:
        return json.dumps({
          "nodes": [
            {"id": "Document Analysis", "label": "Document Analysis", "group": "root", "size": 35},
            {"id": "Artificial Intelligence", "label": "Artificial Intelligence", "group": "parent", "size": 25},
            {"id": "Environmental Issues", "label": "Environmental Issues", "group": "parent", "size": 25},
            {"id": "AI Development & Impact", "label": "AI Development & Impact", "group": "child", "size": 20},
            {"id": "Climate Response", "label": "Climate Response", "group": "child", "size": 20},
            {"id": "AI's Industrial Transformation", "label": "AI's Industrial Transformation", "group": "grandchild", "size": 15},
            {"id": "AI Advancements and Challenges", "label": "AI Advancements and Challenges", "group": "grandchild", "size": 15},
            {"id": "Climate Change Impacts", "label": "Climate Change Impacts", "group": "grandchild", "size": 15},
            {"id": "Renewable Energy Solutions", "label": "Renewable Energy Solutions", "group": "grandchild", "size": 15}
          ],
          "edges": [
            {"source": "Document Analysis", "target": "Artificial Intelligence"},
            {"source": "Document Analysis", "target": "Environmental Issues"},
            {"source": "Artificial Intelligence", "target": "AI Development & Impact"},
            {"source": "Environmental Issues", "target": "Climate Response"},
            {"source": "AI Development & Impact", "target": "AI's Industrial Transformation"},
            {"source": "AI Development & Impact", "target": "AI Advancements and Challenges"},
            {"source": "Climate Response", "target": "Climate Change Impacts"},
            {"source": "Climate Response", "target": "Renewable Energy Solutions"}
          ],
          "consolidation_map": {
            "AI Development & Impact": ["AI's Industrial Transformation", "AI Advancements and Challenges"],
            "Climate Response": ["Climate Change Impacts", "Renewable Energy Solutions"]
          }
        })

    return "{}"


# ----------------------------------------------------------------------------
# STAGE 1: SEMANTIC CHUNKING
# ----------------------------------------------------------------------------
def get_semantic_chunks(text: str, model_path="all-mpnet-base-v2", threshold=80) -> List[str]:
    logging.info(f"🤖 Loading embedding model: {model_path}")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True}
    )
    text_splitter = SemanticChunker(
        embeddings=embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=threshold
    )
    logging.info(f"🔄 Chunking text with {threshold}th percentile threshold...")
    chunks = text_splitter.split_text(text)
    logging.info(f"✅ Created {len(chunks)} chunks.")
    return chunks

# ----------------------------------------------------------------------------
# STAGE 2: AUTOMATED TOPIC TAGGING
# ----------------------------------------------------------------------------
def generate_topic_for_chunk(chunk: str) -> Dict:
    prompt = f"""
    You are an expert data analyst. For the following text chunk, create a JSON object.
    The "main_topic" should be a short, clear title for the chunk, like a section heading.

    ```json
    {{
      "main_topic": "A concise title for this chunk (3-5 words).",
      "summary": "A one-sentence summary of the chunk's main point.",
      "tags": ["A list of 4-5 specific keywords or phrases found in the chunk."]
    }}
    ```

    # Text Chunk to Analyze:
    {chunk}

    # Your JSON Output:
    """
    response_str = abc_response(prompt)
    try:
        data = json.loads(response_str)
        data['original_chunk'] = chunk
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for chunk: {chunk[:50]}...")
        return {}

# ----------------------------------------------------------------------------
# STAGE 3: HIERARCHICAL SYNTHESIS
# ----------------------------------------------------------------------------
def generate_hierarchical_graph(tagged_data: List[Dict], doc_title: str) -> Dict:
    main_topics = [item.get('main_topic', '') for item in tagged_data if item.get('main_topic')]
    topics_list_str = "\n- ".join(main_topics)

    prompt = f"""
    You are an expert Information Architect building a detailed, multi-level outline of a document titled '{doc_title}'.
    Your task has three steps:

    Step 1: CONSOLIDATE. Look at the 'List of Original Topics' below. Group semantically similar topics together and create a single, representative 'Consolidated Topic' label for each group.

    Step 2: CATEGORIZE. Take your new list of 'Consolidated Topics' and organize them under 2-4 high-level 'Parent Categories' that you invent.

    Step 3: STRUCTURE. Generate a single JSON object containing three keys: 'nodes', 'edges', and a 'consolidation_map'. The map must link each new 'Consolidated Topic' to the list of 'Original Topics' it represents.

    Follow the format below precisely.

    ```json
    {{
      "nodes": [
        {{"id": "Root Node ID", "label": "Root Node Label", "group": "root"}},
        {{"id": "Parent Category", "label": "Parent Category", "group": "parent"}},
        {{"id": "Consolidated Topic", "label": "Consolidated Topic", "group": "child"}},
        {{"id": "Original Topic 1", "label": "Original Topic 1", "group": "grandchild"}}
      ],
      "edges": [
        {{"source": "Root Node ID", "target": "Parent Category"}},
        {{"source": "Parent Category", "target": "Consolidated Topic"}},
        {{"source": "Consolidated Topic", "target": "Original Topic 1"}}
      ],
      "consolidation_map": {{
        "Consolidated Topic A": ["Original Topic 1", "Original Topic 2"],
        "Consolidated Topic B": ["Original Topic 3"]
      }}
    }}
    ```

    # List of Original Topics to Organize:
    - {topics_list_str}

    # Your JSON Graph Output:
    """
    logging.info("🧠 Consolidating topics and synthesizing multi-level hierarchy...")
    response_str = abc_response(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON for the hierarchical graph.")
        return {"nodes": [], "edges": []}


# ----------------------------------------------------------------------------
# STAGE 4: VISUALIZATION
# ----------------------------------------------------------------------------
def create_flow_diagram(graph_data: Dict, tagged_data: List[Dict], filename: str = "flow_diagram.html"):
    if not graph_data.get("nodes"):
        logging.warning("No nodes found. Skipping visualization.")
        return

    net = Network(
        height="90vh",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        notebook=False,
        directed=True,
        cdn_resources='in_line'
    )
    
    net.set_options("""
    var options = {
      "physics": {
        "solver": "hierarchicalRepulsion",
        "hierarchicalRepulsion": {
          "nodeDistance": 220,
          "springLength": 150
        }
      }
    }
    """)

    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])
    consolidation_map = graph_data.get('consolidation_map', {})

    for node_data in nodes:
        node_id = node_data['id']
        label = node_data['label']
        group = node_data.get('group')
        size = node_data.get("size", 15)

        color_map = {
            "root": "#8B0000",
            "parent": "#FF4500",
            "child": "#B22222",
            "grandchild": "#1E90FF"
        }
        color = color_map.get(group, "#808080")
        
        hover_title = ""
        if group == 'child' and label in consolidation_map:
            all_summaries = []
            all_tags = set()
            original_topics = consolidation_map[label]
            for topic in original_topics:
                chunk_info = next((item for item in tagged_data if item.get('main_topic') == topic), None)
                if chunk_info:
                    all_summaries.append(chunk_info.get('summary', ''))
                    for tag in chunk_info.get('tags', []):
                        all_tags.add(tag)
            
            hover_title += "<b>Consolidated Summary:</b><br>" + "<br>".join(f"- {s}" for s in all_summaries)
            hover_title += "<br><br><b>Combined Tags:</b><ul>" + "".join(f"<li>{tag}</li>" for tag in sorted(list(all_tags))) + "</ul>"

        elif group == 'grandchild':
            chunk_info = next((item for item in tagged_data if item.get('main_topic') == label), None)
            if chunk_info:
                summary = chunk_info.get('summary', 'No summary.')
                tags_html = "<ul>" + "".join(f"<li>{tag}</li>" for tag in chunk_info.get('tags', [])) + "</ul>"
                chunk_text_html = chunk_info.get('original_chunk', '').replace('\n', '<br>')
                hover_title = f"<b>Summary:</b> {summary}<br><br><b>Tags:</b>{tags_html}"

        net.add_node(node_id, label=label, size=size, color=color, title=hover_title)

    for edge in edges:
        net.add_edge(edge['source'], edge['target'])

    net.write_html(filename, open_browser=False)
    logging.info(f"📦 Self-contained diagram with merged details saved as '{filename}'.")


# ----------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------------
def main_pipeline(text: str, doc_title: str):
    """Runs the full end-to-end pipeline."""
    chunks = get_semantic_chunks(text)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        tagged_data = list(executor.map(generate_topic_for_chunk, chunks))

    graph_data = generate_hierarchical_graph(tagged_data, doc_title)
    
    create_flow_diagram(graph_data, tagged_data)


if __name__ == "__main__":
    input_paragraph = """
    Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

    However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

    Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

    Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
    """
    main_pipeline(input_paragraph, doc_title="Document Analysis")
