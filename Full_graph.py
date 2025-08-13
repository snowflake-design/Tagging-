#!/usr/bin/env python3

import json
import time
import logging
from typing import List, Dict, Any

# For thread-based parallelization
from concurrent.futures import ThreadPoolExecutor

# LangChain components for Stage 1
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Visualization library for Stage 4
from pyvis.network import Network

# --- Configuration & Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION (Updated Stage 2 for more tags)
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    logging.info("Simulating blocking LLM API call...")
    time.sleep(0.5)

    # --- MOCK FOR STAGE 2: TOPIC TAGGING (More Tags) ---
    if "main_topic" in prompt and "Artificial intelligence" in prompt:
        return json.dumps({
            "main_topic": "AI Industry Transformation",
            "summary": "AI, ML, and NLP are rapidly advancing and transforming industries worldwide.",
            "tags": ["AI Transformation", "Machine Learning", "NLP", "Transformer Models", "Computer Vision"]
        })
    if "main_topic" in prompt and "ethical considerations" in prompt:
        return json.dumps({
            "main_topic": "Ethical AI & Data Privacy",
            "summary": "The deployment of AI raises significant ethical concerns like algorithmic bias and data privacy.",
            "tags": ["Ethical AI", "Algorithmic Bias", "Data Privacy", "Regulatory Frameworks", "Unfair Outcomes"]
        })
    if "main_topic" in prompt and "Climate change" in prompt:
        return json.dumps({
            "main_topic": "The Challenge of Climate Change",
            "summary": "Climate change is a pressing global challenge with severe environmental consequences.",
            "tags": ["Climate Change", "Global Crisis", "Rising Temperatures", "Extreme Weather", "Sea Level Rise"]
        })
    if "main_topic" in prompt and "Renewable energy" in prompt:
        return json.dumps({
            "main_topic": "Renewable Energy Solutions",
            "summary": "Renewable energy technologies like solar and wind are key solutions to the climate crisis.",
            "tags": ["Renewable Energy", "Solar & Wind", "Energy Storage", "Cost-Effective", "Smart Grid"]
        })

    # --- MOCK FOR STAGE 3: HIERARCHICAL SYNTHESIS (Unchanged) ---
    if "You are an expert Information Architect" in prompt:
        return json.dumps({
            "nodes": [
                {"id": "AI Systems & Ethics", "label": "AI Systems & Ethics", "group": "parent", "size": 25},
                {"id": "Environmental Challenges & Solutions", "label": "Environmental Challenges & Solutions", "group": "parent", "size": 25},
                {"id": "AI Industry Transformation", "label": "AI Industry Transformation", "group": "child", "size": 15},
                {"id": "Ethical AI & Data Privacy", "label": "Ethical AI & Data Privacy", "group": "child", "size": 15},
                {"id": "The Challenge of Climate Change", "label": "The Challenge of Climate Change", "group": "child", "size": 15},
                {"id": "Renewable Energy Solutions", "label": "Renewable Energy Solutions", "group": "child", "size": 15}
            ],
            "edges": [
                {"source": "AI Systems & Ethics", "target": "AI Industry Transformation"},
                {"source": "AI Systems & Ethics", "target": "Ethical AI & Data Privacy"},
                {"source": "Environmental Challenges & Solutions", "target": "The Challenge of Climate Change"},
                {"source": "Environmental Challenges & Solutions", "target": "Renewable Energy Solutions"}
            ]
        })

    return "{}"

# ----------------------------------------------------------------------------
# STAGE 1: SEMANTIC CHUNKING (Unchanged)
# ----------------------------------------------------------------------------
def get_semantic_chunks(text: str, model_path="all-mpnet-base-v2", threshold=80) -> List:
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
# STAGE 2: AUTOMATED TOPIC TAGGING (CHUNK-LEVEL) - Unchanged Prompt, Ensures Data is Kept
# ----------------------------------------------------------------------------
def generate_topic_for_chunk(chunk: str) -> Dict:
    prompt = f"""
    You are an expert data analyst. For the following text chunk, create a JSON object.
    The "main_topic" should be a short, clear title for the chunk, like a section heading.

    {{
      "main_topic": "A concise title for this chunk (3-5 words).",
      "summary": "A one-sentence summary of the chunk's main point.",
      "tags": ["A list of 4-5 specific keywords or phrases found in the chunk."]
    }}

    # Text Chunk to Analyze:
    {chunk}

    # Your JSON Output:
    """
    response_str = abc_response(prompt)
    try:
        data = json.loads(response_str)
        data['original_chunk'] = chunk  # Ensure we keep the original chunk
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for chunk: {chunk[:50]}...")
        return {}

# ----------------------------------------------------------------------------
# STAGE 3: HIERARCHICAL SYNTHESIS (Unchanged)
# ----------------------------------------------------------------------------
def generate_hierarchical_graph(tagged_data: List) -> Dict:
    main_topics = [item.get('main_topic', '') for item in tagged_data if item.get('main_topic')]
    topics_list_str = "\n- ".join(main_topics)

    prompt = f"""
    You are an expert Information Architect creating a visual table of contents.
    Based on the following list of topics, your job is to:
    1.  Invent 2-4 high-level parent categories that group these topics logically.
    2.  Generate a JSON object for a hierarchical tree with nodes and unlabeled edges.
    3.  Nodes should include your invented parent categories and the topics from the list.

    {{
      "nodes": [
        {{"id": "Parent Category A", "label": "Parent Category A", "group": "parent", "size": 25}},
        {{"id": "Main Topic 1", "label": "Main Topic 1", "group": "child", "size": 15}}
      ],
      "edges": [
        {{"source": "Parent Category A", "target": "Main Topic 1"}}
      ]
    }}

    # List of Main Topics to Organize:
    - {topics_list_str}

    # Your JSON Graph Output:
    """
    logging.info("🧠 Synthesizing hierarchical graph from all topics...")
    response_str = abc_response(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON for the hierarchical graph.")
        return {"nodes": [], "edges": []}

# ----------------------------------------------------------------------------
# STAGE 4: VISUALIZATION (ENHANCED HOVER DETAILS)
# ----------------------------------------------------------------------------
def create_flow_diagram(graph_data: Dict, filename: str = "flow_diagram.html"):
    if not graph_data.get("nodes"):
        logging.warning("No nodes found. Skipping visualization.")
        return

    net = Network(height="800px", width="100%", bgcolor="#f0f0f0", font_color="black", notebook=False, directed=True)

    net.set_options("""
    var options = {
      "physics": {
        "hierarchicalRepulsion": {
          "centralGravity": 0.0,
          "springLength": 200,
          "springConstant": 0.01,
          "nodeDistance": 200,
          "damping": 0.09
        },
        "minVelocity": 0.75,
        "solver": "hierarchicalRepulsion"
      }
    }
    """)

    nodes = graph_data.get('nodes', [])
    edges = graph_data.get('edges', [])

    for node_data in nodes:
        node_id = node_data['id']
        label = node_data['label']
        group = node_data.get('group')
        size = node_data.get("size", 15)
        color = "#FF6347" if group == "parent" else "#1E90FF"

        # Find the corresponding chunk data to create the hover title
        chunk_info = next((item for item in tagged_data if item.get('main_topic') == label or item.get('id') == label), None)
        hover_title = ""
        if chunk_info:
            hover_title += f"<b>Topic:</b> {chunk_info.get('main_topic', label)}<br>"
            hover_title += f"<b>Summary:</b> {chunk_info.get('summary', 'No summary available')}<br>"
            tags = chunk_info.get('tags', [])
            if tags:
                hover_title += "<b>Tags:</b><br><ul><li>" + "<li>".join(tags) + "</ul>"
            # Optional: Add a preview of the original chunk
            # original_chunk = chunk_info.get('original_chunk', '')
            # if original_chunk:
            #     hover_title += f"<br><b>Chunk Preview:</b><br>{original_chunk[:150]}..."

        net.add_node(node_id, label=label, title=hover_title, size=size, color=color)

    for edge in edges:
        net.add_edge(edge['source'], edge['target'])

    net.save_graph(filename)
    logging.info(f"📈 Flow diagram saved as '{filename}'. Open this file in your browser.")

# ----------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE (Unchanged)
# ----------------------------------------------------------------------------
def main_pipeline(text: str):
    chunks = get_semantic_chunks(text)

    logging.info("⚙️ Starting Stage 2: Generating main topics and tags for all chunks...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        global tagged_data  # Make tagged_data accessible in create_flow_diagram
        tagged_data = list(executor.map(generate_topic_for_chunk, chunks))

    print("\n" + "="*60)
    print("🏷️ MAIN TOPICS and TAGS GENERATED:")
    for i, data in enumerate(tagged_data):
        print(f"  Chunk {i+1} Topic: {data.get('main_topic')}, Tags: {data.get('tags')}")
    print("="*60 + "\n")

    graph_data = generate_hierarchical_graph(tagged_data)
    create_flow_diagram(graph_data)

if __name__ == "__main__":
    input_paragraph = """
    Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

    However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

    Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

    Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
    """
    main_pipeline(input_paragraph)
