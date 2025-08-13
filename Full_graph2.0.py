#!/usr/bin/env python3

import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# LangChain components for Stage 1
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Visualization library for Stage 4
from pyvis.network import Network

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    """
    Mocks a blocking LLM API call. It returns a pre-defined JSON string
    based on keywords in the prompt to simulate the hierarchical logic.
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
                {"id": "AI's Industrial Transformation", "label": "AI's Industrial Transformation", "group": "child", "size": 15},
                {"id": "AI Advancements and Challenges", "label": "AI Advancements and Challenges", "group": "child", "size": 15},
                {"id": "Climate Change Impacts", "label": "Climate Change Impacts", "group": "child", "size": 15},
                {"id": "Renewable Energy Solutions", "label": "Renewable Energy Solutions", "group": "child", "size": 15}
            ],
            "edges": [
                {"source": "Document Analysis", "target": "Artificial Intelligence"},
                {"source": "Document Analysis", "target": "Environmental Issues"},
                {"source": "Artificial Intelligence", "target": "AI's Industrial Transformation"},
                {"source": "Artificial Intelligence", "target": "AI Advancements and Challenges"},
                {"source": "Environmental Issues", "target": "Climate Change Impacts"},
                {"source": "Environmental Issues", "target": "Renewable Energy Solutions"}
            ]
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
    You are an expert Information Architect creating a visual table of contents for a document titled '{doc_title}'.
    Your job is to:
    1.  Create a single root node with the id '{doc_title}'.
    2.  Invent 2-4 high-level parent categories that group the topics from the list below.
    3.  Generate a JSON object for a hierarchical tree connecting the root node to the parents, and the parents to the topics.

    # List of Main Topics to Organize:
    - {topics_list_str}
    """
    logging.info("🧠 Synthesizing unified hierarchical graph from all topics...")
    response_str = abc_response(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON for the hierarchical graph.")
        return {"nodes": [], "edges": []}


# ----------------------------------------------------------------------------
# STAGE 4: VISUALIZATION (Reverted to Hover-for-Details)
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
        cdn_resources='in_line'  # Makes the HTML file self-contained
    )
    
    net.set_options("""
    var options = {
      "physics": {
        "solver": "hierarchicalRepulsion",
        "hierarchicalRepulsion": {
          "nodeDistance": 250,
          "springLength": 200
        }
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

        color_map = {"root": "#8B0000", "parent": "#FF4500", "child": "#1E90FF"}
        color = color_map.get(group, "#808080")

        # Prepare the hover tooltip content
        hover_title = ""
        chunk_info = next((item for item in tagged_data if item.get('main_topic') == label), None)
        if chunk_info:
            summary = chunk_info.get('summary', 'No summary.')
            tags_html = "<ul>" + "".join(f"<li>{tag}</li>" for tag in chunk_info.get('tags', [])) + "</ul>"
            
            # Format the content for the hover tooltip (title attribute)
            hover_title = f"<b>Summary:</b> {summary}<br><br><b>Key Tags:</b>{tags_html}"

        # Add the node with the 'title' attribute for the hover tooltip
        net.add_node(node_id, label=label, size=size, color=color, title=hover_title)

    for edge in edges:
        net.add_edge(edge['source'], edge['target'])

    net.write_html(filename, open_browser=False)
    logging.info(f"📦 Self-contained diagram with hover details saved as '{filename}'.")


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
