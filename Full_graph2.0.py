#!/usr/bin/env python3

import json
import time
import logging
import base64  # To safely pass data to JavaScript
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from pyvis.network import Network
from bs4 import BeautifulSoup # To inject custom HTML/JS/CSS

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION (Updated for Root Node Logic)
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    logging.info("Simulating blocking LLM API call...")
    time.sleep(0.5)

    # --- MOCK FOR STAGE 2: TOPIC TAGGING (Unchanged) ---
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

    # --- MOCK FOR STAGE 3: HIERARCHICAL SYNTHESIS (Updated) ---
    if "You are an expert Information Architect" in prompt:
        return json.dumps({
            "nodes": [
                # New "Ultra Parent" or Root Node
                {"id": "Document Analysis", "label": "Document Analysis", "group": "root", "size": 35},
                # Existing Parent Nodes
                {"id": "Artificial Intelligence", "label": "Artificial Intelligence", "group": "parent", "size": 25},
                {"id": "Environmental Issues", "label": "Environmental Issues", "group": "parent", "size": 25},
                # Child Nodes
                {"id": "AI's Industrial Transformation", "label": "AI's Industrial Transformation", "group": "child", "size": 15},
                {"id": "AI Advancements and Challenges", "label": "AI Advancements and Challenges", "group": "child", "size": 15},
                {"id": "Climate Change Impacts", "label": "Climate Change Impacts", "group": "child", "size": 15},
                {"id": "Renewable Energy Solutions", "label": "Renewable Energy Solutions", "group": "child", "size": 15}
            ],
            "edges": [
                # Edges from Root to Parents
                {"source": "Document Analysis", "target": "Artificial Intelligence"},
                {"source": "Document Analysis", "target": "Environmental Issues"},
                # Edges from Parents to Children
                {"source": "Artificial Intelligence", "target": "AI's Industrial Transformation"},
                {"source": "Artificial Intelligence", "target": "AI Advancements and Challenges"},
                {"source": "Environmental Issues", "target": "Climate Change Impacts"},
                {"source": "Environmental Issues", "target": "Renewable Energy Solutions"}
            ]
        })

    return "{}"


# ----------------------------------------------------------------------------
# STAGE 1 & 2 (Unchanged)
# ----------------------------------------------------------------------------
def get_semantic_chunks(text: str) -> List[str]:
    # (Code is identical to previous version, omitted for brevity)
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80)
    return text_splitter.split_text(text)

def generate_topic_for_chunk(chunk: str) -> Dict:
    # (Code is identical to previous version, omitted for brevity)
    prompt = f"""...""" # Using the same prompt as before
    response_str = abc_response(prompt)
    try:
        data = json.loads(response_str)
        data['original_chunk'] = chunk
        return data
    except json.JSONDecodeError: return {}

# ----------------------------------------------------------------------------
# STAGE 3: HIERARCHICAL SYNTHESIS (Updated Prompt)
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

    # Your JSON Graph Output:
    """
    logging.info("🧠 Synthesizing unified hierarchical graph from all topics...")
    # The actual call to abc_response() doesn't need the prompt text since it's mocked,
    # but in a real scenario, you would pass the full prompt.
    response_str = abc_response(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        return {"nodes": [], "edges": []}


# ----------------------------------------------------------------------------
# STAGE 4: VISUALIZATION (Updated with Clickable Popups)
# ----------------------------------------------------------------------------
def create_flow_diagram(graph_data: Dict, tagged_data: List[Dict], filename: str = "flow_diagram.html"):
    if not graph_data.get("nodes"): return

    net = Network(height="90vh", width="100%", bgcolor="#ffffff", font_color="black", notebook=False, directed=True)
    net.set_options("""
    var options = {
      "physics": { "solver": "hierarchicalRepulsion", "hierarchicalRepulsion": { "nodeDistance": 250 } }
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

        # Prepare data for the popup
        onclick_handler = ""
        chunk_info = next((item for item in tagged_data if item.get('main_topic') == label), None)
        if chunk_info:
            summary = chunk_info.get('summary', 'No summary.')
            tags_html = "<ul>" + "".join(f"<li>{tag}</li>" for tag in chunk_info.get('tags', [])) + "</ul>"
            chunk_text = chunk_info.get('original_chunk', 'No original text.').replace('\n', '<br>')

            # We encode the HTML content in Base64 to safely pass it into a JavaScript string
            popup_html_content = f"<h3>{label}</h3><h4>Summary</h4><p>{summary}</p><h4>Tags</h4>{tags_html}<h4>Original Text</h4><p>{chunk_text}</p>"
            encoded_content = base64.b64encode(popup_html_content.encode('utf-8')).decode('utf-8')
            onclick_handler = f"showPopup('{encoded_content}');"

        net.add_node(node_id, label=label, size=size, color=color, title="Click for details", **{"onclick": onclick_handler})

    for edge in edges:
        net.add_edge(edge['source'], edge['target'])

    # --- Inject Custom HTML/CSS/JS for the Popup Modal ---
    net.write_html(filename, open_browser=False)

    with open(filename, 'r+', encoding='utf-8') as f:
        html_content = f.read()
        soup = BeautifulSoup(html_content, 'html.parser')

        # CSS for the popup
        css = """
        <style>
            #popup-container { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-color: rgba(0,0,0,0.5); display: none; justify-content: center; align-items: center; z-index: 1000; }
            #popup-content { background-color: white; padding: 25px; border-radius: 8px; box-shadow: 0 5px 15px rgba(0,0,0,0.3); width: 60%; max-width: 700px; max-height: 80vh; overflow-y: auto; }
            #popup-close { float: right; font-size: 24px; font-weight: bold; cursor: pointer; }
        </style>
        """
        soup.head.append(BeautifulSoup(css, 'html.parser'))

        # HTML for the popup
        popup_html = """
        <div id="popup-container">
            <div id="popup-content">
                <span id="popup-close" onclick="closePopup()">&times;</span>
                <div id="popup-body"></div>
            </div>
        </div>
        """
        soup.body.append(BeautifulSoup(popup_html, 'html.parser'))

        # JavaScript for the popup
        js = """
        <script type="text/javascript">
            function showPopup(encodedContent) {
                var container = document.getElementById('popup-container');
                var body = document.getElementById('popup-body');
                // Decode the Base64 content and set it
                body.innerHTML = atob(encodedContent);
                container.style.display = 'flex';
            }
            function closePopup() {
                var container = document.getElementById('popup-container');
                container.style.display = 'none';
            }
            // Close popup if user clicks outside the content area
            window.onclick = function(event) {
                var container = document.getElementById('popup-container');
                if (event.target == container) {
                    container.style.display = "none";
                }
            }
        </script>
        """
        soup.body.append(BeautifulSoup(js, 'html.parser'))

        # Write the modified HTML back to the file
        f.seek(0)
        f.write(str(soup))
        f.truncate()

    logging.info(f"📈 Unified diagram with clickable popups saved as '{filename}'.")


# ----------------------------------------------------------------------------
# MAIN EXECUTION PIPELINE
# ----------------------------------------------------------------------------
def main_pipeline(text: str, doc_title: str):
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
    # We define the title for our "ultra parent" or root node here
    main_pipeline(input_paragraph, doc_title="Document Analysis")
