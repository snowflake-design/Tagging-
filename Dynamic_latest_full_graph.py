import streamlit as st
import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# Streamlit Flow components
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

# LangChain components
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

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

    # STAGE 2 MOCK
    if "chunk-0" in prompt:
        return json.dumps({
            "main_topic": "AI's Industrial Transformation",
            "summary": "AI, ML, and NLP are rapidly advancing and transforming industries worldwide.",
            "tags": ["AI Transformation", "Machine Learning", "NLP", "Transformer Models", "Computer Vision"]
        })
    if "chunk-1" in prompt:
        return json.dumps({
            "main_topic": "AI Ethics and Challenges",
            "summary": "The deployment of AI raises significant ethical concerns like algorithmic bias and data privacy.",
            "tags": ["Ethical AI", "Algorithmic Bias", "Data Privacy", "Regulatory Frameworks", "Unfair Outcomes"]
        })
    if "chunk-2" in prompt:
        return json.dumps({
            "main_topic": "Climate Change Impacts",
            "summary": "Climate change is a pressing global challenge with severe environmental consequences.",
            "tags": ["Climate Change", "Global Crisis", "Rising Temperatures", "Extreme Weather", "Sea Level Rise"]
        })
    if "chunk-3" in prompt:
        return json.dumps({
            "main_topic": "Renewable Energy Solutions",
            "summary": "Renewable energy technologies like solar and wind are key solutions to the climate crisis.",
            "tags": ["Renewable Energy", "Solar & Wind", "Energy Storage", "Cost-Effective", "Smart Grid"]
        })

    # STAGE 3 MOCK (Generates a variable-depth hierarchy)
    if "You are an expert Information Architect" in prompt:
        return json.dumps({
          "nodes": [
            {"id": "AI & Environmental Topics", "label": "AI & Environmental Topics", "level": 0},
            {"id": "Artificial Intelligence", "label": "Artificial Intelligence", "level": 1},
            {"id": "Environmental Issues", "label": "Environmental Issues", "level": 1},
            {"id": "chunk-0", "label": "AI's Industrial Transformation", "level": 2},
            {"id": "chunk-1", "label": "AI Ethics and Challenges", "level": 2},
            {"id": "chunk-2", "label": "Climate Change Impacts", "level": 2},
            {"id": "chunk-3", "label": "Renewable Energy Solutions", "level": 2}
          ],
          "edges": [
            {"source": "AI & Environmental Topics", "target": "Artificial Intelligence"},
            {"source": "AI & Environmental Topics", "target": "Environmental Issues"},
            {"source": "Artificial Intelligence", "target": "chunk-0"},
            {"source": "Artificial Intelligence", "target": "chunk-1"},
            {"source": "Environmental Issues", "target": "chunk-2"},
            {"source": "Environmental Issues", "target": "chunk-3"}
          ]
        })
    return "{}"


# ----------------------------------------------------------------------------
# STAGE 1: SEMANTIC CHUNKING
# ----------------------------------------------------------------------------
@st.cache_data
def get_semantic_chunks(text: str) -> List[str]:
    """Chunks text semantically using LangChain."""
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80)
    return text_splitter.split_text(text)

# ----------------------------------------------------------------------------
# STAGE 2: AUTOMATED TOPIC TAGGING
# ----------------------------------------------------------------------------
def generate_topic_for_chunk(chunk_with_id: tuple) -> Dict:
    """Generates a main topic and tags for a single chunk with its ID."""
    chunk_id, chunk_text = chunk_with_id
    prompt = f"""
    You are an expert data analyst. For the text chunk with ID '{chunk_id}', create a JSON object.
    The "main_topic" should be a short, clear title for the chunk, like a section heading.

    ```json
    {{
      "main_topic": "A concise title for this chunk (3-5 words).",
      "summary": "A one-sentence summary of the chunk's main point.",
      "tags": ["A list of 4-5 specific keywords or phrases found in the chunk."]
    }}
    ```

    # Text Chunk to Analyze:
    {chunk_text}
    """
    response_str = abc_response(prompt)
    try:
        data = json.loads(response_str)
        data['chunk_id'] = chunk_id
        data['original_chunk'] = chunk_text
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for chunk: {chunk_text[:50]}...")
        return {}

# ----------------------------------------------------------------------------
# STAGE 3: HIERARCHICAL SYNTHESIS
# ----------------------------------------------------------------------------
@st.cache_data
def generate_hierarchical_graph(tagged_data: List[Dict]) -> Dict: # Corrected argument name
    """Analyzes all topics to create a fully dynamic hierarchical graph."""
    topics_with_ids_list = [f"- (id: {item.get('chunk_id')}) {item.get('main_topic')}" for item in tagged_data]
    topics_list_str = "\n".join(topics_with_ids_list)
    
    prompt = f"""
    You are an expert Information Architect. Your goal is to create the most logical hierarchical summary of the following topics.

    Analyze the topics, group them thematically, and create as many layers of abstraction as are necessary to create a clear and intuitive tree structure. A simple document might only need 2 levels, while a more complex one might need 4 or 5. Use your judgment.

    Generate a JSON object with 'nodes' and 'edges'.
    - Each node MUST have an 'id', 'label', and a 'level' (integer starting from 0 for the root).
    - For nodes representing original topics, you MUST use their provided 'id' (e.g., 'chunk-0'). For categories you invent, create a descriptive ID.
    - Follow the format below precisely.
    
    ```json
    {{
        "nodes": [
            {{"id": "Invented Root Label", "label": "Invented Root Label", "level": 0}},
            {{"id": "Invented Category Label", "label": "Invented Category Label", "level": 1}},
            {{"id": "chunk-0", "label": "Original Topic's Label", "level": 2}}
        ],
        "edges": [
            {{"source": "Invented Root Label", "target": "Invented Category Label"}},
            {{"source": "Invented Category Label", "target": "chunk-0"}}
        ]
    }}
    ```

    # List of Original Topics with IDs:
    {topics_list_str}

    # Your JSON Graph Output:
    """
    
    response_str = abc_response(prompt)
    try:
        return json.loads(response_str)
    except json.JSONDecodeError:
        logging.error("Failed to decode JSON for the hierarchical graph.")
        return {"nodes": [], "edges": []}


# ----------------------------------------------------------------------------
# STAGE 4: VISUALIZATION & HELPER FUNCTION
# ----------------------------------------------------------------------------
def create_flow_elements(graph_data: Dict) -> (List[StreamlitFlowNode], List[StreamlitFlowEdge]):
    """Converts graph data into lists of StreamlitFlowNode and StreamlitFlowEdge objects."""
    flow_nodes = []
    color_palette = ["#8B0000", "#FF4500", "#B22222", "#1E90FF", "#2E8B57"]
    
    for node in graph_data.get('nodes', []):
        level = node.get("level", 0)
        color = color_palette[level % len(color_palette)]
        
        flow_node = StreamlitFlowNode(id=node['id'], pos=(0, 0), data={'label': node['label']}, 
                                    node_type='input' if level == 0 else 'default',
                                    source_position='bottom', target_position='top',
                                    style={'background': color, 'color': 'white', 'borderRadius': '8px', 'padding': '10px 15px', 
                                           'width': '220px', 'textAlign': 'center', 'fontSize': '14px', 'border': 'none'})
        flow_nodes.append(flow_node)

    for edge in graph_data.get('edges', []):
        flow_edge = StreamlitFlowEdge(id=f"e-{edge['source']}-{edge['target']}", source=edge['source'], target=edge['target'], 
                                    animated=True, style={'stroke': '#555555', 'strokeWidth': '2px'})
        flow_edges.append(flow_edge)
    
    return flow_nodes, flow_edges

def find_descendant_chunks(start_node_id: str, edges: List[Dict]) -> List[str]:
    """Traverses the graph to find all leaf-node descendants (chunks) of a starting node."""
    descendants = []
    queue = deque([start_node_id])
    nodes_to_visit = {start_node_id}
    
    adj_list = {node: [] for node in {e['source'] for e in edges} | {e['target'] for e in edges}}
    for edge in edges:
        adj_list[edge['source']].append(edge['target'])

    while queue:
        current_node = queue.popleft()
        
        if current_node.startswith('chunk-'):
            descendants.append(current_node)
        
        for neighbor in adj_list.get(current_node, []):
            if neighbor not in nodes_to_visit:
                nodes_to_visit.add(neighbor)
                queue.append(neighbor)
                
    return descendants

# ----------------------------------------------------------------------------
# MAIN STREAMLIT APP LOGIC
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Topic Flow Generator")

# Initialize session state
if 'flow_state' not in st.session_state:
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'tagged_data' not in st.session_state:
    st.session_state.tagged_data = None

st.title("📄 Automated Topic Flow Diagram Generator")

with st.sidebar:
    st.header("Input Text")
    DEFAULT_TEXT = """Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
"""
    user_text = st.text_area("Paste your document text here:", value=DEFAULT_TEXT, height=300, key="user_text_area")

    if st.button("Generate Diagram", type="primary", use_container_width=True):
        with st.spinner("Processing document..."):
            chunks = get_semantic_chunks(user_text)
            chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
            with ThreadPoolExecutor(max_workers=5) as executor:
                tagged_data = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
            graph_data = generate_hierarchical_graph(tagged_data)
            flow_nodes, flow_edges = create_flow_elements(graph_data)
            
            st.session_state.graph_data = graph_data
            st.session_state.tagged_data = tagged_data
            st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
        st.success("Diagram generated!")

if st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    
    st.session_state.flow_state = streamlit_flow('tree_layout', st.session_state.flow_state, 
                                                  layout=TreeLayout(direction='down'), fit_view=True, 
                                                  get_node_on_click=True, height=600)

    if st.session_state.flow_state and st.session_state.flow_state.selected_id:
        selected_node_id = st.session_state.flow_state.selected_id

        with st.sidebar:
            st.markdown("---")
            st.header("Node Details")
            node_info = next((n for n in st.session_state.graph_data.get('nodes', []) if n['id'] == selected_node_id), None)
            
            if node_info:
                st.subheader(node_info['label'])
                
                if selected_node_id.startswith('chunk-'):
                    chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == selected_node_id), None)
                    if chunk_info:
                        st.write("**Tags:**", chunk_info.get('tags', []))
                        st.info(f"**Original Text:**\n\n{chunk_info.get('original_chunk', 'N/A')}")
                else: # It's a root or internal category node
                    descendant_chunk_ids = find_descendant_chunks(selected_node_id, st.session_state.graph_data['edges'])
                    if not descendant_chunk_ids:
                        st.info("This is a category node with no further text chunks.")
                    else:
                        st.write(f"This category contains {len(descendant_chunk_ids)} source text chunk(s):")
                        all_summaries, all_tags, all_chunks = [], set(), []
                        for chunk_id in descendant_chunk_ids:
                            chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == chunk_id), None)
                            if chunk_info:
                                all_summaries.append(chunk_info.get('summary', ''))
                                all_chunks.append(chunk_info.get('original_chunk', ''))
                                for tag in chunk_info.get('tags', []): all_tags.add(tag)
                        
                        st.markdown(f"**Aggregated Summary:** {' '.join(all_summaries)}")
                        st.write("**All Contained Tags:**", sorted(list(all_tags)))
                        combined_text = "\n\n---\n\n".join(all_chunks)
                        with st.expander("Combined Text Preview (Read More...)"):
                            st.info(f"{combined_text[:700]}...")
else:
    st.info("Enter text in the sidebar and click 'Generate Diagram' to begin.")
