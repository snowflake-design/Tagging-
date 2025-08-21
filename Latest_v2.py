import streamlit as st
import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
import io

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
# STAGE 0: CACHE HEAVY RESOURCES (like the embedding model)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """
    Loads and caches the HuggingFace embedding model.
    This function runs only once per session.
    """
    st.toast("Loading embedding model... This happens once per session.")
    return HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

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

    # --- MOCK FOR STAGE 3: HIERARCHICAL SYNTHESIS ---
    if "You are an expert Information Architect" in prompt:
        return json.dumps({
        "nodes": [
            {"id": "Document Analysis", "label": "Document Analysis", "group": "root"},
            {"id": "Artificial Intelligence", "label": "Artificial Intelligence", "group": "parent"},
            {"id": "Environmental Issues", "label": "Environmental Issues", "group": "parent"},
            {"id": "AI Development & Ethics", "label": "AI Development & Ethics", "group": "child"},
            {"id": "Climate Change & Response", "label": "Climate Change & Response", "group": "child"},
            {"id": "chunk-0", "label": "AI's Industrial Transformation", "group": "grandchild"},
            {"id": "chunk-1", "label": "AI Ethics and Challenges", "group": "grandchild"},
            {"id": "chunk-2", "label": "Climate Change Impacts", "group": "grandchild"},
            {"id": "chunk-3", "label": "Renewable Energy Solutions", "group": "grandchild"}
        ],
        "edges": [
            {"source": "Document Analysis", "target": "Artificial Intelligence"},
            {"source": "Document Analysis", "target": "Environmental Issues"},
            {"source": "Artificial Intelligence", "target": "AI Development & Ethics"},
            {"source": "Environmental Issues", "target": "Climate Change & Response"},
            {"source": "AI Development & Ethics", "target": "chunk-0"},
            {"source": "AI Development & Ethics", "target": "chunk-1"},
            {"source": "Climate Change & Response", "target": "chunk-2"},
            {"source": "Climate Change & Response", "target": "chunk-3"}
        ],
        "consolidation_map": {
            "AI Development & Ethics": ["chunk-0", "chunk-1"],
            "Climate Change & Response": ["chunk-2", "chunk-3"]
        }
        })
    return "{}"


# ----------------------------------------------------------------------------
# PDF EXTRACTION FUNCTION
# ----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> str:
    """Extracts text from uploaded PDF file using PyPDF2."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


# ----------------------------------------------------------------------------
# STAGE 1: SEMANTIC CHUNKING
# ----------------------------------------------------------------------------
def get_semantic_chunks(text: str, embeddings) -> List[str]:
    """Chunks text semantically using LangChain. No caching here to ensure re-computation."""
    text_splitter = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=80
    )
    return text_splitter.split_text(text)

# ----------------------------------------------------------------------------
# STAGE 2: AUTOMATED TOPIC TAGGING
# ----------------------------------------------------------------------------
def generate_topic_for_chunk(chunk_with_id: tuple) -> Dict:
    """Generates a main topic and tags for a single chunk. Skips empty/meaningless chunks."""
    chunk_id, chunk_text = chunk_with_id
    
    # Check if chunk is empty or meaningless
    if not chunk_text or len(chunk_text.strip()) < 20: # Increased threshold
        logging.warning(f"Skipping chunk {chunk_id} due to insufficient content.")
        return {}
    
    prompt = f"""
    For the text chunk with ID '{chunk_id}', create a JSON object with a 'main_topic', 'summary', and 'tags'.
    If the text chunk is meaningless, just output {{}}.

    # Text Chunk to Analyze:
    {chunk_text}
    """
    response_str = abc_response(prompt)
    try:
        data = json.loads(response_str)
        if data:  # Only add chunk info if data is not empty
            data['chunk_id'] = chunk_id
            data['original_chunk'] = chunk_text
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for chunk: {chunk_text[:50]}...")
        return {}

# ----------------------------------------------------------------------------
# STAGE 3: HIERARCHICAL SYNTHESIS
# ----------------------------------------------------------------------------
def generate_hierarchical_graph(tagged_data: List[Dict], doc_title: str) -> Dict:
    """Analyzes all topics to create a unified, multi-level hierarchical graph."""
    # This filter is crucial to ensure empty chunks from the previous step are ignored.
    valid_tagged_data = [item for item in tagged_data if item and item.get('chunk_id') and item.get('main_topic')]
    
    if not valid_tagged_data:
        return {"nodes": [], "edges": []}
    
    topics_with_ids_list = [f"- (id: {item['chunk_id']}) {item['main_topic']}" for item in valid_tagged_data]
    topics_list_str = "\n".join(topics_with_ids_list)
    
    prompt = f"""
    You are an expert Information Architect building a multi-level outline of a document titled '{doc_title}'.
    Based on the following 'List of Original Topics with IDs', generate a single JSON object with 'nodes', 'edges', and a 'consolidation_map'.
    You MUST use the provided stable 'id' (e.g., 'chunk-0') as the node's 'id' for grandchild nodes.

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
# STAGE 4: VISUALIZATION
# ----------------------------------------------------------------------------
def create_flow_elements(graph_data: Dict) -> (List[StreamlitFlowNode], List[StreamlitFlowEdge]):
    """Converts graph data into lists of StreamlitFlowNode and StreamlitFlowEdge objects."""
    flow_nodes, flow_edges = [], []
    color_map = {"root": "#8B0000", "parent": "#FF4500", "child": "#B22222", "grandchild": "#1E90FF"}
    for node in graph_data.get('nodes', []):
        group = node.get("group")
        style = {
            'background': color_map.get(group, "#808080"), 'color': 'white', 'borderRadius': '8px',
            'padding': '10px 15px', 'width': '220px', 'textAlign': 'center', 'fontSize': '14px', 'border': 'none'
        }
        flow_node = StreamlitFlowNode(
            id=node['id'], pos=(0, 0), data={'label': node['label']}, 
            node_type='default' if group != 'root' else 'input', 
            source_position='bottom', target_position='top', style=style
        )
        flow_nodes.append(flow_node)

    for edge in graph_data.get('edges', []):
        flow_edge = StreamlitFlowEdge(
            id=f"e-{edge['source']}-{edge['target']}", source=edge['source'], target=edge['target'], 
            animated=True, style={'stroke': '#555555', 'strokeWidth': '2px'}
        )
        flow_edges.append(flow_edge)
    return flow_nodes, flow_edges

# ----------------------------------------------------------------------------
# MAIN STREAMLIT APP LOGIC
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Topic Flow Generator")

# --- Load heavy models once and cache them ---
embeddings = load_embedding_model()

# --- Initialize session state ---
if 'flow_state' not in st.session_state:
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'tagged_data' not in st.session_state:
    st.session_state.tagged_data = None
if 'input_type' not in st.session_state:
    st.session_state.input_type = "Text"

st.title("📄 Automated Topic Flow Diagram Generator")

# --- SIDEBAR FOR INPUT AND CONTROLS ---
with st.sidebar:
    st.header("1. Input Source")
    
    input_type = st.radio(
        "Choose input method:", ["Text", "PDF"],
        key="input_type_radio",
        on_change=lambda: st.session_state.update(graph_data=None, tagged_data=None, flow_state=StreamlitFlowState(nodes=[], edges=[]))
    )
    st.session_state.input_type = input_type
    
    user_text = ""
    
    if st.session_state.input_type == "Text":
        DEFAULT_TEXT = """Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated. Natural language processing has seen breakthroughs with transformer models.

However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes. Data privacy concerns are growing as AI systems require vast amounts of personal information.

Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt. Extreme weather events are becoming more frequent and severe.

Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient. Energy storage technologies are solving intermittency challenges.
"""
        user_text = st.text_area("Paste your document text here:", value=DEFAULT_TEXT, height=300, key="user_text_area")
    
    else: # PDF input
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                user_text = extract_text_from_pdf(uploaded_file)
                if user_text:
                    st.success(f"Extracted {len(user_text.split())} words.")

    if st.button("Generate Diagram", type="primary", use_container_width=True, disabled=not user_text):
        with st.spinner("Processing document... This may take a moment."):
            try:
                # --- STAGE 1 ---
                chunks = get_semantic_chunks(user_text, embeddings)
                chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
                
                # --- STAGE 2 ---
                with ThreadPoolExecutor(max_workers=5) as executor:
                    tagged_data_results = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
                
                # Filter out any empty results from failed chunks
                tagged_data = [item for item in tagged_data_results if item]
                
                if not tagged_data:
                    st.error("No meaningful content could be extracted from the document. Please try a different text.")
                    st.stop()
                
                # --- STAGE 3 ---
                graph_data = generate_hierarchical_graph(tagged_data, doc_title="Document Analysis")
                
                if not graph_data.get('nodes'):
                    st.error("Failed to generate the hierarchical structure. The LLM might have returned an invalid format.")
                    st.stop()
                
                # --- STAGE 4 ---
                flow_nodes, flow_edges = create_flow_elements(graph_data)
                
                # --- Update Session State ---
                st.session_state.graph_data = graph_data
                st.session_state.tagged_data = tagged_data
                st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
                
                st.success("Diagram generated successfully!")
                st.rerun() # Rerun to apply state changes cleanly
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")

# --- MAIN CONTENT AREA FOR DIAGRAM AND DETAILS ---
if st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    st.caption("Click on any node to see its details in the sidebar.")
    
    st.session_state.flow_state = streamlit_flow(
        key='main_flow_diagram',
        state=st.session_state.flow_state,
        layout=TreeLayout(direction='down'),
        fit_view=True,
        get_node_on_click=True,
        height=600
    )

    # --- SIDEBAR FOR NODE DETAILS ---
    if st.session_state.flow_state and st.session_state.flow_state.selected_id:
        selected_node_id = st.session_state.flow_state.selected_id
        node_info = next((n for n in st.session_state.graph_data.get('nodes', []) if n['id'] == selected_node_id), None)
        
        with st.sidebar:
            st.markdown("---")
            st.header("2. Node Details")
            
            if node_info:
                st.subheader(f"`{node_info['label']}`")
                group = node_info.get('group')

                if group == 'parent':
                    st.info("This is a high-level category. Information below is aggregated from all its sub-topics.")
                    child_ids = [edge['target'] for edge in st.session_state.graph_data['edges'] if edge['source'] == selected_node_id]
                    grandchild_ids = [edge['target'] for child_id in child_ids for edge in st.session_state.graph_data['edges'] if edge['source'] == child_id]
                    
                    all_summaries, all_tags = [], set()
                    for gc_id in grandchild_ids:
                        chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == gc_id), None)
                        if chunk_info:
                            all_summaries.append(chunk_info.get('summary', ''))
                            for tag in chunk_info.get('tags', []): all_tags.add(tag)
                    
                    st.markdown(f"**Combined Summary:** {' '.join(all_summaries)}")
                    st.write("**All Tags in this Category:**", sorted(list(all_tags)))

                elif group == 'child':
                    st.info("This is a consolidated topic. Information below is aggregated from its source chunks.")
                    original_chunk_ids = st.session_state.graph_data.get('consolidation_map', {}).get(selected_node_id, [])
                    all_summaries, all_tags = [], set()
                    for chunk_id in original_chunk_ids:
                        chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == chunk_id), None)
                        if chunk_info:
                             all_summaries.append(chunk_info.get('summary', ''))
                             for tag in chunk_info.get('tags', []): all_tags.add(tag)
                    
                    st.markdown(f"**Consolidated Summary:** {' '.join(all_summaries)}")
                    st.write("**Combined Tags:**", sorted(list(all_tags)))
                
                elif group == 'grandchild':
                    st.info("This is an original chunk from the document.")
                    chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == selected_node_id), None)
                    if chunk_info:
                        st.markdown(f"**Summary:** {chunk_info.get('summary', 'N/A')}")
                        st.write("**Tags:**", chunk_info.get('tags', []))
                        with st.expander("Show Original Text"):
                            st.markdown(f"> {chunk_info.get('original_chunk', 'N/A')}")
                
                else: # Root node
                    st.info("This is the root node representing the entire document.")
else:
    st.info("⬅️ Choose your input method in the sidebar and click 'Generate Diagram' to begin.")
