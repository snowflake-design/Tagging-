import streamlit as st
import json
import time
import logging
import warnings
from typing import List, Dict, Any
# Import ThreadPoolExecutor and Lock
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import PyPDF2
import io
from transformers import pipeline
import os

# Streamlit Flow components
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

# LangChain components
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
# Set this environment variable at the top to prevent a harmless but noisy warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------------
# STAGE 0: CACHE HEAVY RESOURCES (Models)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Loads and caches the HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource
def load_pii_model():
    """Loads and caches the PII detection model."""
    return pipeline("token-classification", "Jean-Baptiste/roberta-large-ner-english", aggregation_strategy="first", device=-1)

# ----------------------------------------------------------------------------
# PII MASKING MODULE
# ----------------------------------------------------------------------------
def mask_pii_robust(text: str, ner_pipeline, max_length=512):
    """
    Detects and masks PII in long texts.
    Handles texts longer than the model's token limit by using a sliding window.
    """
    all_entities = []
    stride = max_length // 2
    for i in range(0, len(text), stride):
        window = text[i:i + max_length]
        if not window: break
        window_entities = ner_pipeline(window)
        for entity in window_entities:
            entity['start'] += i
            entity['end'] += i
            all_entities.append(entity)
    unique_entities = []
    if all_entities:
        all_entities.sort(key=lambda x: x['start'])
        last_end = -1
        for entity in all_entities:
            if entity['start'] >= last_end:
                unique_entities.append(entity)
                last_end = entity['end']
    has_pii = len(unique_entities) > 0
    if not has_pii: return text, False
    masked_text = text
    for entity in sorted(unique_entities, key=lambda x: x['start'], reverse=True):
        masked_text = (masked_text[:entity['start']] + f"[{entity['entity_group']}]" + masked_text[entity['end']:])
    return masked_text, has_pii

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION (OR YOUR REAL LLM CALL)
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    # This mock is for placeholder purposes.
    logging.info("Simulating LLM API call...")
    time.sleep(0.5)
    if "chunk-0" in prompt: return json.dumps({"main_topic": "AI's Industrial Transformation", "summary": "...", "tags": []})
    if "chunk-1" in prompt: return json.dumps({"main_topic": "AI Ethics and Challenges", "summary": "...", "tags": []})
    if "chunk-2" in prompt: return json.dumps({"main_topic": "HVAC System Inspection Report", "summary": "...", "tags": []})
    if "You are an expert Information Architect" in prompt:
        return json.dumps({
        "nodes": [{"id": "Document_Analysis", "label": "Document Analysis", "group": "root"}, {"id": "Inspections_and_Reports", "label": "Inspections and Reports", "group": "parent"}, {"id": "Inspection_Reports", "label": "Inspection Reports", "group": "child"}, {"id": "chunk-0", "label": "AI's Industrial Transformation", "group": "grandchild"}, {"id": "chunk-1", "label": "AI Ethics and Challenges", "group": "grandchild"}, {"id": "chunk-2", "label": "HVAC System Inspection Report", "group": "grandchild"}],
        "edges": [{"source": "Document_Analysis", "target": "Inspections_and_Reports"}, {"source": "Inspections_and_Reports", "target": "Inspection_Reports"}, {"source": "Inspection_Reports", "target": "chunk-2"}],
        "consolidation_map": {"Inspection_Reports": ["chunk-2"]}
        })
    return "{}"

# ----------------------------------------------------------------------------
# PDF EXTRACTION, CHUNKING, HIERARCHY, VISUALIZATION FUNCTIONS
# ----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file) -> str:
    try:
        text = ""
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            if page_text := page.extract_text(): text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def get_semantic_chunks(text: str, embeddings) -> List[str]:
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80)
    return text_splitter.split_text(text)

def generate_topic_for_chunk_safe(args: tuple) -> Dict:
    chunk_id, chunk_text, pii_model, lock = args
    if not chunk_text or len(chunk_text.strip()) < 20: return {}
    with lock:
        masked_chunk, has_pii = mask_pii_robust(chunk_text, pii_model)
    prompt = f"""For the text chunk with ID '{chunk_id}', create a JSON object...
    # Text Chunk to Analyze: {masked_chunk}"""
    response_str = abc_response(prompt) # Replace with your real LLM call
    try:
        data = json.loads(response_str)
        if data:
            data.update({'chunk_id': chunk_id, 'original_chunk': chunk_text, 'masked_chunk': masked_chunk, 'has_pii': has_pii})
        return data
    except json.JSONDecodeError:
        return {}

def generate_hierarchical_graph(tagged_data: List[Dict], doc_title: str) -> Dict:
    valid_tagged_data = [item for item in tagged_data if item and item.get('chunk_id')]
    if not valid_tagged_data: return {"nodes": [], "edges": []}
    topics_list = [f"- (id: {item['chunk_id']}) {item['main_topic']}{' (Contains PII)' if item.get('has_pii') else ''}" for item in valid_tagged_data]
    topics_list_str = "\n".join(topics_list)
    prompt = f"""You are an expert Information Architect...
    # List of Original Topics with IDs:\n{topics_list_str}"""
    response_str = abc_response(prompt) # Replace with your real LLM call
    try: return json.loads(response_str)
    except json.JSONDecodeError: return {"nodes": [], "edges": []}


# --- FIX 1: UPDATED VISUALIZATION FUNCTION ---
def create_flow_elements(graph_data: Dict, tagged_data: List[Dict]) -> (List[StreamlitFlowNode], List[StreamlitFlowEdge]):
    """
    Converts graph data into flow elements.
    Programmatically highlights the correct child node if it contains PII.
    """
    flow_nodes, flow_edges = [], []
    color_map = {"root": "#8B0000", "parent": "#FF4500", "child": "#B22222", "grandchild": "#1E90FF"}
    pii_highlight_style = {'border': '3px solid #FF4E4E', 'boxShadow': '0 0 10px #FF4E4E'}

    # Get the ground truth from our PII scan
    pii_chunk_ids = {item['chunk_id'] for item in tagged_data if item.get('has_pii')}

    for node in graph_data.get('nodes', []):
        group = node.get("group")
        style = {'background': color_map.get(group, "#808080"), 'color': 'white', 'borderRadius': '8px', 'padding': '10px 15px', 'width': '220px', 'textAlign': 'center', 'fontSize': '14px', 'border': 'none'}
        
        # --- ROBUST HIGHLIGHTING LOGIC ---
        # We ignore any 'has_pii' from the LLM and calculate it ourselves
        should_highlight = False
        if group == 'child':
            # Find all grandchildren of this child node
            grandchild_ids = {edge['target'] for edge in graph_data.get('edges', []) if edge['source'] == node['id']}
            # Highlight if any of its grandchildren have PII
            if grandchild_ids.intersection(pii_chunk_ids):
                should_highlight = True
        
        if should_highlight:
            style.update(pii_highlight_style)
            
        flow_node = StreamlitFlowNode(id=node['id'], pos=(0, 0), data={'label': node['label']}, node_type='default' if group != 'root' else 'input', source_position='bottom', target_position='top', style=style)
        flow_nodes.append(flow_node)

    for edge in graph_data.get('edges', []):
        flow_edge = StreamlitFlowEdge(id=f"e-{edge['source']}-{edge['target']}", source=edge['source'], target=edge['target'], animated=True, style={'stroke': '#555555', 'strokeWidth': '2px'})
        flow_edges.append(flow_edge)
    return flow_nodes, flow_edges

# ----------------------------------------------------------------------------
# MAIN STREAMLIT APP LOGIC
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Topic Flow Generator")

# Load models and create the lock
embeddings = load_embedding_model()
pii_model = load_pii_model()
pii_lock = Lock()

# Initialize session state
for key, value in {'flow_state': StreamlitFlowState(nodes=[], edges=[]), 'graph_data': None, 'tagged_data': None, 'pii_found_in_document': False}.items():
    if key not in st.session_state: st.session_state[key] = value

st.title("📄 Automated Topic Flow Diagram Generator")

with st.sidebar:
    st.header("1. Input Source")
    input_type = st.radio("Choose input method:", ["Text", "PDF"], key="input_type_radio",
                          on_change=lambda: st.session_state.update(graph_data=None, tagged_data=None, flow_state=StreamlitFlowState(nodes=[], edges=[]), pii_found_in_document=False))
    user_text = ""
    if input_type == "Text":
        user_text = st.text_area("Paste your document text here:", value="The HVAC unit in Ulhasnagar requires inspection. Contact person: Mr. Rajesh Kumar, email rajesh.k@example.com.", height=300)
    else:
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        if uploaded_file:
            user_text = extract_text_from_pdf(uploaded_file)
            if user_text: st.success(f"Extracted {len(user_text.split())} words.")
    if st.session_state.pii_found_in_document:
        st.warning("⚠️ PII was detected and masked. Affected topics are highlighted.")
    if st.button("Generate Diagram", type="primary", use_container_width=True, disabled=not user_text):
        with st.spinner("Processing document (including PII scan)..."):
            chunks = get_semantic_chunks(user_text, embeddings)
            tasks = [(f"chunk-{i}", chunk, pii_model, pii_lock) for i, chunk in enumerate(chunks)]
            with ThreadPoolExecutor(max_workers=5) as executor:
                tagged_data_results = executor.map(generate_topic_for_chunk_safe, tasks)

            tagged_data = [item for item in tagged_data_results if item]
            st.session_state.pii_found_in_document = any(item.get('has_pii', False) for item in tagged_data)
            
            if not tagged_data: st.error("No meaningful content could be extracted."); st.stop()
            
            graph_data = generate_hierarchical_graph(tagged_data, doc_title="Document Analysis")
            if not graph_data.get('nodes'): st.error("Failed to generate the hierarchical structure."); st.stop()
            
            # Pass tagged_data to create_flow_elements for correct highlighting
            flow_nodes, flow_edges = create_flow_elements(graph_data, tagged_data)
            
            st.session_state.update(graph_data=graph_data, tagged_data=tagged_data, flow_state=StreamlitFlowState(nodes=flow_nodes, edges=flow_edges))
            st.success("Diagram generated successfully!")
            st.rerun()

if st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    st.caption("Click on any node to see its details in the sidebar.")
    st.session_state.flow_state = streamlit_flow(key='main_flow_diagram', state=st.session_state.flow_state, layout=TreeLayout(direction='down'), fit_view=True, get_node_on_click=True, height=600)
    if st.session_state.flow_state and st.session_state.flow_state.selected_id:
        selected_node_id = st.session_state.flow_state.selected_id
        node_info = next((n for n in st.session_state.graph_data.get('nodes', []) if n['id'] == selected_node_id), None)
        with st.sidebar:
            st.markdown("---")
            st.header("2. Node Details")
            if node_info:
                st.subheader(f"`{node_info['label']}`")
                group = node_info.get('group')
                if group in ('parent', 'child'):
                    grandchild_ids = []
                    if group == 'parent':
                        child_ids = [edge['target'] for edge in st.session_state.graph_data['edges'] if edge['source'] == selected_node_id]
                        grandchild_ids = [edge['target'] for child_id in child_ids for edge in st.session_state.graph_data['edges'] if edge['source'] == child_id]
                    else: grandchild_ids = [edge['target'] for edge in st.session_state.graph_data['edges'] if edge['source'] == selected_node_id]
                    summaries = [info.get('summary', '') for gc_id in grandchild_ids if (info := next((item for item in st.session_state.tagged_data if item.get('chunk_id') == gc_id), None))]
                    tags = {tag for gc_id in grandchild_ids if (info := next((item for item in st.session_state.tagged_data if item.get('chunk_id') == gc_id), None)) for tag in info.get('tags', [])}
                    st.markdown(f"**Consolidated Summary:** {' '.join(summaries)}")
                    st.write("**Combined Tags:**", sorted(list(tags)))
                
                # --- FIX 2: UPDATED SIDEBAR DISPLAY ---
                elif group == 'grandchild':
                    st.info("Details for this original document chunk:")
                    chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == selected_node_id), None)
                    if chunk_info:
                        if chunk_info.get('has_pii'):
                            st.warning("⚠️ PII was detected in this chunk.")
                        st.markdown(f"**Summary:** {chunk_info.get('summary', 'N/A')}")
                        st.write("**Tags:**", chunk_info.get('tags', []))
                        
                        # Show both original and masked text
                        with st.expander("Show Original (Unmasked) Text"):
                            st.markdown(f"> {chunk_info.get('original_chunk', 'N/A')}")
                        if chunk_info.get('has_pii'):
                            with st.expander("Show Masked Text Sent to LLM"):
                                st.markdown(f"> {chunk_info.get('masked_chunk', 'N/A')}")

else:
    st.info("⬅️ Choose your input method and click 'Generate Diagram' to begin.")
