import streamlit as st
import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import io
import PyPDF2

# The specific Streamlit Flow components
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

# Replaced LangChain with semantic-chunker and sentence-transformers
from semantic_chunker.chunker import SemanticChunker
from sentence_transformers import SentenceTransformer

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Session State Initialization ---
# This robust setup prevents the app from resetting on widget interactions.
def initialize_session_state():
    """Initializes all necessary keys in Streamlit's session state."""
    if 'flow_state' not in st.session_state:
        st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None
    if 'tagged_data' not in st.session_state:
        st.session_state.tagged_data = None
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""
    if 'input_method' not in st.session_state:
        st.session_state.input_method = "Text Input"
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION (As provided)
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
# STAGE 1: SEMANTIC CHUNKING (REPLACED with rango-ramesh/semantic-chunker)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer("all-mpnet-base-v2", device='cpu')

@st.cache_data
def get_semantic_chunks(text: str) -> List[str]:
    """Chunks text semantically using rango-ramesh/semantic-chunker."""
    model = load_embedding_model()
    # Initialize the chunker with the pre-loaded model instance
    chunker = SemanticChunker(
        model=model,
        max_tokens=256,       # Max tokens for a final merged chunk
        cluster_threshold=0.6 # Controls grouping granularity (higher = fewer, larger clusters)
    )
    # The library expects a list of dictionaries, but its internal logic
    # can gracefully handle a list of strings by splitting them first.
    # We will split by paragraph for better initial granularity.
    initial_split = [{"text": p.strip()} for p in text.split('\n\n') if p.strip()]
    merged_chunks = chunker.chunk(initial_split)
    # Return a list of the final text chunks
    return [chunk['text'] for chunk in merged_chunks]

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
# STAGE 3: HIERARCHICAL SYNTHESIS (As provided)
# ----------------------------------------------------------------------------
@st.cache_data
def generate_hierarchical_graph(_tagged_data: List[Dict], doc_title: str) -> Dict:
    """Analyzes all topics to create a unified, multi-level hierarchical graph."""
    topics_with_ids_list = [f"- (id: {item.get('chunk_id')}) {item.get('main_topic')}" for item in _tagged_data]
    topics_list_str = "\n".join(topics_with_ids_list)

    prompt = f"""
    You are an expert Information Architect building a detailed, multi-level outline of a document titled '{doc_title}'.

    Your task has three steps:
    Step 1: CONSOLIDATE. Look at the 'List of Original Topics with IDs' below. Group semantically similar topics and create a 'Consolidated Topic' label for each group.

    Step 2: CATEGORIZE. Organize the 'Consolidated Topics' under 2-4 high-level 'Parent Categories' that you invent.

    Step 3: STRUCTURE. Generate a single JSON object with 'nodes', 'edges', and a 'consolidation_map'.
    - For 'grandchild' nodes (the original topics), you MUST use the provided stable 'id' from the list as the node's 'id'.
    - The 'consolidation_map' must also use these stable 'id's.
    - Follow the format below precisely.

    ```json
    {{
      "nodes": [
        {{"id": "Root Node ID", "label": "Root Node Label", "group": "root"}},
        {{"id": "Parent Category", "label": "Parent Category", "group": "parent"}},
        {{"id": "Consolidated Topic", "label": "Consolidated Topic", "group": "child"}},
        {{"id": "Original Topic 1's ID", "label": "Original Topic 1's Label", "group": "grandchild"}}
      ],
      "edges": [
        {{"source": "Root Node ID", "target": "Parent Category"}},
        {{"source": "Parent Category", "target": "Consolidated Topic"}},
        {{"source": "Consolidated Topic", "target": "Original Topic 1's ID"}}
      ],
      "consolidation_map": {{
        "Consolidated Topic A": ["Original Topic 1's ID", "Original Topic 2's ID"],
        "Consolidated Topic B": ["Original Topic 3's ID"]
      }}
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
# STAGE 4: VISUALIZATION
# ----------------------------------------------------------------------------
def create_flow_elements(graph_data: Dict) -> (List[StreamlitFlowNode], List[StreamlitFlowEdge]):
    """Converts graph data into lists of StreamlitFlowNode and StreamlitFlowEdge objects."""
    flow_nodes = []
    flow_edges = []
    color_map = {"root": "#8B0000", "parent": "#FF4500", "child": "#B22222", "grandchild": "#1E90FF"}
    for node in graph_data.get('nodes', []):
        group = node.get("group")
        flow_node = StreamlitFlowNode(id=node['id'], pos=(0, 0), data={'label': node['label']}, node_type='default' if group != 'root' else 'input', source_position='bottom', target_position='top',
                                      style={'background': color_map.get(group, "#808080"), 'color': 'white', 'borderRadius': '8px', 'padding': '10px 15px', 'width': '220px', 'textAlign': 'center', 'fontSize': '14px', 'border': 'none'})
        flow_nodes.append(flow_node)
    for edge in graph_data.get('edges', []):
        flow_edge = StreamlitFlowEdge(id=f"e-{edge['source']}-{edge['target']}", source=edge['source'], target=edge['target'], animated=True, style={'stroke': '#555555', 'strokeWidth': '2px'})
        flow_edges.append(flow_edge)
    return flow_nodes, flow_edges

# ----------------------------------------------------------------------------
# UTILITY FUNCTION for PDF reading
# ----------------------------------------------------------------------------
def extract_text_from_pdf(pdf_file_obj):
    """Extracts text from an uploaded PDF file object."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

# ----------------------------------------------------------------------------
# MAIN STREAMLIT APP LOGIC
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Topic Flow Generator")

# Initialize state at the very beginning
initialize_session_state()

st.title("📄 Automated Topic Flow Diagram Generator")

# --- SIDEBAR FOR INPUT ---
with st.sidebar:
    st.header("Step 1: Provide Input")
    
    # Radio button to choose input method
    st.session_state.input_method = st.radio(
        "Choose your input method:",
        ("Text Input", "PDF Upload"),
        key="input_method_radio",
        horizontal=True
    )

    DEFAULT_TEXT = """Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
"""

    if st.session_state.input_method == "Text Input":
        user_text_area = st.text_area(
            "Paste your document text here:",
            value=DEFAULT_TEXT,
            height=300,
            key="user_text_area"
        )
        if user_text_area:
            st.session_state.user_text = user_text_area

    elif st.session_state.input_method == "PDF Upload":
        uploaded_file = st.file_uploader("Upload a PDF document", type="pdf", key="pdf_uploader")
        if uploaded_file is not None:
            # To prevent re-reading the file on every interaction, read it once and store in state
            if st.session_state.get('uploaded_file_name') != uploaded_file.name:
                with st.spinner("Extracting text from PDF..."):
                    st.session_state.user_text = extract_text_from_pdf(io.BytesIO(uploaded_file.getvalue()))
                    st.session_state.uploaded_file_name = uploaded_file.name # Track the file name
            st.text_area(
                "Extracted Text (Read-Only):",
                value=st.session_state.user_text,
                height=300,
                disabled=True
            )

    st.header("Step 2: Generate")
    if st.button("Generate Diagram", type="primary", use_container_width=True, key="generate_button"):
        if not st.session_state.user_text or not st.session_state.user_text.strip():
            st.warning("Please provide some text or upload a valid PDF before generating.")
        else:
            with st.spinner("Processing document... This may take a moment."):
                # Stage 1: Chunking
                chunks = get_semantic_chunks(st.session_state.user_text)
                chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
                
                # Stage 2: Tagging in Parallel
                with ThreadPoolExecutor(max_workers=5) as executor:
                    tagged_data = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
                
                # Stage 3: Hierarchical Synthesis
                graph_data = generate_hierarchical_graph(tagged_data, doc_title="Document Analysis")
                
                # Stage 4: Visualization Element Creation
                flow_nodes, flow_edges = create_flow_elements(graph_data)
                
                # Update session state with results
                st.session_state.graph_data = graph_data
                st.session_state.tagged_data = tagged_data
                st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
                st.session_state.processing_complete = True
            st.success("Diagram generated successfully!")

# --- MAIN PANEL FOR VISUALIZATION and DETAILS ---
if st.session_state.processing_complete and st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    
    # The streamlit_flow component for rendering the graph
    st.session_state.flow_state = streamlit_flow('tree_layout', 
                                                 st.session_state.flow_state, 
                                                 layout=TreeLayout(direction='down'), 
                                                 fit_view=True, 
                                                 get_node_on_click=True, 
                                                 height=600)

    # --- SIDEBAR FOR NODE DETAILS (only shows after a node is clicked) ---
    if st.session_state.flow_state and st.session_state.flow_state.selected_id:
        selected_node_id = st.session_state.flow_state.selected_id

        with st.sidebar:
            st.markdown("---")
            st.header("Node Details")
            node_info = next((n for n in st.session_state.graph_data.get('nodes', []) if n['id'] == selected_node_id), None)
            
            if node_info:
                st.subheader(node_info['label'])
                group = node_info.get('group')

                if group == 'parent':
                    child_ids = [edge['target'] for edge in st.session_state.graph_data['edges'] if edge['source'] == selected_node_id]
                    grandchild_ids = []
                    for child_id in child_ids:
                        grandchild_ids.extend([edge['target'] for edge in st.session_state.graph_data['edges'] if edge['source'] == child_id])
                    all_summaries, all_tags, all_chunks = [], set(), []
                    for gc_id in grandchild_ids:
                        chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == gc_id), None)
                        if chunk_info:
                            all_summaries.append(chunk_info.get('summary', ''))
                            all_chunks.append(chunk_info.get('original_chunk', ''))
                            for tag in chunk_info.get('tags', []): all_tags.add(tag)
                    st.markdown(f"**Aggregated Summary:** {' '.join(all_summaries)}")
                    st.write("**All Contained Tags:**", sorted(list(all_tags)))
                    combined_text = "\n\n---\n\n".join(all_chunks)
                    with st.expander("Combined Text Preview (Read More...)"):
                        st.info(f"{combined_text[:500]}...")

                elif group == 'child':
                    original_chunk_ids = st.session_state.graph_data.get('consolidation_map', {}).get(selected_node_id, [])
                    all_summaries, all_tags = [], set()
                    for chunk_id in original_chunk_ids:
                        chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == chunk_id), None)
                        if chunk_info:
                             all_summaries.append(chunk_info.get('summary', ''))
                             for tag in chunk_info.get('tags', []): all_tags.add(tag)
                    st.markdown(f"**Consolidated Summary:** {' '.join(all_summaries)}")
                    st.write("**Combined Tags:**", sorted(list(all_tags)))
                    st.markdown("---")
                    st.write("**Original Source Chunks:**")
                    for chunk_id in original_chunk_ids:
                        chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == chunk_id), None)
                        if chunk_info:
                            with st.expander(f"Source: {chunk_info.get('main_topic', chunk_id)}"):
                                st.info(chunk_info.get('original_chunk', 'N/A'))
                
                elif group == 'grandchild':
                    chunk_info = next((item for item in st.session_state.tagged_data if item.get('chunk_id') == selected_node_id), None)
                    if chunk_info:
                        st.markdown(f"**Summary:** {chunk_info.get('summary', 'N/A')}")
                        st.write("**Tags:**", chunk_info.get('tags', []))
                        st.info(f"**Original Text:**\n\n{chunk_info.get('original_chunk', 'N/A')}")
                
                else: # Root node
                    st.info("This is the root node representing the entire document.")
else:
    st.info("Enter text in the sidebar and click 'Generate Diagram' to begin.")
