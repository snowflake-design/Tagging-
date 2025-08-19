import streamlit as st
import json
import time
import logging
import re
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import PyPDF2
import io

# The specific Streamlit Flow components
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

# LangChain components
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache embeddings model for faster loading
@st.cache_resource
def get_embeddings_model():
    """Cache the embeddings model to avoid reloading."""
    return HuggingFaceEmbeddings(
        model_name="all-mpnet-base-v2", 
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )

# ----------------------------------------------------------------------------
# PDF PROCESSING
# ----------------------------------------------------------------------------
def extract_text_from_pdf(uploaded_file) -> tuple:
    """Extract text from uploaded PDF file using PyPDF2. Returns (text, page_count)."""
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        page_count = len(pdf_reader.pages)
        
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        
        # Clean and validate text
        text = clean_extracted_text(text)
        return text, page_count
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return "", 0

def clean_extracted_text(text: str) -> str:
    """Clean extracted text and ensure proper sentence structure."""
    if not text.strip():
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Fix broken sentences (common in PDF extraction)
    text = re.sub(r'([a-z])\s*\n\s*([A-Z])', r'\1. \2', text)
    
    # Ensure sentences end with proper punctuation
    text = re.sub(r'([a-zA-Z0-9])\s*\n\s*([A-Z])', r'\1. \2', text)
    
    # Remove lines that are too short (likely artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        # Keep lines that are substantial (more than just punctuation or very short)
        if len(line) > 10 and not re.match(r'^[^a-zA-Z]*$', line):
            # Ensure line ends with proper punctuation if it doesn't
            if line and not line[-1] in '.!?':
                line += '.'
            cleaned_lines.append(line)
    
    return ' '.join(cleaned_lines)

# ----------------------------------------------------------------------------
# MOCK LLM FUNCTION (Optimized)
# ----------------------------------------------------------------------------
def abc_response(prompt: str) -> str:
    """
    Mocks a blocking LLM API call. Reduced sleep time for faster processing.
    """
    logging.info("Simulating blocking LLM API call...")
    time.sleep(0.2)  # Reduced from 0.5 to 0.2 for faster processing

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
# STAGE 1: SEMANTIC CHUNKING (Optimized)
# ----------------------------------------------------------------------------
@st.cache_data
def get_semantic_chunks(text: str) -> List[str]:
    """Chunks text semantically using LangChain with cached embeddings."""
    embeddings = get_embeddings_model()  # Use cached embeddings
    text_splitter = SemanticChunker(
        embeddings=embeddings, 
        breakpoint_threshold_type="percentile", 
        breakpoint_threshold_amount=80
    )
    chunks = text_splitter.split_text(text)
    
    # Filter out very short chunks that might be artifacts
    valid_chunks = []
    for chunk in chunks:
        # Keep chunks that have substantial content
        if len(chunk.strip()) > 50 and len(chunk.split()) > 8:
            valid_chunks.append(chunk)
    
    return valid_chunks

def get_document_properties(text: str, page_count: int = None) -> dict:
    """Get document properties for display during processing."""
    word_count = len(text.split())
    char_count = len(text)
    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
    
    # Estimate chunks (rough calculation before actual chunking)
    estimated_chunks = max(1, word_count // 200)  # Rough estimate: ~200 words per chunk
    
    properties = {
        "word_count": word_count,
        "character_count": char_count,
        "paragraph_count": paragraph_count,
        "estimated_chunks": estimated_chunks
    }
    
    if page_count is not None:
        properties["page_count"] = page_count
    
    return properties

# ----------------------------------------------------------------------------
# STAGE 2: AUTOMATED TOPIC TAGGING (Optimized)
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
# STAGE 3: HIERARCHICAL SYNTHESIS (Cached)
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
# STAGE 4: VISUALIZATION (Optimized)
# ----------------------------------------------------------------------------
@st.cache_data
def create_flow_elements(graph_data: Dict) -> (List[StreamlitFlowNode], List[StreamlitFlowEdge]):
    """Converts graph data into lists of StreamlitFlowNode and StreamlitFlowEdge objects."""
    flow_nodes = []
    flow_edges = []
    color_map = {"root": "#8B0000", "parent": "#FF4500", "child": "#B22222", "grandchild": "#1E90FF"}
    
    for node in graph_data.get('nodes', []):
        group = node.get("group")
        flow_node = StreamlitFlowNode(
            id=node['id'], 
            pos=(0, 0), 
            data={'label': node['label']}, 
            node_type='default' if group != 'root' else 'input', 
            source_position='bottom', 
            target_position='top',
            style={
                'background': color_map.get(group, "#808080"), 
                'color': 'white', 
                'borderRadius': '8px', 
                'padding': '10px 15px', 
                'width': '220px', 
                'textAlign': 'center', 
                'fontSize': '14px', 
                'border': 'none'
            }
        )
        flow_nodes.append(flow_node)
        
    for edge in graph_data.get('edges', []):
        flow_edge = StreamlitFlowEdge(
            id=f"e-{edge['source']}-{edge['target']}", 
            source=edge['source'], 
            target=edge['target'], 
            animated=True, 
            style={'stroke': '#555555', 'strokeWidth': '2px'}
        )
        flow_edges.append(flow_edge)
        
    return flow_nodes, flow_edges

# ----------------------------------------------------------------------------
# MAIN STREAMLIT APP LOGIC
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Topic Flow Generator")

if 'flow_state' not in st.session_state:
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'tagged_data' not in st.session_state:
    st.session_state.tagged_data = None
if 'last_input_hash' not in st.session_state:
    st.session_state.last_input_hash = None
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None

def clear_session_state():
    """Clear session state for new processing."""
    st.session_state.graph_data = None
    st.session_state.tagged_data = None
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
    st.session_state.processing_error = None

def get_input_hash(text: str, input_method: str) -> str:
    """Generate hash for input to detect changes."""
    import hashlib
    return hashlib.md5(f"{input_method}:{text}".encode()).hexdigest()

st.title("📄 Automated Topic Flow Diagram Generator")

with st.sidebar:
    st.header("Input Source")
    
    # Input method selection
    input_method = st.radio("Choose input method:", ["Text Input", "PDF Upload"])
    
    if input_method == "Text Input":
        DEFAULT_TEXT = """Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
"""
        user_text = st.text_area("Paste your document text here:", value=DEFAULT_TEXT, height=300, key="user_text_area")
        
    else:  # PDF Upload
        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'], key="pdf_uploader")
        
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                user_text, page_count = extract_text_from_pdf(uploaded_file)
                
            if user_text:
                st.success(f"✅ PDF processed successfully! Extracted {len(user_text.split())} words from {page_count} pages.")
                with st.expander("Preview extracted text"):
                    st.text(user_text[:500] + "..." if len(user_text) > 500 else user_text)
            else:
                st.error("❌ Failed to extract text from PDF. Please check the file and try again.")
                user_text = ""
                page_count = 0
        else:
            user_text = ""
            page_count = 0

    # Generate button (only show if we have text)
    if (input_method == "Text Input" and user_text.strip()) or (input_method == "PDF Upload" and user_text):
        # Check if input has changed
        current_input_hash = get_input_hash(user_text, input_method)
        input_changed = st.session_state.last_input_hash != current_input_hash
        
        if input_changed and st.session_state.graph_data is not None:
            st.info("📝 Input has changed. Click 'Generate Diagram' to process the new content.")
        
        if st.button("Generate Diagram", type="primary", use_container_width=True):
            # Clear previous state for new processing
            clear_session_state()
            st.session_state.last_input_hash = current_input_hash
            
            try:
                with st.spinner("Analyzing document and preparing processing..."):
                    # Get document properties first
                    page_count_for_props = page_count if input_method == "PDF Upload" else None
                    doc_props = get_document_properties(user_text, page_count_for_props)
                    
                    # Display document information
                    col1, col2, col3, col4 = st.columns(4)
                    
                    if input_method == "PDF Upload":
                        col1.metric("📄 Pages", f"{doc_props['page_count']:,}")
                    
                    col2.metric("📝 Words", f"{doc_props['word_count']:,}")
                    col3.metric("📊 Paragraphs", f"{doc_props['paragraph_count']:,}")
                    col4.metric("🧩 Est. Chunks", f"{doc_props['estimated_chunks']:,}")
                    
                    st.info(f"🔍 **Processing Overview:** This document will be semantically chunked into meaningful sections. "
                           f"Based on the document size (~{doc_props['word_count']:,} words), this may take a moment to gather knowledge and analyze content structure.")
                
                with st.spinner("🧩 Creating semantic chunks... This may take a moment for large documents."):
                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    
                    # Stage 1: Chunking
                    progress_bar.progress(25)
                    chunks = get_semantic_chunks(user_text)
                    
                    # Update user with actual chunk count
                    st.info(f"✅ Document successfully chunked into **{len(chunks)} semantic sections**")
                
                with st.spinner("🏷️ Analyzing topics and generating tags for each section..."):
                    # Stage 2: Topic generation with parallel processing
                    progress_bar.progress(50)
                    chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
                    
                    # Increased max_workers for faster parallel processing
                    with ThreadPoolExecutor(max_workers=min(10, len(chunks_with_ids))) as executor:
                        tagged_data = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
                
                with st.spinner("🌳 Building hierarchical structure and relationships..."):
                    # Stage 3: Hierarchical synthesis
                    progress_bar.progress(75)
                    doc_title = "PDF Analysis" if input_method == "PDF Upload" else "Document Analysis"
                    graph_data = generate_hierarchical_graph(tagged_data, doc_title=doc_title)
                
                with st.spinner("🎨 Creating interactive visualization..."):
                    # Stage 4: Visualization
                    progress_bar.progress(90)
                    flow_nodes, flow_edges = create_flow_elements(graph_data)
                    
                    progress_bar.progress(100)
                    
                    st.session_state.graph_data = graph_data
                    st.session_state.tagged_data = tagged_data
                    st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
                    
                st.success("🎉 Diagram generated successfully! Click on any node in the diagram to explore details.")
                
            except Exception as e:
                st.session_state.processing_error = str(e)
                st.error(f"❌ **Processing Error:** {str(e)}")
                st.error("Please try again or contact support if the issue persists.")
                logging.error(f"Processing error: {str(e)}")
                clear_session_state()
                
    else:
        if st.session_state.graph_data is not None:
            st.warning("⚠️ Please provide new input to generate a fresh diagram.")
        else:
            st.info("📋 Please provide text input or upload a PDF file to generate the diagram.")

if st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    
    st.session_state.flow_state = streamlit_flow(
        'tree_layout', 
        st.session_state.flow_state, 
        layout=TreeLayout(direction='down'), 
        fit_view=True, 
        get_node_on_click=True, 
        height=600
    )

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
    if st.session_state.processing_error:
        st.error(f"⚠️ **Last Processing Error:** {st.session_state.processing_error}")
        st.info("Please try with different input or check the document format.")
    else:
        st.info("📝 Enter text in the sidebar and click 'Generate Diagram' to begin.")
