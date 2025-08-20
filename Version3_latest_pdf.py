import streamlit as st
import json
import time
import logging
import hashlib
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# PDF Processing Library
import io
from PyPDF2 import PdfReader

# The specific Streamlit Flow components
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.layouts import TreeLayout
from streamlit_flow.state import StreamlitFlowState

# LangChain components
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize session state
def init_session_state():
    """Initialize all session state variables"""
    if 'flow_state' not in st.session_state:
        st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
    if 'graph_data' not in st.session_state:
        st.session_state.graph_data = None
    if 'tagged_data' not in st.session_state:
        st.session_state.tagged_data = None
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
    if 'content_hash' not in st.session_state:
        st.session_state.content_hash = None
    if 'cached_results' not in st.session_state:
        st.session_state.cached_results = {}
    if 'current_input_method' not in st.session_state:
        st.session_state.current_input_method = None
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'current_text_content' not in st.session_state:
        st.session_state.current_text_content = None

def generate_content_hash(text: str, source_type: str, source_name: str = "") -> str:
    """Generate SHA-256 hash of content with source context for proper caching"""
    content_with_context = f"{source_type}:{source_name}:{text}"
    return hashlib.sha256(content_with_context.encode('utf-8')).hexdigest()

def is_content_cached(content_hash: str) -> bool:
    """Check if content results are already cached"""
    return content_hash in st.session_state.cached_results

def cache_results(content_hash: str, graph_data: Dict, tagged_data: List[Dict], flow_state: StreamlitFlowState, source_info: Dict):
    """Cache processing results for future use with source information"""
    st.session_state.cached_results[content_hash] = {
        'graph_data': graph_data,
        'tagged_data': tagged_data,
        'flow_state': flow_state,
        'source_info': source_info,
        'timestamp': time.time()
    }

def load_cached_results(content_hash: str):
    """Load cached results and update session state"""
    if content_hash in st.session_state.cached_results:
        cached = st.session_state.cached_results[content_hash]
        st.session_state.graph_data = cached['graph_data']
        st.session_state.tagged_data = cached['tagged_data']
        st.session_state.flow_state = cached['flow_state']
        st.session_state.content_hash = content_hash
        st.session_state.processing_complete = True
        return True
    return False

def clear_current_results():
    """Clear current processing results when switching input methods or content"""
    st.session_state.graph_data = None
    st.session_state.tagged_data = None
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
    st.session_state.processing_complete = False
    st.session_state.content_hash = None

def check_content_change(input_method: str, uploaded_file=None, user_text: str = ""):
    """Check if content has changed and clear results if needed"""
    content_changed = False
    
    if input_method == "PDF Upload":
        current_pdf = uploaded_file.name if uploaded_file else None
        if st.session_state.current_pdf_name != current_pdf:
            content_changed = True
            st.session_state.current_pdf_name = current_pdf
    
    elif input_method == "Text Input":
        if st.session_state.current_text_content != user_text:
            content_changed = True
            st.session_state.current_text_content = user_text
    
    if content_changed:
        clear_current_results()

# PDF TEXT EXTRACTION FUNCTION
def extract_text_from_pdf(file_buffer) -> tuple:
    """Opens and reads the text from an in-memory PDF file buffer provided by Streamlit."""
    logging.info("Reading text from uploaded PDF buffer...")
    full_text = ""
    page_count = 0
    try:
        reader = PdfReader(file_buffer)
        page_count = len(reader.pages)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"
        logging.info("Text extraction from PDF complete.")
        return full_text, page_count
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        logging.error(f"Could not read PDF file: {e}")
        return "", 0

# MOCK LLM FUNCTION
def abc_response(prompt: str) -> str:
    """
    Mocks a blocking LLM API call. It returns a pre-defined JSON string
    based on keywords in the prompt to simulate the final hierarchical logic.
    """
    logging.info("Simulating blocking LLM API call...")
    time.sleep(0.5)

    # MOCK FOR STAGE 2: TOPIC TAGGING
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

    # MOCK FOR STAGE 3: HIERARCHICAL SYNTHESIS
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


# STAGE 1: SEMANTIC CHUNKING
@st.cache_data
def get_semantic_chunks(text: str) -> List[str]:
    """Chunks text semantically using LangChain."""
    embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    text_splitter = SemanticChunker(embeddings=embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=80)
    return text_splitter.split_text(text)

# STAGE 2: AUTOMATED TOPIC TAGGING
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

# STAGE 3: HIERARCHICAL SYNTHESIS
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


# STAGE 4: VISUALIZATION
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

# Processing function
def process_document(text_to_process: str, doc_title: str, source_type: str, source_name: str = "", page_count: int = 0):
    """Process document and update session state with proper caching"""
    try:
        # Generate content hash with source context
        content_hash = generate_content_hash(text_to_process, source_type, source_name)
        
        # Check if results are already cached
        if is_content_cached(content_hash):
            st.info("Content already processed! Loading cached results...")
            load_cached_results(content_hash)
            st.success("Cached results loaded successfully!")
            st.rerun()
            return
        
        # Process new content
        with st.spinner(f"Processing document... {'This may take longer for larger PDFs.' if page_count > 0 else 'Please wait.'}"):
            progress_bar = st.progress(0)
            
            # Stage 1: Chunking
            progress_bar.progress(25)
            chunks = get_semantic_chunks(text_to_process)
            chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
            
            # Stage 2: Topic Generation
            progress_bar.progress(50)
            with ThreadPoolExecutor(max_workers=5) as executor:
                tagged_data = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
            
            # Stage 3: Hierarchical Graph
            progress_bar.progress(75)
            graph_data = generate_hierarchical_graph(tagged_data, doc_title=doc_title)
            
            # Stage 4: Flow Elements
            progress_bar.progress(90)
            flow_nodes, flow_edges = create_flow_elements(graph_data)
            flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
            
            # Update session state
            st.session_state.graph_data = graph_data
            st.session_state.tagged_data = tagged_data
            st.session_state.flow_state = flow_state
            st.session_state.content_hash = content_hash
            st.session_state.processing_complete = True
            
            # Cache the results for future use with source info
            source_info = {
                'type': source_type,
                'name': source_name,
                'title': doc_title
            }
            cache_results(content_hash, graph_data, tagged_data, flow_state, source_info)
            
            progress_bar.progress(100)
            time.sleep(0.5)  # Brief pause to show completion
            
        st.success("Document processing completed successfully!")
        st.rerun()
        
    except Exception as e:
        st.error(f"Error processing document: {str(e)}")
        logging.error(f"Processing error: {e}")

# MAIN STREAMLIT APP LOGIC
st.set_page_config(layout="wide", page_title="Advanced Topic Flow Generator")

# Initialize session state
init_session_state()

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-bottom: 2rem;
    }
    
    .sidebar-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    
    .info-card {
        background: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    
    .generate-btn {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Advanced Topic Flow Diagram Generator</h1>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.header("Input Source Selection")
    
    # Radio button for input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "PDF Upload"],
        help="Select whether you want to paste text or upload a PDF document"
    )
    
    # Check if input method changed and clear results if so
    if st.session_state.current_input_method != input_method:
        clear_current_results()
        st.session_state.current_input_method = input_method
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if input_method == "PDF Upload":
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("PDF Document Upload")
        uploaded_file = st.file_uploader(
            "Choose a PDF file", 
            type="pdf",
            help="Upload your PDF document for analysis"
        )
        
        # Show PDF info if file is uploaded
        if uploaded_file is not None:
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.1f} KB"
            }
            st.json(file_details)
            
            # Try to get page count for estimation
            try:
                text_content, page_count = extract_text_from_pdf(uploaded_file)
                if page_count > 0:
                    st.info(f"Document contains {page_count} pages")
                    estimated_time = max(10, page_count * 2)  # Estimate 2 seconds per page, minimum 10 seconds
                    st.warning(f"Estimated processing time: {estimated_time} seconds")
            except:
                st.warning("Could not determine page count")
        
        # Check for PDF change
        check_content_change("PDF Upload", uploaded_file)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:  # Text Input
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("Text Input")
        DEFAULT_TEXT = """Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
"""
        user_text = st.text_area(
            "Paste your document text here:", 
            value=DEFAULT_TEXT, 
            height=300, 
            key="user_text_area",
            help="Paste the text you want to analyze"
        )
        
        # Check for text change
        check_content_change("Text Input", user_text=user_text)
        st.markdown('</div>', unsafe_allow_html=True)

    # Generate button
    if st.button("Generate Topic Flow Diagram", type="primary", use_container_width=True):
        text_to_process = ""
        doc_title = "Document Analysis"
        source_type = ""
        source_name = ""
        page_count = 0

        if input_method == "PDF Upload" and uploaded_file is not None:
            text_content, page_count = extract_text_from_pdf(uploaded_file)
            text_to_process = text_content
            doc_title = uploaded_file.name.replace('.pdf', '')
            source_type = "PDF"
            source_name = uploaded_file.name
        elif input_method == "Text Input" and user_text.strip():
            text_to_process = user_text
            source_type = "TEXT"
            source_name = "manual_input"
        
        if text_to_process:
            process_document(text_to_process, doc_title, source_type, source_name, page_count)
        else:
            if input_method == "PDF Upload":
                st.warning("Please upload a PDF file to generate a diagram.")
            else:
                st.warning("Please enter some text to generate a diagram.")

    # Cache management section
    st.markdown("---")
    if st.session_state.cached_results:
        st.subheader("Cache Management")
        cache_count = len(st.session_state.cached_results)
        st.info(f"Cached results: {cache_count} document(s)")
        
        # Show cache details
        with st.expander("View Cached Documents"):
            for hash_key, cached_item in st.session_state.cached_results.items():
                source_info = cached_item.get('source_info', {})
                st.text(f"• {source_info.get('type', 'Unknown')}: {source_info.get('name', 'Unknown')} ({source_info.get('title', 'No title')})")
        
        if st.button("Clear Cache", help="Clear all cached processing results"):
            st.session_state.cached_results = {}
            st.success("Cache cleared successfully!")
            st.rerun()

# Main content area
if st.session_state.processing_complete and st.session_state.graph_data:
    st.header("Interactive Topic Hierarchy Diagram")
    
    # Display the flow diagram
    st.session_state.flow_state = streamlit_flow(
        'tree_layout', 
        st.session_state.flow_state, 
        layout=TreeLayout(direction='down'), 
        fit_view=True, 
        get_node_on_click=True, 
        height=600
    )

    # Node details in sidebar
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
                
                else:  # Root node
                    st.info("This is the root node representing the entire document.")

else:
    st.markdown('<div class="info-card">', unsafe_allow_html=True)
    st.info("Welcome to the Advanced Topic Flow Generator! Choose your input method from the sidebar, then click 'Generate Topic Flow Diagram' to create an interactive visualization of your document's structure.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show some features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Features")
        st.markdown("- Semantic text chunking")
        st.markdown("- Automated topic extraction")
        st.markdown("- Hierarchical organization")
        st.markdown("- Interactive visualization")
    
    with col2:
        st.markdown("### Supported Formats")
        st.markdown("- PDF documents")
        st.markdown("- Plain text input")
        st.markdown("- Multi-page documents")
        st.markdown("- Various content types")
    
    with col3:
        st.markdown("### How It Works")
        st.markdown("1. Upload PDF or paste text")
        st.markdown("2. System analyzes content")
        st.markdown("3. Topics are extracted")
        st.markdown("4. Hierarchy is generated")
