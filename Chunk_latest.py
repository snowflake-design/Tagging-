import streamlit as st
import json
import time
import logging
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
    if 'last_processed_file' not in st.session_state:
        st.session_state.last_processed_file = None
    if 'last_processed_text' not in st.session_state:
        st.session_state.last_processed_text = None
    if 'selected_node_details' not in st.session_state:
        st.session_state.selected_node_details = None

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

    # Check for empty chunk request
    if "empty" in prompt.lower() or len(prompt.strip()) < 50:
        return "{}"

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
    
    # Check if chunk is empty or too short
    if not chunk_text or len(chunk_text.strip()) < 20:
        logging.info(f"Empty or too short chunk detected: {chunk_id}")
        return {}
    
    prompt = f"""
    You are an expert data analyst. For the text chunk with ID '{chunk_id}', create a JSON object.
    The "main_topic" should be a short, clear title for the chunk, like a section heading.
    
    If the text chunk is empty or contains no meaningful content, return empty JSON: {{}}

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
        # If LLM returns empty JSON for empty content
        if not data:
            return {}
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
    # Filter out empty dictionaries
    valid_tagged_data = [item for item in _tagged_data if item and 'chunk_id' in item and 'main_topic' in item]
    
    if not valid_tagged_data:
        logging.warning("No valid tagged data available for hierarchy generation")
        return {"nodes": [], "edges": []}
    
    topics_with_ids_list = [f"- (id: {item.get('chunk_id')}) {item.get('main_topic')}" for item in valid_tagged_data]
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

# Function to get node details without causing rerun
def get_node_details(node_id: str, graph_data: Dict, tagged_data: List[Dict]) -> Dict:
    """Get details for a selected node without causing session state issues"""
    node_info = next((n for n in graph_data.get('nodes', []) if n['id'] == node_id), None)
    
    if not node_info:
        return {}
    
    details = {
        'id': node_id,
        'label': node_info['label'],
        'group': node_info.get('group'),
        'content': {}
    }
    
    group = node_info.get('group')
    
    if group == 'parent':
        child_ids = [edge['target'] for edge in graph_data['edges'] if edge['source'] == node_id]
        grandchild_ids = []
        for child_id in child_ids:
            grandchild_ids.extend([edge['target'] for edge in graph_data['edges'] if edge['source'] == child_id])
        
        all_summaries, all_tags, all_chunks = [], set(), []
        for gc_id in grandchild_ids:
            chunk_info = next((item for item in tagged_data if item.get('chunk_id') == gc_id), None)
            if chunk_info:
                all_summaries.append(chunk_info.get('summary', ''))
                all_chunks.append(chunk_info.get('original_chunk', ''))
                for tag in chunk_info.get('tags', []): 
                    all_tags.add(tag)
        
        details['content'] = {
            'type': 'parent',
            'aggregated_summary': ' '.join(all_summaries),
            'all_tags': sorted(list(all_tags)),
            'combined_text': "\n\n---\n\n".join(all_chunks)
        }

    elif group == 'child':
        original_chunk_ids = graph_data.get('consolidation_map', {}).get(node_id, [])
        all_summaries, all_tags = [], set()
        source_chunks = []
        
        for chunk_id in original_chunk_ids:
            chunk_info = next((item for item in tagged_data if item.get('chunk_id') == chunk_id), None)
            if chunk_info:
                all_summaries.append(chunk_info.get('summary', ''))
                for tag in chunk_info.get('tags', []): 
                    all_tags.add(tag)
                source_chunks.append({
                    'id': chunk_id,
                    'topic': chunk_info.get('main_topic', chunk_id),
                    'text': chunk_info.get('original_chunk', 'N/A')
                })
        
        details['content'] = {
            'type': 'child',
            'consolidated_summary': ' '.join(all_summaries),
            'combined_tags': sorted(list(all_tags)),
            'source_chunks': source_chunks
        }
    
    elif group == 'grandchild':
        chunk_info = next((item for item in tagged_data if item.get('chunk_id') == node_id), None)
        if chunk_info:
            details['content'] = {
                'type': 'grandchild',
                'summary': chunk_info.get('summary', 'N/A'),
                'tags': chunk_info.get('tags', []),
                'original_text': chunk_info.get('original_chunk', 'N/A')
            }
    
    else:  # Root node
        details['content'] = {
            'type': 'root',
            'description': "This is the root node representing the entire document."
        }
    
    return details

# Processing function
def process_document(text_to_process: str, doc_title: str, page_count: int = 0):
    """Process document and update session state"""
    try:
        with st.spinner(f"Processing document... {'This may take longer for larger PDFs.' if page_count > 0 else 'Please wait.'}"):
            progress_bar = st.progress(0)
            
            # Stage 1: Chunking
            progress_bar.progress(25)
            chunks = get_semantic_chunks(text_to_process)
            chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
            
            # Stage 2: Topic Generation
            progress_bar.progress(50)
            with ThreadPoolExecutor(max_workers=5) as executor:
                tagged_data_raw = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
            
            # Filter out empty results from topic generation
            tagged_data = [item for item in tagged_data_raw if item and 'chunk_id' in item]
            
            if not tagged_data:
                st.error("No valid content found in the document after processing. Please check your input.")
                return
            
            # Stage 3: Hierarchical Graph
            progress_bar.progress(75)
            graph_data = generate_hierarchical_graph(tagged_data, doc_title=doc_title)
            
            if not graph_data.get('nodes'):
                st.error("Could not generate hierarchy from the document content.")
                return
            
            # Stage 4: Flow Elements
            progress_bar.progress(90)
            flow_nodes, flow_edges = create_flow_elements(graph_data)
            
            # Update session state
            st.session_state.graph_data = graph_data
            st.session_state.tagged_data = tagged_data
            st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
            st.session_state.processing_complete = True
            st.session_state.selected_node_details = None
            
            progress_bar.progress(100)
            time.sleep(0.5)  # Brief pause to show completion
            
        st.success("Document processing completed successfully!")
        
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
        st.markdown('</div>', unsafe_allow_html=True)

    # Generate button
    if st.button("Generate Topic Flow Diagram", type="primary", use_container_width=True):
        text_to_process = ""
        doc_title = "Document Analysis"
        page_count = 0

        if input_method == "PDF Upload" and uploaded_file is not None:
            # Check if this is the same file as last processed
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            if st.session_state.last_processed_file != file_key:
                text_content, page_count = extract_text_from_pdf(uploaded_file)
                text_to_process = text_content
                doc_title = uploaded_file.name.replace('.pdf', '')
                st.session_state.last_processed_file = file_key
                st.session_state.processing_complete = False  # Reset processing state
        elif input_method == "Text Input" and user_text.strip():
            # Check if this is different text than last processed
            if st.session_state.last_processed_text != user_text:
                text_to_process = user_text
                st.session_state.last_processed_text = user_text
                st.session_state.processing_complete = False  # Reset processing state
        
        if text_to_process:
            process_document(text_to_process, doc_title, page_count)
        else:
            if input_method == "PDF Upload":
                st.warning("Please upload a PDF file to generate a diagram.")
            else:
                st.warning("Please enter some text to generate a diagram.")

# Main content area
if st.session_state.processing_complete and st.session_state.graph_data:
    st.header("Interactive Topic Hierarchy Diagram")
    
    # Display the flow diagram
    flow_result = streamlit_flow(
        'tree_layout', 
        st.session_state.flow_state, 
        layout=TreeLayout(direction='down'), 
        fit_view=True, 
        get_node_on_click=True, 
        height=600
    )

    # Update the flow state and handle node selection without rerun
    if flow_result:
        st.session_state.flow_state = flow_result
        
        # Check if a node was selected and update details
        if flow_result.selected_id and flow_result.selected_id != st.session_state.get('last_selected_id'):
            st.session_state.selected_node_details = get_node_details(
                flow_result.selected_id, 
                st.session_state.graph_data, 
                st.session_state.tagged_data
            )
            st.session_state.last_selected_id = flow_result.selected_id

    # Display node details in sidebar if a node is selected
    if st.session_state.selected_node_details:
        details = st.session_state.selected_node_details
        
        with st.sidebar:
            st.markdown("---")
            st.header("Node Details")
            st.subheader(details['label'])
            
            content = details.get('content', {})
            content_type = content.get('type')
            
            if content_type == 'parent':
                st.markdown(f"**Aggregated Summary:** {content.get('aggregated_summary', 'N/A')}")
                st.write("**All Contained Tags:**", content.get('all_tags', []))
                combined_text = content.get('combined_text', '')
                if combined_text:
                    with st.expander("Combined Text Preview (Read More...)"):
                        st.info(f"{combined_text[:500]}...")

            elif content_type == 'child':
                st.markdown(f"**Consolidated Summary:** {content.get('consolidated_summary', 'N/A')}")
                st.write("**Combined Tags:**", content.get('combined_tags', []))
                st.markdown("---")
                st.write("**Original Source Chunks:**")
                for chunk in content.get('source_chunks', []):
                    with st.expander(f"Source: {chunk['topic']}"):
                        st.info(chunk['text'])
            
            elif content_type == 'grandchild':
                st.markdown(f"**Summary:** {content.get('summary', 'N/A')}")
                st.write("**Tags:**", content.get('tags', []))
                st.info(f"**Original Text:**\n\n{content.get('original_text', 'N/A')}")
            
            else:  # Root node
                st.info(content.get('description', 'Root node'))

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
