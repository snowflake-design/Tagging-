import streamlit as st
import json
import time
import logging
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
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""


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
    
    # Check if chunk is empty or meaningless
    if not chunk_text or len(chunk_text.strip()) < 10:
        return {}
    
    prompt = f"""
    You are an expert data analyst. For the text chunk with ID '{chunk_id}', create a JSON object.
    The "main_topic" should be a short, clear title for the chunk, like a section heading.
    
    If the text chunk is empty, meaningless, or doesn't contain substantial content, just output {{}} and skip this chunk.

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
        if data:  # Only add chunk info if data is not empty
            data['chunk_id'] = chunk_id
            data['original_chunk'] = chunk_text
        return data
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON for chunk: {chunk_text[:50]}...")
        return {}

# ----------------------------------------------------------------------------
# STAGE 3: HIERARCHICAL SYNTHESIS (Prompt Corrected)
# ----------------------------------------------------------------------------
@st.cache_data
def generate_hierarchical_graph(_tagged_data: List[Dict], doc_title: str) -> Dict:
    """Analyzes all topics to create a unified, multi-level hierarchical graph."""
    # Filter out empty dictionaries
    valid_tagged_data = [item for item in _tagged_data if item and item.get('chunk_id') and item.get('main_topic')]
    
    if not valid_tagged_data:
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
# MAIN STREAMLIT APP LOGIC
# ----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Topic Flow Generator")

# Initialize session state with keys to prevent reloading
if 'flow_state' not in st.session_state:
    st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
if 'graph_data' not in st.session_state:
    st.session_state.graph_data = None
if 'tagged_data' not in st.session_state:
    st.session_state.tagged_data = None
if 'input_type' not in st.session_state:
    st.session_state.input_type = "Text"
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = ""

st.title("📄 Automated Topic Flow Diagram Generator")

with st.sidebar:
    st.header("Input Source")
    
    # Radio button for input type selection
    input_type = st.radio(
        "Choose input method:",
        ["Text", "PDF"],
        key="input_type_radio",
        index=0 if st.session_state.input_type == "Text" else 1
    )
    
    # Update session state if input type changed
    if input_type != st.session_state.input_type:
        st.session_state.input_type = input_type
        # Clear previous data when switching input types
        st.session_state.graph_data = None
        st.session_state.tagged_data = None
        st.session_state.flow_state = StreamlitFlowState(nodes=[], edges=[])
    
    user_text = ""
    
    if st.session_state.input_type == "Text":
        st.subheader("Text Input")
        DEFAULT_TEXT = """Artificial intelligence is rapidly transforming industries worldwide. Machine learning algorithms are becoming increasingly sophisticated and capable of handling complex tasks. Natural language processing has seen significant breakthroughs with the advent of transformer models. Computer vision applications are now widely deployed across various sectors.

However, ethical considerations around AI deployment are becoming increasingly important. Bias in algorithms can lead to unfair outcomes and perpetuate existing inequalities. Data privacy concerns are growing as AI systems require vast amounts of personal information. Regulatory frameworks are being developed to govern AI use responsibly.

Climate change represents one of the most pressing challenges of our time. Rising global temperatures are causing ice caps to melt and sea levels to rise. Extreme weather events are becoming more frequent and severe. International cooperation is essential to address this global crisis effectively.

Renewable energy technologies are emerging as crucial solutions. Solar panels and wind turbines are becoming more efficient and cost-effective. Energy storage technologies are solving intermittency challenges. Smart grid systems are enabling better integration of renewable sources.
"""
        user_text = st.text_area("Paste your document text here:", value=DEFAULT_TEXT, height=300, key="user_text_area")
    
    else:  # PDF input
        st.subheader("PDF Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                user_text = extract_text_from_pdf(uploaded_file)
                if user_text:
                    st.success(f"Extracted {len(user_text)} characters from PDF")
                    with st.expander("Preview extracted text"):
                        st.text(user_text[:500] + "..." if len(user_text) > 500 else user_text)
                else:
                    st.error("Failed to extract text from PDF")

    # Generate button
    if st.button("Generate Diagram", type="primary", use_container_width=True, disabled=not user_text):
        with st.spinner("Processing document..."):
            try:
                chunks = get_semantic_chunks(user_text)
                chunks_with_ids = [(f"chunk-{i}", chunk) for i, chunk in enumerate(chunks)]
                
                with ThreadPoolExecutor(max_workers=5) as executor:
                    tagged_data = list(executor.map(generate_topic_for_chunk, chunks_with_ids))
                
                # Filter out empty results
                tagged_data = [item for item in tagged_data if item]
                
                if not tagged_data:
                    st.error("No meaningful content found to process.")
                    st.stop()
                
                graph_data = generate_hierarchical_graph(tagged_data, doc_title="Document Analysis")
                
                if not graph_data.get('nodes'):
                    st.error("Failed to generate hierarchical structure.")
                    st.stop()
                
                flow_nodes, flow_edges = create_flow_elements(graph_data)
                
                st.session_state.graph_data = graph_data
                st.session_state.tagged_data = tagged_data
                st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges, selected_id=None)
                st.session_state.processed_text = user_text
                
                st.success("Diagram generated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

# Main content area
if st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    
    # Use a unique key to prevent unnecessary re-renders
    st.session_state.flow_state = streamlit_flow(
        'tree_layout', 
        st.session_state.flow_state, 
        layout=TreeLayout(direction='down'), 
        fit_view=True, 
        get_node_on_click=True, 
        height=600,
        key="main_flow_diagram"
    )

    # Handle node selection in sidebar
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
    st.info("Choose your input method in the sidebar and generate a diagram to begin.")
