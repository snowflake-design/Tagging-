import streamlit as st
import json
import time
import logging
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor

# The specific Streamlit Flow components we will use
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
            {"id": "Document Analysis", "label": "Document Analysis", "group": "root"},
            {"id": "Artificial Intelligence", "label": "Artificial Intelligence", "group": "parent"},
            {"id": "Environmental Issues", "label": "Environmental Issues", "group": "parent"},
            {"id": "AI Development & Impact", "label": "AI Development & Impact", "group": "child"},
            {"id": "Climate Response", "label": "Climate Response", "group": "child"},
            {"id": "AI's Industrial Transformation", "label": "AI's Industrial Transformation", "group": "grandchild"},
            {"id": "AI Advancements and Challenges", "label": "AI Advancements and Challenges", "group": "grandchild"},
            {"id": "Climate Change Impacts", "label": "Climate Change Impacts", "group": "grandchild"},
            {"id": "Renewable Energy Solutions", "label": "Renewable Energy Solutions", "group": "grandchild"}
          ],
          "edges": [
            {"source": "Document Analysis", "target": "Artificial Intelligence"},
            {"source": "Document Analysis", "target": "Environmental Issues"},
            {"source": "Artificial Intelligence", "target": "AI Development & Impact"},
            {"source": "Environmental Issues", "target": "Climate Response"},
            {"source": "AI Development & Impact", "target": "AI's Industrial Transformation"},
            {"source": "AI Development & Impact", "target": "AI Advancements and Challenges"},
            {"source": "Climate Response", "target": "Climate Change Impacts"},
            {"source": "Climate Response", "target": "Renewable Energy Solutions"}
          ],
          "consolidation_map": {
            "AI Development & Impact": ["AI's Industrial Transformation", "AI Advancements and Challenges"],
            "Climate Response": ["Climate Change Impacts", "Renewable Energy Solutions"]
          }
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
    logging.info(f"Chunking complete. Found {len(text_splitter.split_text(text))} chunks.")
    return text_splitter.split_text(text)

# ----------------------------------------------------------------------------
# STAGE 2: AUTOMATED TOPIC TAGGING
# ----------------------------------------------------------------------------
def generate_topic_for_chunk(chunk: str) -> Dict:
    """Generates a main topic and tags for a single chunk."""
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
@st.cache_data
def generate_hierarchical_graph(_tagged_data: List[Dict], doc_title: str) -> Dict:
    """Analyzes all topics to create a unified, multi-level hierarchical graph."""
    main_topics = [item.get('main_topic', '') for item in _tagged_data if item.get('main_topic')]
    topics_list_str = "\n- ".join(main_topics)
    
    prompt = f"""
    You are an expert Information Architect building a detailed, multi-level outline of a document titled '{doc_title}'.
    Your task has three steps:

    Step 1: CONSOLIDATE. Look at the 'List of Original Topics' below. Group semantically similar topics together and create a single, representative 'Consolidated Topic' label for each group.

    Step 2: CATEGORIZE. Take your new list of 'Consolidated Topics' and organize them under 2-4 high-level 'Parent Categories' that you invent.

    Step 3: STRUCTURE. Generate a single JSON object containing three keys: 'nodes', 'edges', and a 'consolidation_map'. The map must link each new 'Consolidated Topic' to the list of 'Original Topics' it represents.

    Follow the format below precisely.

    ```json
    {{
      "nodes": [
        {{"id": "Root Node ID", "label": "Root Node Label", "group": "root"}},
        {{"id": "Parent Category", "label": "Parent Category", "group": "parent"}},
        {{"id": "Consolidated Topic", "label": "Consolidated Topic", "group": "child"}},
        {{"id": "Original Topic 1", "label": "Original Topic 1", "group": "grandchild"}}
      ],
      "edges": [
        {{"source": "Root Node ID", "target": "Parent Category"}},
        {{"source": "Parent Category", "target": "Consolidated Topic"}},
        {{"source": "Consolidated Topic", "target": "Original Topic 1"}}
      ],
      "consolidation_map": {{
        "Consolidated Topic A": ["Original Topic 1", "Original Topic 2"],
        "Consolidated Topic B": ["Original Topic 3"]
      }}
    }}
    ```

    # List of Original Topics to Organize:
    - {topics_list_str}

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
        # Define the node type for streamlit-flow, making the root an 'input' node for visual distinction
        node_type = 'input' if group == 'root' else 'default'
        
        flow_node = StreamlitFlowNode(
            id=node['id'],
            pos=(0, 0),  # Position is handled by TreeLayout
            data={'label': node['label']},
            node_type=node_type,
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
        with st.spinner("Processing document... This may take a moment."):
            chunks = get_semantic_chunks(user_text)
            with ThreadPoolExecutor(max_workers=5) as executor:
                tagged_data = list(executor.map(generate_topic_for_chunk, chunks))
            graph_data = generate_hierarchical_graph(tagged_data, doc_title="Document Analysis")
            flow_nodes, flow_edges = create_flow_elements(graph_data)
            
            st.session_state.graph_data = graph_data
            st.session_state.tagged_data = tagged_data
            st.session_state.flow_state = StreamlitFlowState(nodes=flow_nodes, edges=flow_edges)
        st.success("Diagram generated!")

if st.session_state.graph_data:
    st.header("Topic Hierarchy Diagram")
    flow_state = streamlit_flow('tree_layout',
                                st.session_state.flow_state,
                                layout=TreeLayout(direction='down'), # Use the automatic tree layout
                                fit_view=True,
                                get_node_on_click=True,
                                height=600)

    # Display details of the clicked node in the sidebar
    if flow_state and flow_state.get_selected_nodes():
        selected_node = flow_state.get_selected_nodes()[0]
        selected_node_id = selected_node['id']

        with st.sidebar:
            st.markdown("---")
            st.header("Node Details")
            node_info = next((n for n in st.session_state.graph_data.get('nodes', []) if n['id'] == selected_node_id), None)
            
            if node_info:
                st.subheader(node_info['label'])
                group = node_info.get('group')

                if group == 'child': # Consolidated node
                    original_topics = st.session_state.graph_data.get('consolidation_map', {}).get(selected_node_id, [])
                    for topic in original_topics:
                        chunk_info = next((item for item in st.session_state.tagged_data if item.get('main_topic') == topic), None)
                        if chunk_info:
                            with st.expander(f"Source: {topic}", expanded=True):
                                st.markdown(f"**Summary:** {chunk_info.get('summary', 'N/A')}")
                                st.write("**Tags:**", chunk_info.get('tags', []))
                
                elif group == 'grandchild': # Original chunk node
                    chunk_info = next((item for item in st.session_state.tagged_data if item.get('main_topic') == selected_node_id), None)
                    if chunk_info:
                        st.markdown(f"**Summary:** {chunk_info.get('summary', 'N/A')}")
                        st.write("**Tags:**", chunk_info.get('tags', []))
                        st.info(f"**Original Text:**\n\n{chunk_info.get('original_chunk', 'N/A')}")
                
                else: # Root or Parent node
                    st.info("This is a high-level category node.")
else:
    st.info("Enter text in the sidebar and click 'Generate Diagram' to begin.")
