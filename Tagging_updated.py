import streamlit as st
import pickle
import hashlib
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from pathlib import Path
import traceback

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# PDF processing
import PyPDF2

# Streamlit Flow - Updated imports for v1.6.1
from streamlit_flow import streamlit_flow
from streamlit_flow.elements import StreamlitFlowNode, StreamlitFlowEdge
from streamlit_flow.state import StreamlitFlowState
from streamlit_flow.layouts import TreeLayout

# Set page config
st.set_page_config(
    page_title="🏷️ Semantic Document Tagger",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (keeping your existing styles)
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .chunk-preview {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 5px;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    
    .tag-badge {
        display: inline-block;
        background: #667eea;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    
    .success-message {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .info-box {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    
    .tree-node {
        border: 2px solid #333;
        border-radius: 8px;
        padding: 8px;
        margin: 4px;
        background: #f8f9fa;
        cursor: pointer;
    }
    
    .tree-node:hover {
        background: #e9ecef;
    }
    
    .tree-level-0 { border-color: #2E86AB; background: #e3f2fd; }
    .tree-level-1 { border-color: #A23B72; background: #f3e5f5; }
    .tree-level-2 { border-color: #F18F01; background: #fff3e0; }
    .tree-level-3 { border-color: #C73E1D; background: #ffebee; }
</style>
""", unsafe_allow_html=True)

# Improved LLM response function with mock implementation
def abc_response(prompt: str, verbose: bool = True) -> str:
    """
    Improved mock LLM function with realistic processing time and responses
    Replace this with your actual LLM implementation
    """
    if verbose:
        st.write(f"🤖 Processing LLM request...")
    
    # Simulate realistic processing time
    time.sleep(0.5)  # Reduced from potentially long delays
    
    if "hierarchy" in prompt.lower():
        # Mock hierarchy response based on common document patterns
        mock_hierarchy = {
            "nodes": [
                {
                    "id": "root",
                    "label": "Document Overview",
                    "level": 0,
                    "parent": None,
                    "confidence": 0.95,
                    "chunks": [],
                    "node_type": "input",
                    "position": {"x": 250, "y": 50}
                },
                {
                    "id": "main_topic_1",
                    "label": "Main Topic",
                    "level": 1,
                    "parent": "root",
                    "confidence": 0.88,
                    "chunks": ["chunk_1", "chunk_2"],
                    "node_type": "default",
                    "position": {"x": 150, "y": 150}
                },
                {
                    "id": "subtopic_1",
                    "label": "Key Concept",
                    "level": 2,
                    "parent": "main_topic_1",
                    "confidence": 0.82,
                    "chunks": ["chunk_1"],
                    "node_type": "output",
                    "position": {"x": 100, "y": 250}
                },
                {
                    "id": "details_1",
                    "label": "Specific Details",
                    "level": 2,
                    "parent": "main_topic_1",
                    "confidence": 0.79,
                    "chunks": ["chunk_2"],
                    "node_type": "output",
                    "position": {"x": 200, "y": 250}
                }
            ],
            "edges": [
                {
                    "id": "root-main_topic_1",
                    "source": "root",
                    "target": "main_topic_1",
                    "relationship": "parent-child"
                },
                {
                    "id": "main_topic_1-subtopic_1",
                    "source": "main_topic_1",
                    "target": "subtopic_1",
                    "relationship": "parent-child"
                },
                {
                    "id": "main_topic_1-details_1",
                    "source": "main_topic_1",
                    "target": "details_1",
                    "relationship": "parent-child"
                }
            ]
        }
        return json.dumps(mock_hierarchy)
    else:
        # Generate mock tags based on chunk content
        mock_tags = ["document_analysis", "content_structure", "semantic_tags", "information_extraction", "text_processing"]
        return ", ".join(mock_tags)

class DocumentProcessor:
    def __init__(self):
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize embedding model with better error handling
        if 'embed_model' not in st.session_state:
            try:
                with st.spinner("🔄 Loading embedding model (this may take a moment on first run)..."):
                    st.session_state.embed_model = HuggingFaceEmbedding(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        cache_folder="./models"
                    )
                st.success("✅ Embedding model loaded successfully!")
            except Exception as e:
                st.error(f"❌ Error loading embedding model: {str(e)}")
                st.warning("Using fallback text processing...")
                st.session_state.embed_model = None
        
        self.embed_model = st.session_state.embed_model
        
        # Initialize semantic splitter with fallback
        if self.embed_model:
            self.semantic_splitter = SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=self.embed_model
            )
        else:
            self.semantic_splitter = None
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from uploaded PDF file with verbose output"""
        try:
            st.write("📄 Reading PDF file...")
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            st.write(f"📖 Found {len(pdf_reader.pages)} pages")
            
            text = ""
            progress_bar = st.progress(0)
            
            for i, page in enumerate(pdf_reader.pages):
                st.write(f"📑 Processing page {i+1}/{len(pdf_reader.pages)}")
                text += page.extract_text() + "\n"
                progress_bar.progress((i + 1) / len(pdf_reader.pages))
            
            st.write(f"✅ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            st.error(f"❌ Error extracting text from PDF: {str(e)}")
            return ""
    
    def get_file_hash(self, content: str) -> str:
        """Generate hash for content caching"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def load_from_cache(self, file_hash: str) -> Dict[str, Any]:
        """Load processed results from cache"""
        cache_file = self.cache_dir / f"{file_hash}.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def save_to_cache(self, file_hash: str, data: Dict[str, Any]):
        """Save processed results to cache"""
        cache_file = self.cache_dir / f"{file_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    
    def simple_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
        """Simple text chunking fallback when semantic chunking fails"""
        st.write("🔄 Using simple text chunking...")
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            if chunk_text.strip():
                chunks.append({
                    'id': f'chunk_{len(chunks)+1}',
                    'text': chunk_text.strip(),
                    'metadata': {'start_char': i, 'end_char': i + len(chunk_text)},
                    'start_char_idx': i,
                    'end_char_idx': i + len(chunk_text)
                })
        
        st.write(f"✅ Created {len(chunks)} chunks using simple chunking")
        return chunks
    
    def semantic_chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Split text into semantic chunks with fallback and verbose output"""
        st.write("🧩 Starting semantic chunking process...")
        
        # Check if text is too long and suggest chunking
        if len(text) > 50000:
            st.warning(f"⚠️ Large document ({len(text)} characters). This may take longer...")
        
        try:
            if not self.semantic_splitter:
                st.warning("⚠️ Semantic splitter not available, using simple chunking")
                return self.simple_chunk_text(text)
            
            st.write("📝 Creating document object...")
            document = Document(text=text)
            
            st.write("🔄 Running semantic analysis...")
            with st.spinner("Analyzing semantic boundaries..."):
                nodes = self.semantic_splitter.get_nodes_from_documents([document])
            
            st.write(f"✅ Found {len(nodes)} semantic boundaries")
            
            # Convert to our format with progress
            chunks = []
            progress_bar = st.progress(0)
            
            for i, node in enumerate(nodes):
                chunks.append({
                    'id': f'chunk_{i+1}',
                    'text': node.text,
                    'metadata': node.metadata,
                    'start_char_idx': getattr(node, 'start_char_idx', None),
                    'end_char_idx': getattr(node, 'end_char_idx', None)
                })
                progress_bar.progress((i + 1) / len(nodes))
            
            st.write(f"✅ Successfully created {len(chunks)} semantic chunks")
            return chunks
            
        except Exception as e:
            st.error(f"❌ Error in semantic chunking: {str(e)}")
            st.write("🔄 Falling back to simple chunking...")
            return self.simple_chunk_text(text)
    
    def generate_tags_for_chunk(self, chunk: Dict[str, Any]) -> List[str]:
        """Generate tags for a single chunk with verbose output"""
        prompt = f"""
        Analyze the following text chunk and generate 3-4 relevant tags that capture:
        1. Major topic/theme discussed
        2. Key entities mentioned
        3. Content type/category
        
        Text chunk:
        {chunk['text'][:300]}...
        
        Return only the tags as a comma-separated list:
        """
        
        try:
            response = abc_response(prompt, verbose=False)
            tags = [tag.strip() for tag in response.split(',') if tag.strip()]
            return tags[:4]  # Limit to 4 tags for faster processing
        except Exception as e:
            st.write(f"⚠️ Error generating tags for {chunk['id']}: {str(e)}")
            return [f"content_{chunk['id']}", "text_segment"]
    
    def generate_tags_parallel(self, chunks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Generate tags for all chunks in parallel with verbose output"""
        st.write(f"🏷️ Starting tag generation for {len(chunks)} chunks...")
        chunk_tags = {}
        
        # Reduce max workers for faster individual processing
        max_workers = min(3, len(chunks))
        st.write(f"🔄 Using {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self.generate_tags_for_chunk, chunk): chunk['id'] 
                for chunk in chunks
            }
            
            # Collect results as they complete
            progress_bar = st.progress(0)
            status_text = st.empty()
            completed = 0
            total = len(chunks)
            
            for future in as_completed(future_to_chunk):
                chunk_id = future_to_chunk[future]
                try:
                    tags = future.result()
                    chunk_tags[chunk_id] = tags
                    completed += 1
                    status_text.write(f"✅ Processed chunk {completed}/{total}: {chunk_id}")
                except Exception as e:
                    st.write(f"❌ Error processing chunk {chunk_id}: {str(e)}")
                    chunk_tags[chunk_id] = [f"error_{chunk_id}"]
                    completed += 1
                
                progress_bar.progress(completed / total)
        
        st.write(f"🎉 Tag generation complete! Generated tags for {len(chunk_tags)} chunks")
        return chunk_tags
    
    def create_tag_hierarchy(self, chunk_tags: Dict[str, List[str]], chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create hierarchical tag structure with verbose output and fallback"""
        st.write("🌳 Building tag hierarchy...")
        
        # Collect all unique tags
        all_tags = set()
        for tags in chunk_tags.values():
            all_tags.update(tags)
        
        st.write(f"📊 Found {len(all_tags)} unique tags across all chunks")
        
        # Create tag-to-chunks mapping
        tag_chunk_mapping = {}
        for chunk_id, tags in chunk_tags.items():
            for tag in tags:
                if tag not in tag_chunk_mapping:
                    tag_chunk_mapping[tag] = []
                tag_chunk_mapping[tag].append(chunk_id)
        
        # Simplified hierarchy creation for faster processing
        try:
            st.write("🤖 Generating hierarchy with LLM...")
            
            # Simplified prompt for faster processing
            simplified_prompt = f"""
            Create a simple hierarchy from these tags: {', '.join(list(all_tags)[:10])}
            Return JSON with nodes and edges for a tree structure.
            """
            
            response = abc_response(simplified_prompt)
            hierarchy = json.loads(response)
            
            # Enhance with chunk associations
            for node in hierarchy.get('nodes', []):
                tag_name = node['label'].lower().replace(' ', '_')
                node['chunks'] = tag_chunk_mapping.get(tag_name, [])
            
            st.write("✅ Hierarchy created successfully!")
            return hierarchy
            
        except Exception as e:
            st.write(f"⚠️ LLM hierarchy failed: {str(e)}")
            st.write("🔄 Creating simple fallback hierarchy...")
            return self.create_simple_hierarchy(all_tags, tag_chunk_mapping)
    
    def create_simple_hierarchy(self, all_tags: set, tag_chunk_mapping: Dict) -> Dict[str, Any]:
        """Create a simple fallback hierarchy"""
        nodes = [
            {
                "id": "root",
                "label": "Document Tags",
                "level": 0,
                "parent": None,
                "chunks": [],
                "node_type": "input",
                "position": {"x": 250, "y": 50}
            }
        ]
        
        edges = []
        
        # Add tags as children of root
        for i, tag in enumerate(list(all_tags)[:10]):  # Limit for performance
            node_id = f"tag_{i}"
            nodes.append({
                "id": node_id,
                "label": tag,
                "level": 1,
                "parent": "root",
                "chunks": tag_chunk_mapping.get(tag, []),
                "node_type": "output",
                "position": {"x": 100 + (i % 5) * 100, "y": 150 + (i // 5) * 100}
            })
            
            edges.append({
                "id": f"root-{node_id}",
                "source": "root",
                "target": node_id,
                "relationship": "parent-child"
            })
        
        return {"nodes": nodes, "edges": edges}

def create_fallback_tree_visualization(hierarchy: Dict[str, Any], chunks: List[Dict[str, Any]]) -> None:
    """Create a simple fallback tree visualization using HTML/CSS"""
    st.markdown("### 🌳 Tree Structure (Fallback View)")
    
    # Create a simple hierarchical display
    nodes_by_level = {}
    for node in hierarchy.get('nodes', []):
        level = node.get('level', 0)
        if level not in nodes_by_level:
            nodes_by_level[level] = []
        nodes_by_level[level].append(node)
    
    # Display tree levels
    for level in sorted(nodes_by_level.keys()):
        st.markdown(f"**Level {level}**")
        
        cols = st.columns(min(len(nodes_by_level[level]), 4))
        for i, node in enumerate(nodes_by_level[level]):
            with cols[i % len(cols)]:
                chunk_count = len(node.get('chunks', []))
                st.markdown(f"""
                <div class="tree-node tree-level-{level}">
                    <strong>{node['label']}</strong><br>
                    <small>{chunk_count} chunks</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

def convert_to_streamlit_flow(hierarchy: Dict[str, Any], chunks: List[Dict[str, Any]]) -> tuple:
    """Convert hierarchy to Streamlit Flow format with proper state management"""
    nodes = []
    edges = []
    
    # Create chunk lookup
    chunk_lookup = {chunk['id']: chunk for chunk in chunks}
    
    for node_data in hierarchy.get('nodes', []):
        # Color based on level
        level = node_data.get('level', 0)
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        color = colors[min(level, len(colors)-1)]
        
        # Size based on chunks
        chunk_count = len(node_data.get('chunks', []))
        width = max(100, min(200, 100 + chunk_count * 10))
        
        node = StreamlitFlowNode(
            id=node_data['id'],
            pos=(node_data['position']['x'], node_data['position']['y']),
            data={
                'content': f"**{node_data['label']}**\n{chunk_count} chunks",
                'label': node_data['label'],
                'level': level,
                'chunks': node_data.get('chunks', []),
                'chunk_count': chunk_count
            },
            node_type=node_data.get('node_type', 'default'),
            style={
                'background': color,
                'border': '2px solid #333',
                'color': 'white',
                'borderRadius': '10px',
                'width': f'{width}px',
                'height': '60px'
            },
            source_position='bottom',
            target_position='top'
        )
        nodes.append(node)
    
    for edge_data in hierarchy.get('edges', []):
        edge = StreamlitFlowEdge(
            id=edge_data['id'],
            source=edge_data['source'],
            target=edge_data['target'],
            animated=True,
            style={'stroke': '#333', 'strokeWidth': 2}
        )
        edges.append(edge)
    
    return nodes, edges

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏷️ Semantic Document Tagger</h1>
        <p>Intelligent document analysis with hierarchical tag visualization</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Sidebar for input
    with st.sidebar:
        st.markdown("### 📄 Document Input")
        
        input_type = st.radio(
            "Choose input method:",
            ["📝 Paste Text", "📁 Upload PDF"],
            horizontal=True
        )
        
        text_content = ""
        
        if input_type == "📝 Paste Text":
            text_content = st.text_area(
                "Paste your text here:",
                height=300,
                placeholder="Enter your document text here..."
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload PDF file",
                type=['pdf'],
                help="Upload a PDF file to extract and analyze text"
            )
            
            if uploaded_file is not None:
                text_content = processor.extract_text_from_pdf(uploaded_file)
        
        # Processing settings
        st.markdown("### ⚙️ Processing Settings")
        
        use_fast_mode = st.checkbox("🚀 Fast Mode", value=True, help="Use simplified processing for faster results")
        max_chunks = st.slider("Max Chunks", 5, 50, 20, help="Limit chunks for faster processing")
        
        process_button = st.button(
            "🚀 Process Document",
            disabled=not text_content,
            use_container_width=True
        )
    
    # Main content area
    if text_content and process_button:
        # Create processing status container
        status_container = st.container()
        
        with status_container:
            st.markdown("### 🔄 Processing Status")
            
            # Check cache first
            file_hash = processor.get_file_hash(text_content)
            cached_result = processor.load_from_cache(file_hash)
            
            if cached_result and not use_fast_mode:
                st.success("📱 Using cached results")
                chunks = cached_result['chunks']
                chunk_tags = cached_result['chunk_tags']
                hierarchy = cached_result['hierarchy']
            else:
                # Process document with verbose output
                try:
                    # Step 1: Chunking
                    st.markdown("#### 🧩 Step 1: Text Chunking")
                    chunks = processor.semantic_chunk_text(text_content)
                    
                    # Limit chunks if in fast mode
                    if use_fast_mode and len(chunks) > max_chunks:
                        st.warning(f"⚡ Fast mode: Using first {max_chunks} chunks out of {len(chunks)}")
                        chunks = chunks[:max_chunks]
                    
                    if not chunks:
                        st.error("❌ No chunks created. Please check your text input.")
                        return
                    
                    # Step 2: Tag Generation
                    st.markdown("#### 🏷️ Step 2: Tag Generation")
                    chunk_tags = processor.generate_tags_parallel(chunks)
                    
                    if not chunk_tags:
                        st.error("❌ No tags generated. Please try again.")
                        return
                    
                    # Step 3: Hierarchy Creation
                    st.markdown("#### 🌳 Step 3: Hierarchy Creation")
                    hierarchy = processor.create_tag_hierarchy(chunk_tags, chunks)
                    
                    if not hierarchy:
                        st.error("❌ Failed to create hierarchy.")
                        return
                    
                    # Cache results
                    if not use_fast_mode:
                        cache_data = {
                            'chunks': chunks,
                            'chunk_tags': chunk_tags,
                            'hierarchy': hierarchy,
                            'processed_at': time.time()
                        }
                        processor.save_to_cache(file_hash, cache_data)
                        st.success("💾 Results cached for future use")
                
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
                    st.error(f"Error details: {traceback.format_exc()}")
                    return
        
        # Display Results
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Semantic Chunks", len(chunks))
        with col2:
            total_tags = sum(len(tags) for tags in chunk_tags.values())
            st.metric("Total Tags", total_tags)
        with col3:
            unique_tags = len(set().union(*chunk_tags.values()))
            st.metric("Unique Tags", unique_tags)
        with col4:
            hierarchy_levels = len(set(node.get('level', 0) for node in hierarchy.get('nodes', [])))
            st.metric("Hierarchy Levels", hierarchy_levels)
        
        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["🌳 Hierarchy Visualization", "📑 Chunks & Tags", "📈 Analysis Summary"])
        
        with tab1:
            st.markdown("### 🌳 Interactive Tag Hierarchy")
            
            # Try Streamlit Flow first, fallback to simple tree
            try:
                if hierarchy and hierarchy.get('nodes'):
                    nodes, edges = convert_to_streamlit_flow(hierarchy, chunks)
                    
                    # Initialize flow state properly for v1.6.1
                    if 'tag_flow_state' not in st.session_state:
                        st.session_state.tag_flow_state = StreamlitFlowState(
                            key="tag_hierarchy",
                            nodes=nodes,
                            edges=edges,
                            layout=TreeLayout(direction="down")
                        )
                    
                    # Display the flow
                    try:
                        selected_state = streamlit_flow(
                            key='tag_hierarchy_flow',
                            state=st.session_state.tag_flow_state,
                            height=600,
                            fit_view=True,
                            pan_on_drag=True,
                            zoom_on_scroll=True
                        )
                        
                        # Update state
                        if selected_state:
                            st.session_state.tag_flow_state = selected_state
                        
                        # Handle selection
                        if hasattr(st.session_state.tag_flow_state, 'selected_id') and st.session_state.tag_flow_state.selected_id:
                            selected_node = next((n for n in nodes if n.id == st.session_state.tag_flow_state.selected_id), None)
                            if selected_node:
                                st.success(f"Selected: {selected_node.data['label']}")
                                st.write(f"Chunks: {selected_node.data['chunk_count']}")
                    
                    except Exception as flow_error:
                        st.warning(f"⚠️ Streamlit Flow error: {str(flow_error)}")
                        st.markdown("**Falling back to simple tree view:**")
                        create_fallback_tree_visualization(hierarchy, chunks)
                
                else:
                    st.warning("No hierarchy data available")
                    
            except Exception as e:
                st.error(f"❌ Visualization error: {str(e)}")
                create_fallback_tree_visualization(hierarchy, chunks)
        
        with tab2:
            st.markdown("### 📑 Chunks and Generated Tags")
            
            # Add search functionality
            search_term = st.text_input("🔍 Search chunks:", placeholder="Enter keywords to filter chunks...")
            
            filtered_chunks = chunks
            if search_term:
                filtered_chunks = [
                    chunk for chunk in chunks 
                    if search_term.lower() in chunk['text'].lower()
                ]
                st.info(f"Found {len(filtered_chunks)} chunks matching '{search_term}'")
            
            # Display chunks with pagination
            chunks_per_page = 5
            total_pages = (len(filtered_chunks) + chunks_per_page - 1) // chunks_per_page
            
            if total_pages > 1:
                page = st.selectbox("Page", range(1, total_pages + 1)) - 1
                start_idx = page * chunks_per_page
                end_idx = min(start_idx + chunks_per_page, len(filtered_chunks))
                display_chunks = filtered_chunks[start_idx:end_idx]
                st.info(f"Showing chunks {start_idx + 1}-{end_idx} of {len(filtered_chunks)}")
            else:
                display_chunks = filtered_chunks
            
            for i, chunk in enumerate(display_chunks):
                with st.expander(f"Chunk {chunk['id']} ({len(chunk['text'])} characters)"):
                    st.markdown("**Text Preview:**")
                    st.markdown(f"""
                    <div class="chunk-preview">
                        {chunk['text'][:500]}{'...' if len(chunk['text']) > 500 else ''}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**Generated Tags:**")
                    tags = chunk_tags.get(chunk['id'], [])
                    if tags:
                        tags_html = ''.join([f'<span class="tag-badge">{tag}</span>' for tag in tags])
                        st.markdown(tags_html, unsafe_allow_html=True)
                    else:
                        st.warning("No tags generated for this chunk")
                    
                    # Additional metadata
                    if chunk.get('metadata'):
                        st.markdown("**Metadata:**")
                        st.json(chunk['metadata'])
        
        with tab3:
            st.markdown("### 📈 Analysis Summary")
            
            # Tag frequency analysis
            tag_frequency = {}
            for tags in chunk_tags.values():
                for tag in tags:
                    tag_frequency[tag] = tag_frequency.get(tag, 0) + 1
            
            if tag_frequency:
                # Sort by frequency
                sorted_tags = sorted(tag_frequency.items(), key=lambda x: x[1], reverse=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Most Frequent Tags:**")
                    for i, (tag, freq) in enumerate(sorted_tags[:10]):
                        st.markdown(f"{i+1}. **{tag}**: {freq} occurrences")
                
                with col2:
                    st.markdown("**Tag Distribution:**")
                    # Create a simple bar chart representation
                    for tag, freq in sorted_tags[:5]:
                        percentage = (freq / sum(tag_frequency.values())) * 100
                        st.markdown(f"**{tag}**: {percentage:.1f}%")
                        st.progress(percentage / 100)
            
            # Document insights
            st.markdown("---")
            st.markdown("**Document Insights:**")
            
            avg_chunk_size = len(text_content) // len(chunks) if chunks else 0
            avg_tags_per_chunk = total_tags / len(chunks) if chunks else 0
            tag_diversity = (unique_tags / total_tags * 100) if total_tags > 0 else 0
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown(f"""
                <div class="info-box">
                    📄 <strong>Document Length:</strong> {len(text_content):,} characters<br>
                    🧩 <strong>Average Chunk Size:</strong> {avg_chunk_size:,} characters<br>
                    📊 <strong>Chunk Distribution:</strong> {len(chunks)} total chunks
                </div>
                """, unsafe_allow_html=True)
            
            with insight_col2:
                st.markdown(f"""
                <div class="info-box">
                    🏷️ <strong>Tags per Chunk:</strong> {avg_tags_per_chunk:.1f} average<br>
                    🎯 <strong>Tag Diversity:</strong> {tag_diversity:.1f}% unique tags<br>
                    🌳 <strong>Hierarchy Depth:</strong> {hierarchy_levels} levels
                </div>
                """, unsafe_allow_html=True)
            
            # Export options
            st.markdown("---")
            st.markdown("### 💾 Export Options")
            
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                if st.button("📄 Export Tags as JSON"):
                    export_data = {
                        'chunks': chunks,
                        'chunk_tags': chunk_tags,
                        'hierarchy': hierarchy,
                        'tag_frequency': tag_frequency,
                        'metadata': {
                            'total_chunks': len(chunks),
                            'total_tags': total_tags,
                            'unique_tags': unique_tags,
                            'processed_at': time.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    }
                    
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(export_data, indent=2),
                        file_name=f"document_analysis_{int(time.time())}.json",
                        mime="application/json"
                    )
            
            with export_col2:
                if st.button("📊 Export Tag Summary"):
                    summary_text = f"""Document Analysis Summary
Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

Document Statistics:
- Total Characters: {len(text_content):,}
- Total Chunks: {len(chunks)}
- Average Chunk Size: {avg_chunk_size:,} characters
- Total Tags: {total_tags}
- Unique Tags: {unique_tags}
- Tag Diversity: {tag_diversity:.1f}%

Top 10 Most Frequent Tags:
"""
                    for i, (tag, freq) in enumerate(sorted_tags[:10]):
                        summary_text += f"{i+1}. {tag}: {freq} occurrences\n"
                    
                    st.download_button(
                        label="Download Summary",
                        data=summary_text,
                        file_name=f"document_summary_{int(time.time())}.txt",
                        mime="text/plain"
                    )
            
            with export_col3:
                if st.button("🌳 Export Hierarchy"):
                    hierarchy_text = "Document Tag Hierarchy\n" + "="*30 + "\n\n"
                    
                    # Group nodes by level
                    nodes_by_level = {}
                    for node in hierarchy.get('nodes', []):
                        level = node.get('level', 0)
                        if level not in nodes_by_level:
                            nodes_by_level[level] = []
                        nodes_by_level[level].append(node)
                    
                    # Format hierarchy
                    for level in sorted(nodes_by_level.keys()):
                        hierarchy_text += f"Level {level}:\n"
                        for node in nodes_by_level[level]:
                            indent = "  " * level
                            chunk_count = len(node.get('chunks', []))
                            hierarchy_text += f"{indent}- {node['label']} ({chunk_count} chunks)\n"
                        hierarchy_text += "\n"
                    
                    st.download_button(
                        label="Download Hierarchy",
                        data=hierarchy_text,
                        file_name=f"document_hierarchy_{int(time.time())}.txt",
                        mime="text/plain"
                    )
    
    # Footer with tips
    st.markdown("---")
    st.markdown("### 💡 Tips for Better Results")
    
    tip_col1, tip_col2 = st.columns(2)
    
    with tip_col1:
        st.markdown("""
        **🚀 Performance Tips:**
        - Use Fast Mode for quick analysis
        - Limit document size to < 50k characters for best performance
        - Enable caching by turning off Fast Mode for repeated analysis
        - Use simple text format for faster processing
        """)
    
    with tip_col2:
        st.markdown("""
        **🎯 Quality Tips:**
        - Provide well-structured documents for better chunking
        - Use documents with clear sections and topics
        - Review generated tags and provide feedback
        - Export results for further analysis
        """)
    
    # Debug information (optional)
    if st.checkbox("🔧 Show Debug Info"):
        st.markdown("### 🔧 Debug Information")
        
        debug_col1, debug_col2 = st.columns(2)
        
        with debug_col1:
            st.markdown("**Session State Keys:**")
            st.write(list(st.session_state.keys()))
        
        with debug_col2:
            st.markdown("**Cache Directory:**")
            if processor.cache_dir.exists():
                cache_files = list(processor.cache_dir.glob("*.pkl"))
                st.write(f"Cache files: {len(cache_files)}")
                for cache_file in cache_files[:5]:  # Show first 5
                    st.write(f"- {cache_file.name}")
            else:
                st.write("Cache directory not found")

if __name__ == "__main__":
    main()
