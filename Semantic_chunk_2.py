import streamlit as st
import PyPDF2
from io import BytesIO
import logging
from typing import List, Dict, Any
import os
from pathlib import Path
import asyncio
import concurrent.futures
from functools import lru_cache
import hashlib
import pickle

# LlamaIndex imports
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FastSemanticChunker:
    """
    High-performance semantic chunker with caching and parallelization.
    """
    
    def __init__(self, 
                 local_model_path: str = None,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 buffer_size: int = 1,
                 breakpoint_percentile_threshold: int = 95):
        """
        Initialize the fast semantic chunker.
        
        Args:
            local_model_path: Path to local embedding model
            model_name: HuggingFace model name (fallback if local path not available)
            buffer_size: Number of sentences to group when computing embeddings
            breakpoint_percentile_threshold: Percentile threshold for breakpoints
        """
        self.local_model_path = local_model_path
        self.model_name = model_name
        self.buffer_size = buffer_size
        self.breakpoint_percentile_threshold = breakpoint_percentile_threshold
        self.embed_model = None
        self.splitter = None
        self.cache_dir = Path(".cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize the embedding model
        self._initialize_models()
    
    @lru_cache(maxsize=1)
    def _initialize_models(self):
        """Initialize the embedding model and semantic splitter with caching."""
        try:
            model_path = self.local_model_path or self.model_name
            st.info(f"🔄 Loading model from: {model_path}")
            
            # Use local path if available, otherwise download
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_path,
                device="cpu",
                embed_batch_size=32,  # Increased batch size for speed
                max_length=512,
                cache_folder=str(self.cache_dir / "models") if not self.local_model_path else None
            )
            
            # Initialize semantic splitter with optimizations
            self.splitter = SemanticSplitterNodeParser(
                buffer_size=self.buffer_size,
                breakpoint_percentile_threshold=self.breakpoint_percentile_threshold,
                embed_model=self.embed_model
            )
            
            st.success(f"✅ Model loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            raise
    
    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text to use as cache key."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _load_from_cache(self, text_hash: str) -> List[Dict[str, Any]]:
        """Load chunks from cache if available."""
        cache_file = self.cache_dir / f"chunks_{text_hash}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                cache_file.unlink(missing_ok=True)
        return None
    
    def _save_to_cache(self, text_hash: str, chunks: List[Dict[str, Any]]):
        """Save chunks to cache for future use."""
        cache_file = self.cache_dir / f"chunks_{text_hash}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(chunks, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def extract_text_from_pdf_fast(self, pdf_file) -> str:
        """
        Fast PDF text extraction with parallel processing.
        
        Args:
            pdf_file: Streamlit uploaded file object
            
        Returns:
            Extracted text as string
        """
        try:
            st.info("📄 Extracting text from PDF...")
            
            pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
            total_pages = len(pdf_reader.pages)
            
            # Parallel text extraction for multiple pages
            def extract_page_text(page_num):
                return pdf_reader.pages[page_num].extract_text()
            
            # Use ThreadPoolExecutor for I/O bound operations
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                page_texts = list(executor.map(extract_page_text, range(total_pages)))
            
            text = "\n\n".join(page_texts)
            st.success(f"✅ Extracted text from {total_pages} pages")
            
            return text
            
        except Exception as e:
            st.error(f"❌ Error extracting PDF: {str(e)}")
            return ""
    
    def perform_semantic_chunking_fast(self, text: str) -> List[Dict[str, Any]]:
        """
        Fast semantic chunking with caching and optimization.
        
        Args:
            text: Input text to chunk
            
        Returns:
            List of dictionaries containing chunk information
        """
        try:
            # Check cache first
            text_hash = self._get_text_hash(text)
            cached_chunks = self._load_from_cache(text_hash)
            
            if cached_chunks:
                st.success(f"✅ Loaded {len(cached_chunks)} chunks from cache!")
                return cached_chunks
            
            st.info("🧠 Performing semantic chunking...")
            
            # Create document and perform chunking
            document = Document(text=text)
            nodes = self.splitter.get_nodes_from_documents([document])
            
            # Prepare chunks with minimal processing
            chunks_info = []
            for i, node in enumerate(nodes):
                chunk_info = {
                    'index': i + 1,
                    'text': node.text,
                    'node_id': node.node_id
                }
                chunks_info.append(chunk_info)
            
            # Cache the results
            self._save_to_cache(text_hash, chunks_info)
            
            st.success(f"✅ Created {len(chunks_info)} semantic chunks!")
            return chunks_info
            
        except Exception as e:
            st.error(f"❌ Error during chunking: {str(e)}")
            return []

def main():
    """Main Streamlit application."""
    
    st.set_page_config(
        page_title="Fast Semantic Chunker",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ Fast Semantic Chunker with Local Models")
    st.markdown("High-performance semantic chunking with local embedding models and caching.")
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Local model path input
    local_model_path = st.sidebar.text_input(
        "Local Model Path (optional)",
        placeholder="/path/to/your/local/model",
        help="Path to local embedding model directory. Leave empty to download model."
    )
    
    # Model selection (fallback)
    model_name = st.sidebar.selectbox(
        "Model (if no local path)",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2", 
            "BAAI/bge-small-en-v1.5"
        ]
    )
    
    # Advanced settings
    with st.sidebar.expander("Advanced Settings"):
        buffer_size = st.slider("Buffer Size", 1, 5, 1)
        threshold = st.slider("Breakpoint Threshold (%)", 50, 99, 95)
        
        # Cache management
        if st.button("🗑️ Clear Cache"):
            cache_dir = Path(".cache")
            if cache_dir.exists():
                for file in cache_dir.glob("chunks_*.pkl"):
                    file.unlink()
                st.success("Cache cleared!")
    
    # Initialize chunker
    if 'chunker' not in st.session_state:
        with st.spinner("Initializing fast chunker..."):
            try:
                st.session_state.chunker = FastSemanticChunker(
                    local_model_path=local_model_path if local_model_path else None,
                    model_name=model_name,
                    buffer_size=buffer_size,
                    breakpoint_percentile_threshold=threshold
                )
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
                st.stop()
    
    # Input section
    st.header("📥 Input")
    input_method = st.radio("Input method:", ["Paste Text", "Upload PDF"], horizontal=True)
    
    text_content = ""
    
    if input_method == "Paste Text":
        text_content = st.text_area(
            "Paste your text:",
            height=200,
            placeholder="Enter text for semantic chunking..."
        )
        
    elif input_method == "Upload PDF":
        uploaded_file = st.file_uploader("Upload PDF:", type=['pdf'])
        
        if uploaded_file:
            if st.button("🔄 Extract Text", type="primary"):
                text_content = st.session_state.chunker.extract_text_from_pdf_fast(uploaded_file)
    
    # Chunking section
    if text_content.strip():
        if st.button("🚀 Perform Semantic Chunking", type="primary"):
            with st.spinner("Processing..."):
                chunks = st.session_state.chunker.perform_semantic_chunking_fast(text_content)
                
                if chunks:
                    st.session_state.chunks = chunks
    
    # Display results
    if 'chunks' in st.session_state and st.session_state.chunks:
        st.header("📊 Results")
        
        chunks = st.session_state.chunks
        st.info(f"Generated {len(chunks)} semantic chunks")
        
        # Chunk viewer
        chunk_idx = st.selectbox(
            "Select chunk:",
            range(len(chunks)),
            format_func=lambda x: f"Chunk {x+1}"
        )
        
        if chunk_idx is not None:
            chunk = chunks[chunk_idx]
            st.subheader(f"Chunk {chunk['index']}")
            st.text_area("Content:", chunk['text'], height=300)
        
        # Bulk display option
        if st.checkbox("Show All Chunks"):
            for chunk in chunks:
                with st.expander(f"Chunk {chunk['index']}"):
                    st.text(chunk['text'])
        
        # Export
        if st.button("💾 Download All Chunks"):
            export_text = "\n\n=== NEXT CHUNK ===\n\n".join([chunk['text'] for chunk in chunks])
            st.download_button(
                "📄 Download",
                export_text,
                file_name="semantic_chunks.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()
