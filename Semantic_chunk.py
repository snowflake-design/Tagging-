import streamlit as st
from llama_index.core import Settings, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
import tempfile

# --- UI CONFIGURATION ---
st.set_page_config(layout="wide")
st.title("📄 Advanced Semantic Chunker")
st.markdown("""
This tool splits text into semantically coherent chunks. Choose your input method below: paste text directly or upload a PDF.
""")

# --- SIDEBAR & MODEL CONFIGURATION ---
st.sidebar.header("⚙️ Configuration")

local_model_path = st.sidebar.text_input(
    "Enter Local Model Path",
    value="C:/Users/YourUser/models/all-MiniLM-L6-v2",
    help="Provide the absolute path to your downloaded sentence-transformer model folder."
)

st.sidebar.header("🔧 Controls")
breakpoint_percentile_threshold = st.sidebar.slider(
    "Breakpoint Percentile Threshold",
    min_value=80,
    max_value=99,
    value=95,
    help="A **lower** value is more sensitive to topic shifts, creating **more, smaller chunks**. A **higher** value is less sensitive, creating **fewer, larger chunks**."
)

# --- MODEL LOADING & LLAMAINDEX SETTINGS ---
@st.cache_resource
def configure_llama_index(model_path):
    """Loads the embedding model from a local path and configures LlamaIndex settings."""
    st.write(f"Attempting to load model from: {model_path}")
    try:
        embed_model = HuggingFaceEmbedding(model_name=model_path)
        Settings.embed_model = embed_model
        st.success("Model loaded and configured successfully!")
        return embed_model
    except Exception as e:
        st.error(f"Failed to load model. Please check the path. Error: {e}")
        return None

embed_model = configure_llama_index(local_model_path)


# --- CORE PROCESSING FUNCTION ---
def process_and_display_chunks(text_content):
    """Takes text, performs chunking, and displays the results."""
    st.success("Text loaded! Now performing semantic chunking...")
    
    # Wrap the text in a LlamaIndex Document object
    documents = [Document(text=text_content)]

    # Instantiate the SemanticSplitterNodeParser
    parser = SemanticSplitterNodeParser(
        breakpoint_percentile_threshold=breakpoint_percentile_threshold
    )
    
    # Get the nodes (chunks)
    nodes = parser.get_nodes_from_documents(documents)

    st.header(f"Found {len(nodes)} Semantic Chunks")
    st.markdown(f"_(Based on a **{breakpoint_percentile_threshold}th percentile** similarity threshold)_")

    # Display the chunks
    for i, node in enumerate(nodes):
        with st.expander(f"**Chunk {i + 1}** - ({len(node.get_content().split())} words)"):
            st.write(node.get_content())


# --- MAIN UI TABS ---
if embed_model:
    tab1, tab2 = st.tabs(["📤 Upload PDF", "📋 Paste Text"])

    # --- PDF UPLOAD TAB ---
    with tab1:
        st.header("Option 1: Upload a PDF Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")
        
        if uploaded_file is not None:
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    st.write("Extracting text with LlamaIndex's PDF Reader...")
                    # Use SimpleDirectoryReader which is robust for PDFs
                    reader = SimpleDirectoryReader(input_dir=temp_dir)
                    docs_from_pdf = reader.load_data()
                    
                    pdf_text = "".join([doc.get_content() for doc in docs_from_pdf])

                    if pdf_text.strip():
                        process_and_display_chunks(pdf_text)
                    else:
                        st.error("Could not extract any text from the PDF. The PDF might be an image or have a complex layout.")
            except Exception as e:
                st.error(f"An error occurred while processing the PDF: {e}")

    # --- TEXT PASTE TAB ---
    with tab2:
        st.header("Option 2: Paste Your Text")
        pasted_text = st.text_area("Paste your paragraph or text here...", height=300, label_visibility="collapsed")

        if st.button("Chunk Pasted Text"):
            if pasted_text.strip():
                process_and_display_chunks(pasted_text)
            else:
                st.warning("Please paste some text into the text area before chunking.")
else:
    st.warning("Model could not be loaded. Please verify the local path in the sidebar.")
