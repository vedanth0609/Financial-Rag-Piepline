"""
Advanced Parent-Child Chunking & ChromaDB ingestion pipeline using LlamaIndex
"""
import os
import chromadb
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import init_llm_and_embeddings

# Initialize global LLM and embeddings settings
init_llm_and_embeddings()

# Configuration constants
CHROMA_PERSIST_DIR = "data/production/chroma_db"
CHROMA_COLLECTION_NAME = "financial_rag_collection"
STORAGE_DIR = "data/production/storage"


def setup_chroma_vector_store():
    """
    Task 1: Persistent ChromaDB Vector Store Integration
    Initialize a persistent ChromaDB client and wrap it in LlamaIndex ChromaVectorStore
    """
    # Ensure the directory exists
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    
    # Initialize persistent ChromaDB client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    
    # Get or create collection
    chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Wrap in LlamaIndex ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    return vector_store, chroma_client


def setup_hierarchical_node_parser():
    """
    Task 2: Advanced Hierarchical (Parent-Child) Node Parsing
    Configure HierarchicalNodeParser for multi-granular chunking
    """
    # Configure hierarchical node parser with two resolutions
    # Parent chunks: ~512 tokens (larger contextual blocks)
    # Child chunks: ~128 tokens (focused sub-segments)
    node_parser = HierarchicalNodeParser.from_defaults(
        chunk_sizes=[512, 128],  # Parent and child chunk sizes
        chunk_overlap=32,  # Overlap between chunks
    )
    
    return node_parser


def build_or_load_index(file_path: str):
    """
    Task 3: Automated Ingestion Workflow
    Load document, extract hierarchical nodes, store in ChromaDB with parent context
    """
    print(f"Loading document from: {file_path}")
    
    # Load document text
    with open(file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    # Create LlamaIndex Document
    document = Document(text=document_text)
    
    # Setup hierarchical node parser
    node_parser = setup_hierarchical_node_parser()
    
    # Extract hierarchical nodes (parent-child relationships)
    nodes = node_parser.get_nodes_from_documents([document])
    
    # Separate leaf nodes (child chunks) and root nodes (parent chunks)
    leaf_nodes = get_leaf_nodes(nodes)
    root_nodes = get_root_nodes(nodes)
    
    print(f"Extracted {len(leaf_nodes)} child nodes and {len(root_nodes)} parent nodes")
    
    # Setup ChromaDB vector store
    vector_store, chroma_client = setup_chroma_vector_store()
    
    # Create storage context with vector store
    # Don't use persist_dir initially to avoid loading from non-existent storage
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )
    
    # Build index with only leaf nodes (child chunks for vector retrieval)
    # Parent nodes will be added to docstore separately
    index = VectorStoreIndex(
        nodes=leaf_nodes,
        storage_context=storage_context
    )
    
    # Manually add parent nodes to docstore so AutoMergingRetriever can access them
    # This is required because we only indexed leaf nodes in the vector store
    for parent_node in root_nodes:
        storage_context.docstore.add_documents([parent_node])
    
    # Persist index to disk
    os.makedirs(STORAGE_DIR, exist_ok=True)
    index.storage_context.persist(persist_dir=STORAGE_DIR)
    
    print(f"Index successfully built and persisted to {STORAGE_DIR}")
    
    return index, leaf_nodes, root_nodes


def create_auto_merging_query_engine(index, leaf_nodes, root_nodes):
    """
    Task 4: Recursive / Auto-Merging Retrieval Engine
    Construct query engine with custom parent context merging
    This manually retrieves parent nodes when child chunks match, ensuring reliability
    """
    # Get base retriever from index
    base_retriever = index.as_retriever(
        similarity_top_k=5  # Retrieve top 5 child chunks
    )
    
    # Create a mapping of parent node IDs to parent nodes
    parent_map = {node.node_id: node for node in root_nodes}
    
    # Custom retriever that merges parent context
    class ParentMergingRetriever:
        def __init__(self, base_retriever, parent_map, storage_context):
            self.base_retriever = base_retriever
            self.parent_map = parent_map
            self.storage_context = storage_context
        
        def retrieve(self, query_bundle):
            # Retrieve child nodes
            child_nodes = self.base_retriever.retrieve(query_bundle)
            
            # Merge parent context for each child node
            merged_nodes = []
            for child_node in child_nodes:
                # Check if child has parent relationship
                if hasattr(child_node, 'relationships') and child_node.relationships:
                    parent_rel = child_node.relationships.get('4')  # NodeRelationship.PARENT
                    if parent_rel:
                        parent_id = parent_rel.node_id
                        if parent_id in self.parent_map:
                            parent_node = self.parent_map[parent_id]
                            # Create a merged node with parent context
                            merged_text = f"[PARENT CONTEXT]\n{parent_node.text}\n\n[CHILD CHUNK]\n{child_node.text}"
                            # Create a new node with merged text
                            from llama_index.core import Document
                            merged_node = Document(text=merged_text)
                            merged_node.metadata = child_node.metadata
                            merged_nodes.append(merged_node)
                            print(f"Merged parent context for child node")
                        else:
                            # Parent not found in map, use child as-is
                            merged_nodes.append(child_node)
                    else:
                        # No parent relationship, use child as-is
                        merged_nodes.append(child_node)
                else:
                    # No relationships, use child as-is
                    merged_nodes.append(child_node)
            
            return merged_nodes
    
    # Create custom merging retriever
    merging_retriever = ParentMergingRetriever(base_retriever, parent_map, index.storage_context)
    
    # Create query engine with custom retriever
    # Uses configured gemini-3.5-flash LLM and gemini-embedding-2
    query_engine = RetrieverQueryEngine.from_args(
        retriever=merging_retriever,
        llm=Settings.llm,
        embed_model=Settings.embed_model
    )
    
    return query_engine


def main():
    """
    Task 5: Local Execution & Retrieval Proof-of-Concept
    Execute the pipeline with diagnostic query
    """
    print("=" * 80)
    print("Financial RAG Pipeline - Parent-Child Chunking & Auto-Merging Retrieval")
    print("=" * 80)
    
    # Build or load index
    file_path = "data/mock/financial_sample.txt"
    index, leaf_nodes, root_nodes = build_or_load_index(file_path)
    
    print("\n" + "=" * 80)
    print("Index Statistics:")
    print(f"  - Child nodes (stored in vector store): {len(leaf_nodes)}")
    print(f"  - Parent nodes (for context): {len(root_nodes)}")
    
    # Sample child node for inspection
    if leaf_nodes:
        print(f"\nSample child node length: {len(leaf_nodes[0].text)} characters")
        print(f"Sample child node text: {leaf_nodes[0].text[:200]}...")
    
    # Sample parent node for inspection
    if root_nodes:
        print(f"\nSample parent node length: {len(root_nodes[0].text)} characters")
        print(f"Sample parent node text: {root_nodes[0].text[:200]}...")
    
    # Create auto-merging query engine
    query_engine = create_auto_merging_query_engine(index, leaf_nodes, root_nodes)
    
    # Run diagnostic query
    print("\n" + "=" * 80)
    print("Running Diagnostic Query...")
    print("=" * 80)
    
    query = "What are OmniCorp's data privacy compliance risks?"
    print(f"\nQuery: {query}")
    
    # Execute query
    response = query_engine.query(query)
    
    print("\n" + "=" * 80)
    print("Query Response:")
    print("=" * 80)
    print(f"\n{response}")
    
    # Print retrieved context node information
    print("\n" + "=" * 80)
    print("Retrieval Details:")
    print("=" * 80)
    
    # Access the retrieved nodes from the response
    if hasattr(response, 'source_nodes'):
        print(f"\nNumber of retrieved nodes: {len(response.source_nodes)}")
        for i, node in enumerate(response.source_nodes):
            print(f"\nNode {i+1}:")
            print(f"  - Length: {len(node.text)} characters")
            print(f"  - Text preview: {node.text[:150]}...")
            if hasattr(node, 'metadata'):
                print(f"  - Metadata: {node.metadata}")
    
    print("\n" + "=" * 80)
    print("Pipeline execution completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
