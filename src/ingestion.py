"""
Advanced Multi-Page Financial PDF Ingestion Pipeline
Handles complex corporate financial documents with hierarchical chunking,
layout metadata tracking, and persistent ChromaDB syncing.
"""
import os
import re
import chromadb
from typing import List, Dict, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Document,
    Settings,
    SimpleDirectoryReader
)
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    get_leaf_nodes,
    get_root_nodes
)
try:
    from llama_index.readers.file import PyMuPDFReader
except ImportError:
    PyMuPDFReader = None
from llama_index.vector_stores.chroma import ChromaVectorStore

from config import init_llm_and_embeddings

# Initialize global LLM and embeddings settings
init_llm_and_embeddings()

# Configuration constants
CHROMA_PERSIST_DIR = "data/production/chroma_db"
CHROMA_COLLECTION_NAME = "financial_rag_collection"
STORAGE_DIR = "data/production/storage"
DOCUMENTS_DIR = "data/production/documents"

console = Console()


class FinancialDocumentIngestion:
    """
    Robust ingestion pipeline for multi-page corporate financial PDF documents
    with layout metadata tracking and hierarchical chunking.
    """
    
    def __init__(self):
        self.console = Console()
        self.node_parser = self._setup_hierarchical_node_parser()
        self.vector_store, self.chroma_client = self._setup_chroma_vector_store()
        
    def _setup_hierarchical_node_parser(self):
        """
        Configure HierarchicalNodeParser for multi-granular chunking
        Parent chunks: ~1024 tokens (larger contextual blocks - sections/pages)
        Child chunks: ~256 tokens (focused sub-segments - sentences/rows)
        """
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[1024, 256],  # Parent and child chunk sizes
            chunk_overlap=64,  # Overlap between chunks
        )
        return node_parser
    
    def _setup_chroma_vector_store(self):
        """
        Initialize persistent ChromaDB client and wrap in LlamaIndex ChromaVectorStore
        without erasing existing collection schema.
        """
        # Ensure the directory exists
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        
        # Initialize persistent ChromaDB client
        chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        
        # Get or create collection (preserves existing data)
        chroma_collection = chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Wrap in LlamaIndex ChromaVectorStore
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        return vector_store, chroma_client
    
    def _detect_financial_tables(self, text: str) -> bool:
        """
        Detect if text contains structured financial tables based on patterns.
        """
        # Pattern indicators for financial tables
        table_indicators = [
            r'\|\s*\$[\d,]+\.?\d*\s*\|',  # Dollar amounts in table format
            r'\|\s*[\d,]+\s*\|\s*[\d,]+\s*\|',  # Multiple numbers in table format
            r'\bRevenue\b.*\b\d+\b',  # Revenue with numbers
            r'\bNet Income\b.*\b\d+\b',  # Net Income with numbers
            r'\bQ[1-4]\s*\d{4}\b',  # Quarterly references
            r'\|\s*%+\s*\|',  # Percentage in table format
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _extract_layout_metadata(self, file_path: str, page_number: int) -> Dict[str, Any]:
        """
        Extract layout metadata for each document block.
        """
        file_name = Path(file_path).name
        return {
            "file_name": file_name,
            "page_number": page_number,
            "file_path": str(file_path),
            "document_type": "financial_pdf"
        }
    
    def _load_pdf_documents(self, directory: str) -> List[Document]:
        """
        Load PDF and text documents from directory with layout metadata.
        Supports both PDF (PyMuPDF) and text files for testing.
        """
        self.console.print(f"\n[bold cyan]Loading documents from:[/bold cyan] {directory}")
        
        documents = []
        
        # Load PDF files using PyMuPDF reader
        pdf_files = list(Path(directory).glob("*.pdf"))
        if pdf_files:
            if PyMuPDFReader is None:
                self.console.print("[yellow]PyMuPDFReader not available, skipping PDF files[/yellow]")
            else:
                self.console.print(f"[green]Found {len(pdf_files)} PDF file(s)[/green]")
                reader = PyMuPDFReader()
                
                for pdf_file in pdf_files:
                    self.console.print(f"  [yellow]Processing PDF:[/yellow] {pdf_file.name}")
                    
                    try:
                        # Load PDF with page-by-page extraction
                        pdf_docs = reader.load_data(file_path=str(pdf_file))
                        
                        # Add layout metadata to each document
                        for i, doc in enumerate(pdf_docs):
                            metadata = self._extract_layout_metadata(str(pdf_file), i + 1)
                            doc.metadata.update(metadata)
                            documents.append(doc)
                        
                        self.console.print(f"    [green]✓[/green] Extracted {len(pdf_docs)} pages")
                        
                    except Exception as e:
                        self.console.print(f"    [red]✗[/red] Error loading {pdf_file.name}: {str(e)}")
                        continue
        
        # Load text files for testing
        text_files = list(Path(directory).glob("*.txt"))
        if text_files:
            self.console.print(f"[green]Found {len(text_files)} text file(s)[/green]")
            
            for text_file in text_files:
                self.console.print(f"  [yellow]Processing text:[/yellow] {text_file.name}")
                
                try:
                    # Load text file
                    with open(text_file, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    # Create document with layout metadata
                    doc = Document(text=text_content)
                    metadata = self._extract_layout_metadata(str(text_file), 1)
                    doc.metadata.update(metadata)
                    documents.append(doc)
                    
                    self.console.print(f"    [green]✓[/green] Loaded text file")
                    
                except Exception as e:
                    self.console.print(f"    [red]✗[/red] Error loading {text_file.name}: {str(e)}")
                    continue
        
        if not documents:
            self.console.print("[yellow]No PDF or text files found in directory[/yellow]")
            return []
        
        self.console.print(f"[green]✓[/green] Total documents loaded: {len(documents)}")
        return documents
    
    def _classify_and_tag_documents(self, documents: List[Document]) -> List[Document]:
        """
        Classify documents into dense paragraphs and structured financial tables.
        """
        self.console.print("\n[bold cyan]Classifying document content types[/bold cyan]")
        
        table_docs = []
        paragraph_docs = []
        
        for doc in documents:
            if self._detect_financial_tables(doc.text):
                doc.metadata["content_type"] = "financial_table"
                table_docs.append(doc)
            else:
                doc.metadata["content_type"] = "dense_paragraph"
                paragraph_docs.append(doc)
        
        self.console.print(f"  [green]✓[/green] Financial tables: {len(table_docs)}")
        self.console.print(f"  [green]✓[/green] Dense paragraphs: {len(paragraph_docs)}")
        
        return documents
    
    def _create_hierarchical_nodes(self, documents: List[Document]) -> tuple:
        """
        Create hierarchical nodes with parent-child relationships.
        Parent chunks represent broader sections/pages, child chunks are granular.
        """
        self.console.print("\n[bold cyan]Creating hierarchical node structure[/bold cyan]")
        
        # Extract hierarchical nodes (parent-child relationships)
        nodes = self.node_parser.get_nodes_from_documents(documents)
        
        # Separate leaf nodes (child chunks) and root nodes (parent chunks)
        leaf_nodes = get_leaf_nodes(nodes)
        root_nodes = get_root_nodes(nodes)
        
        self.console.print(f"  [green]✓[/green] Parent nodes (broader sections): {len(root_nodes)}")
        self.console.print(f"  [green]✓[/green] Child nodes (granular segments): {len(leaf_nodes)}")
        
        return leaf_nodes, root_nodes
    
    def _sync_to_chromadb(self, leaf_nodes: List, root_nodes: List) -> VectorStoreIndex:
        """
        Sync hierarchical nodes to persistent ChromaDB without erasing existing schema.
        """
        self.console.print("\n[bold cyan]Syncing to persistent ChromaDB[/bold cyan]")
        
        # Create storage context with existing vector store
        storage_context = StorageContext.from_defaults(
            vector_store=self.vector_store
        )
        
        # Build index with only leaf nodes (child chunks for vector retrieval)
        index = VectorStoreIndex(
            nodes=leaf_nodes,
            storage_context=storage_context
        )
        
        # Manually add parent nodes to docstore for AutoMergingRetriever access
        for parent_node in root_nodes:
            storage_context.docstore.add_documents([parent_node])
        
        # Persist index to disk
        os.makedirs(STORAGE_DIR, exist_ok=True)
        index.storage_context.persist(persist_dir=STORAGE_DIR)
        
        self.console.print(f"  [green]✓[/green] Index synced to ChromaDB")
        self.console.print(f"  [green]✓[/green] Storage persisted to: {STORAGE_DIR}")
        
        return index
    
    def ingest_directory(self, directory: str = DOCUMENTS_DIR) -> Dict[str, Any]:
        """
        Complete ingestion pipeline for directory of PDF documents.
        """
        self.console.print(Panel.fit(
            "[bold cyan]Financial PDF Ingestion Pipeline[/bold cyan]\n"
            "[yellow]Multi-page document processing with hierarchical chunking[/yellow]",
            title="INGESTION"
        ))
        
        # Step 1: Load PDF documents with layout metadata
        documents = self._load_pdf_documents(directory)
        
        if not documents:
            self.console.print("[red]No documents to ingest[/red]")
            return {"status": "failed", "reason": "no_documents"}
        
        # Step 2: Classify and tag content types
        documents = self._classify_and_tag_documents(documents)
        
        # Step 3: Create hierarchical nodes
        leaf_nodes, root_nodes = self._create_hierarchical_nodes(documents)
        
        # Step 4: Sync to ChromaDB
        index = self._sync_to_chromadb(leaf_nodes, root_nodes)
        
        # Step 5: Generate ingestion report
        report = self._generate_ingestion_report(documents, leaf_nodes, root_nodes)
        
        self.console.print("\n[bold green]✓ Ingestion completed successfully![/bold green]")
        
        return {
            "status": "success",
            "documents_processed": len(documents),
            "leaf_nodes": len(leaf_nodes),
            "root_nodes": len(root_nodes),
            "index": index,
            "report": report
        }
    
    def _generate_ingestion_report(self, documents: List, leaf_nodes: List, root_nodes: List) -> Dict[str, Any]:
        """
        Generate detailed ingestion report with statistics.
        """
        # Count content types
        table_count = sum(1 for doc in documents if doc.metadata.get("content_type") == "financial_table")
        paragraph_count = sum(1 for doc in documents if doc.metadata.get("content_type") == "dense_paragraph")
        
        # Count unique files
        unique_files = set(doc.metadata.get("file_name") for doc in documents)
        
        report = {
            "total_documents": len(documents),
            "unique_files": len(unique_files),
            "financial_tables": table_count,
            "dense_paragraphs": paragraph_count,
            "parent_nodes": len(root_nodes),
            "child_nodes": len(leaf_nodes),
            "files_processed": list(unique_files)
        }
        
        # Display report table
        self.console.print("\n[bold]Ingestion Report:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green", justify="right")
        
        table.add_row("Total Documents", str(len(documents)))
        table.add_row("Unique Files", str(len(unique_files)))
        table.add_row("Financial Tables", str(table_count))
        table.add_row("Dense Paragraphs", str(paragraph_count))
        table.add_row("Parent Nodes", str(len(root_nodes)))
        table.add_row("Child Nodes", str(len(leaf_nodes)))
        
        self.console.print(table)
        
        return report


def main():
    """
    Execute the ingestion pipeline for production documents.
    """
    ingestion = FinancialDocumentIngestion()
    result = ingestion.ingest_directory()
    
    if result["status"] == "success":
        console.print("\n[bold green]Pipeline execution completed successfully![/bold green]")
    else:
        console.print(f"\n[bold red]Pipeline execution failed:[/bold red] {result.get('reason', 'unknown')}")


if __name__ == "__main__":
    main()
