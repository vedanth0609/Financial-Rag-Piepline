"""
Strictly throttled, zero-cost-embedding evaluation harness using Ragas and LlamaIndex
Protects free-tier Gemini API key from HTTP 429 rate limits during both pipeline inference
and Ragas grading phases through complete sequential serialization and intentional delays.
"""
import json
import time
import sys
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Monkey patch to fix Ragas import issue with langchain-community
# Inject the vertexai module into sys.modules before Ragas imports it
from langchain_google_vertexai import ChatVertexAI
import types
vertexai_module = types.ModuleType('langchain_community.chat_models.vertexai')
vertexai_module.ChatVertexAI = ChatVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = vertexai_module

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.embeddings import LlamaIndexEmbeddingsWrapper
from datasets import Dataset

from config import init_llm_and_embeddings
from llama_index.core import Settings
from pipeline import build_or_load_index, create_auto_merging_query_engine


class RateThrottledLLM:
    """
    Custom rate-throttled LLM wrapper that injects mandatory 10-second delays
    into all core generation methods to prevent Gemini API quota exhaustion.
    """
    def __init__(self, base_llm):
        self.base_llm = base_llm
        self._last_call_time = 0
        
    def _enforce_throttle(self):
        """Enforce 10-second minimum delay between API calls."""
        current_time = time.time()
        time_since_last_call = current_time - self._last_call_time
        if time_since_last_call < 10:
            sleep_time = 10 - time_since_last_call
            time.sleep(sleep_time)
        self._last_call_time = time.time()
    
    def complete(self, prompt: str, **kwargs) -> str:
        """Throttled synchronous completion."""
        self._enforce_throttle()
        return self.base_llm.complete(prompt, **kwargs)
    
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Throttled synchronous chat."""
        self._enforce_throttle()
        return self.base_llm.chat(messages, **kwargs)
    
    async def acomplete(self, prompt: str, **kwargs) -> str:
        """Throttled asynchronous completion."""
        self._enforce_throttle()
        return await self.base_llm.acomplete(prompt, **kwargs)
    
    async def achat(self, messages: List[Dict], **kwargs) -> str:
        """Throttled asynchronous chat."""
        self._enforce_throttle()
        return await self.base_llm.achat(messages, **kwargs)
    
    def __getattr__(self, name):
        """Delegate any other attributes to the base LLM."""
        return getattr(self.base_llm, name)



def main():
    """
    Main evaluation harness with strict throttling to prevent rate limits
    """
    console = Console()
    
    console.print(Panel.fit(
        "[bold cyan]Financial RAG Pipeline - Throttled Evaluation Harness[/bold cyan]\n"
        "[yellow]Protecting free-tier Gemini API from HTTP 429 rate limits[/yellow]",
        title="RAGAS EVALUATION"
    ))
    
    # Initialize LLM and embeddings settings
    init_llm_and_embeddings()
    
    # Initialize Ragas metrics with throttled components
    console.print("\n[bold]Phase 0: Framework Initialization[/bold]")
    console.print("[green]✓[/green] Using LlamaIndex Gemini LLM (zero-cost local embeddings)")
    console.print("[green]✓[/green] Using local BAAI/bge-small-en-v1.5 embeddings")
    console.print("[green]✓[/green] Ragas metrics: Faithfulness and AnswerRelevancy")
    
    # Create rate-throttled LLM wrapper
    throttled_llm = RateThrottledLLM(Settings.llm)
    console.print("[green]✓[/green] Rate-throttled LLM wrapper initialized (10-second delay per call)")
    
    # Wrap local embeddings with Ragas LlamaIndexEmbeddingsWrapper
    wrapped_embeddings = LlamaIndexEmbeddingsWrapper(Settings.embed_model)
    console.print("[green]✓[/green] Local embeddings wrapped for Ragas (zero-cost vector math)")
    
    # Initialize Ragas metrics with throttled LLM
    faithfulness_metric = faithfulness
    answer_relevancy_metric = answer_relevancy
    console.print("[green]✓[/green] Ragas metrics initialized with throttled LLM")
    
    # Phase 1: Throttled Sequential Ingestion & Inference Loop
    console.print("\n[bold]Phase 1: Throttled Sequential Ingestion & Inference[/bold]")
    console.print("[yellow]Loading test questions from data/mock/test_questions.json[/yellow]")
    
    with open("data/mock/test_questions.json", "r") as f:
        test_data = json.load(f)
    
    qa_pairs = test_data["questions"]
    console.print(f"[green]✓[/green] Loaded {len(qa_pairs)} QA pairs")
    
    # Build or load index and create query engine
    console.print("\n[yellow]Building/loading index and creating query engine...[/yellow]")
    file_path = "data/mock/financial_sample.txt"
    index, leaf_nodes, root_nodes = build_or_load_index(file_path)
    query_engine = create_auto_merging_query_engine(index, leaf_nodes, root_nodes)
    console.print("[green]✓[/green] Query engine initialized with Parent-Child recursive retrieval")
    
    # Collect evaluation samples with strict throttling
    evaluation_samples = []
    
    for i, qa_pair in enumerate(qa_pairs):
        console.print(f"\n[bold cyan]Processing question {i+1}/{len(qa_pairs)}[/bold cyan]")
        console.print(f"Question: {qa_pair['question']}")
        
        # Query the engine
        response = query_engine.query(qa_pair['question'])
        generated_answer = str(response)
        
        # Extract retrieved context chunks
        retrieved_contexts = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                retrieved_contexts.append(node.text)
        
        console.print(f"[green]✓[/green] Generated answer ({len(generated_answer)} chars)")
        console.print(f"[green]✓[/green] Retrieved {len(retrieved_contexts)} context chunks")
        
        # Store sample for evaluation
        evaluation_samples.append({
            "question": qa_pair['question'],
            "answer": generated_answer,
            "contexts": retrieved_contexts,
            "ground_truth": qa_pair['answer']
        })
        
        # STRICT THROTTLING: 10-second delay after each query to stay under free-tier RPM
        if i < len(qa_pairs) - 1:
            console.print("[yellow]⏳ Throttling: 10-second delay to protect API rate limit...[/yellow]")
            time.sleep(10)
    
    console.print("\n[green]✓[/green] Phase 1 complete: All queries executed with throttling")
    
    # Phase 2: Row-by-Row Throttled Evaluation Loop
    console.print("\n[bold]Phase 2: Row-by-Row Throttled Evaluation[/bold]")
    console.print("[yellow]Evaluating samples sequentially to avoid parallel API calls[/yellow]")
    
    # Accumulate scores manually
    faithfulness_scores = []
    answer_relevancy_scores = []
    results_table_data = []
    
    for i, sample in enumerate(evaluation_samples):
        console.print(f"\n[bold cyan]Evaluating sample {i+1}/{len(evaluation_samples)}[/bold cyan]")
        console.print(f"Question: {sample['question'][:80]}...")
        
        # Create single-item dataset for this row
        single_row_dataset = Dataset.from_dict({
            "question": [sample["question"]],
            "answer": [sample["answer"]],
            "contexts": [sample["contexts"]]
        })
        
        # Compute Faithfulness score with throttling
        console.print("  [yellow]Computing Faithfulness score...[/yellow]")
        faithfulness_result = evaluate(
            dataset=single_row_dataset,
            metrics=[faithfulness_metric],
            llm=throttled_llm,
            embeddings=wrapped_embeddings
        )
        faithfulness_score = faithfulness_result['faithfulness'][0] if 'faithfulness' in faithfulness_result else 0.0
        faithfulness_scores.append(faithfulness_score)
        console.print(f"  [green]✓[/green] Faithfulness: {faithfulness_score:.4f}")
        
        # STRICT THROTTLING: 10-second delay after Faithfulness
        console.print("  [yellow]⏳ Throttling: 10-second delay after Faithfulness...[/yellow]")
        time.sleep(10)
        
        # Compute AnswerRelevancy score with throttling
        console.print("  [yellow]Computing AnswerRelevancy score...[/yellow]")
        answer_relevancy_result = evaluate(
            dataset=single_row_dataset,
            metrics=[answer_relevancy_metric],
            llm=throttled_llm,
            embeddings=wrapped_embeddings
        )
        answer_relevancy_score = answer_relevancy_result['answer_relevancy'][0] if 'answer_relevancy' in answer_relevancy_result else 0.0
        answer_relevancy_scores.append(answer_relevancy_score)
        console.print(f"  [green]✓[/green] AnswerRelevancy: {answer_relevancy_score:.4f}")
        
        # Store for table display
        results_table_data.append({
            "question": sample["question"],
            "faithfulness": faithfulness_score,
            "answer_relevancy": answer_relevancy_score
        })
        
        # STRICT THROTTLING: 10-second delay after AnswerRelevancy (except last row)
        if i < len(evaluation_samples) - 1:
            console.print("  [yellow]⏳ Throttling: 10-second delay before next sample...[/yellow]")
            time.sleep(10)
    
    console.print("\n[green]✓[/green] Phase 2 complete: All evaluations executed with throttling")
    
    # Calculate overall averages
    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0.0
    avg_answer_relevancy = sum(answer_relevancy_scores) / len(answer_relevancy_scores) if answer_relevancy_scores else 0.0
    
    # Professional UI Console Reporting
    console.print("\n[bold]Phase 3: Professional UI Console Reporting[/bold]")
    
    # Create results table
    table = Table(title="RAG Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Question", style="cyan", width=60)
    table.add_column("Faithfulness", style="green", justify="right")
    table.add_column("Answer Relevancy", style="green", justify="right")
    
    for row_data in results_table_data:
        table.add_row(
            row_data["question"],
            f"{row_data['faithfulness']:.4f}",
            f"{row_data['answer_relevancy']:.4f}"
        )
    
    console.print(table)
    
    # Print overall averages
    console.print("\n[bold]Overall Dataset Averages:[/bold]")
    console.print(f"  [cyan]Average Faithfulness:[/cyan]     {avg_faithfulness:.4f}")
    console.print(f"  [cyan]Average Answer Relevancy:[/cyan] {avg_answer_relevancy:.4f}")
    
    console.print("\n[green]✓[/green] Evaluation harness completed successfully!")
    console.print("[yellow]All API calls were throttled to protect free-tier rate limits[/yellow]")


if __name__ == "__main__":
    main()
