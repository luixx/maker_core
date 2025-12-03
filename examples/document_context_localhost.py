"""Example using documents as context with self-hosted/localhost LLM."""

import asyncio
import os
from src.maker_core import Maker
from src.maker_core.providers import CustomLLMProvider


# Sample document content for demonstration
SAMPLE_DOCUMENT = """
ARTIFICIAL INTELLIGENCE IN HEALTHCARE: A COMPREHENSIVE REVIEW

Abstract:
Artificial Intelligence (AI) is transforming healthcare through improved diagnostics,
personalized treatment plans, and operational efficiency. This review examines current
applications and future prospects.

1. Introduction
Healthcare systems worldwide face challenges including rising costs, physician burnout,
and the need for more accurate diagnoses. AI technologies, particularly machine learning
and deep learning, offer promising solutions to these challenges.

2. Current Applications

2.1 Medical Imaging
AI algorithms have achieved remarkable accuracy in analyzing medical images such as
X-rays, MRIs, and CT scans. Deep learning models can detect anomalies that human
radiologists might miss, including early-stage cancers and subtle fractures.

2.2 Drug Discovery
AI accelerates drug discovery by analyzing molecular structures and predicting drug
efficacy. This reduces development time from years to months and significantly lowers
costs. Companies are using AI to identify potential drug candidates for diseases like
Alzheimer's and cancer.

2.3 Personalized Medicine
By analyzing patient data including genetics, lifestyle, and medical history, AI systems
can recommend personalized treatment plans. This approach improves outcomes and
reduces adverse reactions to medications.

2.4 Administrative Tasks
AI-powered systems handle scheduling, billing, and documentation, reducing
administrative burden on healthcare providers and allowing more time for patient care.

3. Challenges

3.1 Data Privacy
Healthcare data is highly sensitive. AI systems must comply with regulations like HIPAA
while maintaining data security and patient privacy.

3.2 Algorithm Bias
AI models trained on non-diverse datasets may produce biased results, potentially
leading to health disparities among different demographic groups.

3.3 Integration with Existing Systems
Implementing AI in established healthcare infrastructure requires significant investment
and workflow changes.

4. Future Directions
The future of AI in healthcare includes real-time health monitoring through wearables,
AI-assisted surgery, and predictive analytics for disease prevention. As technology
advances and regulatory frameworks evolve, AI will become increasingly integrated
into standard healthcare practice.

5. Conclusion
AI has tremendous potential to improve healthcare outcomes, reduce costs, and enhance
patient experiences. However, careful attention must be paid to ethical considerations,
data privacy, and equitable access to ensure AI benefits all patients.

Authors: Dr. Jane Smith, Dr. John Doe
Institution: Medical AI Research Institute
Published: 2024
"""


async def basic_example_ollama():
    """Basic example: Ask questions about a document using Ollama."""
    print("=" * 80)
    print("Example 1: Basic Document Question (Ollama)")
    print("=" * 80)
    print()
    print("Using Ollama on localhost:11434")
    print("Make sure Ollama is running: ollama serve")
    print("And a model is available: ollama pull llama2")
    print()
    
    # Configure Ollama provider
    provider = CustomLLMProvider(
        base_url="http://localhost:11434/v1",
        model="llama2",
        timeout=120  # Self-hosted models may need more time
    )
    
    maker = Maker(provider=provider)
    
    # Ask a simple question with document as context
    question = "What are the main applications of AI in healthcare?"
    print(f"Question: {question}\n")
    
    try:
        result = await maker.ask(
            question,
            options={"context": SAMPLE_DOCUMENT}
        )
        
        print(f"Answer: {result['answer']}\n")
        print(f"Confidence: {result['confidence'].value}")
        print(f"Decomposition used: {result['decomposition_used']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("  1. Is Ollama running? Run: ollama serve")
        print("  2. Is the model downloaded? Run: ollama pull llama2")
        print("  3. Check if port 11434 is accessible")
        print()


async def basic_example_vllm():
    """Basic example using vLLM server."""
    print("=" * 80)
    print("Example 2: Basic Document Question (vLLM)")
    print("=" * 80)
    print()
    print("Using vLLM on localhost:8000")
    print("Start vLLM with:")
    print("  python -m vllm.entrypoints.openai.api_server \\")
    print("    --model meta-llama/Llama-2-7b-chat-hf \\")
    print("    --port 8000")
    print()
    
    # Configure vLLM provider
    provider = CustomLLMProvider(
        base_url="http://localhost:8000/v1",
        model="meta-llama/Llama-2-7b-chat-hf",
        timeout=120
    )
    
    maker = Maker(provider=provider)
    
    question = "What are the main challenges of AI in healthcare?"
    print(f"Question: {question}\n")
    
    try:
        result = await maker.ask(
            question,
            options={"context": SAMPLE_DOCUMENT}
        )
        
        print(f"Answer: {result['answer']}\n")
        print(f"Confidence: {result['confidence'].value}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure vLLM server is running on port 8000")
        print()


async def complex_question_example():
    """Example: Complex question requiring decomposition with localhost LLM."""
    print("=" * 80)
    print("Example 3: Complex Question with Decomposition")
    print("=" * 80)
    print()
    
    # Using Ollama (most common for local development)
    provider = CustomLLMProvider(
        base_url="http://localhost:11434/v1",
        model="llama2",
        timeout=180  # More time for complex questions
    )
    
    maker = Maker(provider=provider)
    
    # Add event listeners to see the process
    maker.on("decomposed", lambda data: 
        print(f"✓ Question decomposed into {data['count']} sub-questions"))
    maker.on("votingStart", lambda data: 
        print(f"→ Voting on: {data['question'][:60]}..."))
    maker.on("votingComplete", lambda data: 
        print(f"✓ Consensus reached in {data['rounds']} rounds"))
    
    # Complex question that will trigger decomposition
    question = (
        "Based on the document, what are the benefits and challenges "
        "of implementing AI in healthcare, and what does the future hold?"
    )
    print(f"Question: {question}\n")
    
    try:
        result = await maker.ask(
            question,
            options={"context": SAMPLE_DOCUMENT}
        )
        
        print(f"\n{'='*80}")
        print("Final Answer:")
        print(result['answer'])
        print(f"\nConfidence: {result['confidence'].value}")
        print(f"Sub-questions answered: {len(result['sub_results'])}")
        
        if result['sub_results']:
            print("\nSub-questions that were answered:")
            for i, sub in enumerate(result['sub_results'], 1):
                print(f"  {i}. {sub['question']}")
        print()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure Ollama is running with llama2 model")
        print()


async def multiple_documents_example():
    """Example: Using multiple documents with localhost LLM."""
    print("=" * 80)
    print("Example 4: Multiple Documents")
    print("=" * 80)
    print()
    
    # Simulate multiple documents
    doc1 = """
    DOCUMENT 1: AI Ethics Guidelines
    
    1. Transparency: AI systems must be explainable
    2. Fairness: Avoid bias and discrimination
    3. Privacy: Protect user data
    4. Accountability: Clear responsibility for AI decisions
    """
    
    doc2 = """
    DOCUMENT 2: Implementation Best Practices
    
    1. Start with pilot projects
    2. Train staff adequately
    3. Monitor performance continuously
    4. Gather user feedback
    5. Iterate and improve
    """
    
    # Combine documents
    combined_context = f"{doc1}\n\n{doc2}"
    
    provider = CustomLLMProvider(
        base_url="http://localhost:11434/v1",
        model="llama2"
    )
    
    maker = Maker(provider=provider)
    
    question = "What are the key considerations for implementing ethical AI?"
    print(f"Question: {question}\n")
    print("Using context from 2 documents...\n")
    
    try:
        result = await maker.ask(
            question,
            options={"context": combined_context}
        )
        
        print(f"Answer: {result['answer']}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")


async def file_loading_example():
    """Example: Loading documents from files with localhost LLM."""
    print("=" * 80)
    print("Example 5: Loading Documents from Files")
    print("=" * 80)
    print()
    
    # Create a sample file for demonstration
    sample_file = "sample_doc_localhost.txt"
    with open(sample_file, "w") as f:
        f.write("""
Python Best Practices for 2024

1. Use Type Hints
   - Improves code clarity
   - Enables better IDE support
   - Catches errors early

2. Write Tests
   - Unit tests for functions
   - Integration tests for workflows
   - Maintain high coverage

3. Follow PEP 8
   - Consistent code style
   - Easy to read and maintain
   - Use tools like Black

4. Document Your Code
   - Docstrings for all public functions
   - README for project overview
   - Comments for complex logic

5. Use Virtual Environments
   - Isolate project dependencies
   - Reproducible builds
   - Avoid version conflicts
        """)
    
    print(f"Created sample file: {sample_file}")
    
    # Load document from file
    with open(sample_file, "r") as f:
        document_content = f.read()
    
    print(f"Loaded document ({len(document_content)} characters)\n")
    
    provider = CustomLLMProvider(
        base_url="http://localhost:11434/v1",
        model="llama2"
    )
    
    maker = Maker(provider=provider)
    
    question = "What are the top 3 Python best practices mentioned?"
    print(f"Question: {question}\n")
    
    try:
        result = await maker.ask(
            question,
            options={"context": document_content}
        )
        
        print(f"Answer: {result['answer']}\n")
    except Exception as e:
        print(f"❌ Error: {e}\n")
    finally:
        # Clean up
        os.remove(sample_file)
        print(f"Cleaned up: {sample_file}")


async def lm_studio_example():
    """Example: Using LM Studio."""
    print("=" * 80)
    print("Example 6: Using LM Studio")
    print("=" * 80)
    print()
    print("LM Studio provides a GUI for running local LLMs")
    print("Default URL: http://localhost:1234/v1")
    print()
    print("Setup:")
    print("  1. Download and open LM Studio")
    print("  2. Load a model (e.g., Mistral, Llama)")
    print("  3. Click 'Start Server' in the Local Server tab")
    print()
    
    provider = CustomLLMProvider(
        base_url="http://localhost:1234/v1",
        model="local-model",  # LM Studio uses generic model name
        timeout=120
    )
    
    maker = Maker(provider=provider)
    
    question = "What is the main topic of this document?"
    print(f"Question: {question}\n")
    
    try:
        result = await maker.ask(
            question,
            options={"context": SAMPLE_DOCUMENT[:1000]}  # Use truncated version
        )
        
        print(f"Answer: {result['answer']}\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure LM Studio server is running on port 1234")
        print()


async def main():
    """Run examples."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*15 + "DOCUMENT CONTEXT - LOCALHOST LLM" + " "*30 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    print("This example shows how to use documents with self-hosted LLMs.")
    print()
    
    examples = [
        ("1", "Basic question (Ollama)", basic_example_ollama),
        ("2", "Basic question (vLLM)", basic_example_vllm),
        ("3", "Complex question with decomposition", complex_question_example),
        ("4", "Multiple documents", multiple_documents_example),
        ("5", "Loading from files", file_loading_example),
        ("6", "Using LM Studio", lm_studio_example),
    ]
    
    print("Available examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    print("  7. Run all examples")
    print()
    
    # Configuration help
    print("📝 Quick Configuration Guide:")
    print()
    print("Ollama (easiest):")
    print("  • Install: https://ollama.ai/")
    print("  • Run: ollama serve")
    print("  • Pull model: ollama pull llama2")
    print("  • URL: http://localhost:11434/v1")
    print()
    print("vLLM:")
    print("  • Install: pip install vllm")
    print("  • Run: python -m vllm.entrypoints.openai.api_server --model <model>")
    print("  • URL: http://localhost:8000/v1")
    print()
    print("LM Studio:")
    print("  • Download: https://lmstudio.ai/")
    print("  • Start server from GUI")
    print("  • URL: http://localhost:1234/v1")
    print()
    
    choice = input("Choose example (1-7, or press Enter for Ollama example): ").strip()
    print()
    
    if choice == "7":
        # Run all examples
        for _, _, example_func in examples:
            await example_func()
            print("\n")
    elif choice == "":
        # Default to Ollama example
        await basic_example_ollama()
    else:
        # Run specific example
        for num, _, example_func in examples:
            if choice == num:
                await example_func()
                break
        else:
            print("Invalid choice. Running Ollama example...")
            await basic_example_ollama()


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║             MAKER with Documents + Self-Hosted LLMs                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

This example demonstrates using MAKER with:
  ✅ Self-hosted/localhost LLMs
  ✅ Document context
  ✅ No API keys required
  ✅ Full privacy - everything runs locally

Supported platforms:
  • Ollama (easiest, recommended)
  • vLLM (fastest inference)
  • LM Studio (GUI-based)
  • Text Generation Inference
  • LocalAI

The context feature works exactly the same with self-hosted LLMs!
Just pass your document via options={"context": document_text}

IMPORTANT: Make sure your LLM server is running before running examples.
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nCommon issues:")
        print("  1. LLM server not running")
        print("  2. Wrong port number")
        print("  3. Model not loaded")
        print("  4. Timeout (increase timeout parameter)")
