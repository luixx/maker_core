"""Example using documents as context with MAKER."""

import asyncio
import os
from src.maker_core import Maker
from src.maker_core.providers import OpenAIProvider


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


async def basic_example():
    """Basic example: Ask questions about a document."""
    print("=" * 80)
    print("Example 1: Basic Document Question")
    print("=" * 80)
    print()
    
    # Set up provider (replace with your API key)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY environment variable to run this example")
        print("Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    maker = Maker(provider=provider)
    
    # Ask a simple question with document as context
    question = "What are the main applications of AI in healthcare?"
    print(f"Question: {question}\n")
    
    result = await maker.ask(
        question,
        options={"context": SAMPLE_DOCUMENT}
    )
    
    print(f"Answer: {result['answer']}\n")
    print(f"Confidence: {result['confidence'].value}")
    print(f"Decomposition used: {result['is_decomposed']}")
    print()


async def complex_question_example():
    """Example: Complex question requiring decomposition."""
    print("=" * 80)
    print("Example 2: Complex Question with Document Context")
    print("=" * 80)
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    maker = Maker(provider=provider)
    
    # Add event listeners to see the decomposition process
    maker.on("decomposed", lambda data: 
        print(f"✓ Question decomposed into {data['count']} sub-questions"))
    maker.on("votingComplete", lambda data: 
        print(f"✓ Sub-question answered in {data['rounds']} rounds"))
    
    # Complex question that will trigger decomposition
    question = (
        "Based on the document, what are the benefits and challenges "
        "of implementing AI in healthcare, and what does the future hold?"
    )
    print(f"Question: {question}\n")
    
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


async def multiple_documents_example():
    """Example: Using multiple documents as combined context."""
    print("=" * 80)
    print("Example 3: Multiple Documents")
    print("=" * 80)
    print()
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
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
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    maker = Maker(provider=provider)
    
    question = "What are the key considerations for implementing ethical AI?"
    print(f"Question: {question}\n")
    print("Using context from 2 documents...\n")
    
    result = await maker.ask(
        question,
        options={"context": combined_context}
    )
    
    print(f"Answer: {result['answer']}\n")


async def file_loading_example():
    """Example: Loading documents from files."""
    print("=" * 80)
    print("Example 4: Loading Documents from Files")
    print("=" * 80)
    print()
    
    # Create a sample file for demonstration
    sample_file = "sample_doc.txt"
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
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY environment variable")
        os.remove(sample_file)
        return
    
    # Load document from file
    with open(sample_file, "r") as f:
        document_content = f.read()
    
    print(f"Loaded document ({len(document_content)} characters)\n")
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    maker = Maker(provider=provider)
    
    question = "What are the top 3 Python best practices mentioned?"
    print(f"Question: {question}\n")
    
    result = await maker.ask(
        question,
        options={"context": document_content}
    )
    
    print(f"Answer: {result['answer']}\n")
    
    # Clean up
    os.remove(sample_file)
    print(f"Cleaned up: {sample_file}")


async def chunking_example():
    """Example: Handling long documents with chunking."""
    print("=" * 80)
    print("Example 5: Handling Long Documents (Chunking)")
    print("=" * 80)
    print()
    
    # Simulate a very long document
    long_document = SAMPLE_DOCUMENT * 3  # Repeat to make it longer
    
    print(f"Document length: {len(long_document)} characters")
    print(f"Approximate tokens: ~{len(long_document) // 4}\n")
    
    # Simple chunking strategy
    chunk_size = 2000  # characters
    chunks = [
        long_document[i:i+chunk_size] 
        for i in range(0, len(long_document), chunk_size)
    ]
    
    print(f"Split into {len(chunks)} chunks\n")
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  Set OPENAI_API_KEY environment variable")
        return
    
    provider = OpenAIProvider(api_key=api_key, model="gpt-4")
    maker = Maker(provider=provider)
    
    # Strategy 1: Use first chunk (often contains summary/intro)
    print("Strategy 1: Using first chunk only")
    question = "What is this document about?"
    
    result = await maker.ask(
        question,
        options={"context": chunks[0]}
    )
    
    print(f"Answer: {result['answer']}\n")
    
    # Strategy 2: Combine selected chunks (if you know which are relevant)
    print("\nStrategy 2: Combining multiple relevant chunks")
    relevant_chunks = chunks[:2]  # First two chunks
    combined = "\n\n---\n\n".join(relevant_chunks)
    
    result = await maker.ask(
        question,
        options={"context": combined}
    )
    
    print(f"Answer: {result['answer']}\n")


async def main():
    """Run all examples."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "DOCUMENT CONTEXT EXAMPLES" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    print()
    
    examples = [
        ("1", "Basic document question", basic_example),
        ("2", "Complex question with decomposition", complex_question_example),
        ("3", "Multiple documents", multiple_documents_example),
        ("4", "Loading from files", file_loading_example),
        ("5", "Handling long documents", chunking_example),
    ]
    
    print("Available examples:")
    for num, desc, _ in examples:
        print(f"  {num}. {desc}")
    print("  6. Run all examples")
    print()
    
    choice = input("Choose example (1-6, or press Enter for all): ").strip()
    print()
    
    if choice == "6" or choice == "":
        # Run all examples
        for _, _, example_func in examples:
            await example_func()
            print("\n")
    else:
        # Run specific example
        for num, _, example_func in examples:
            if choice == num:
                await example_func()
                break
        else:
            print("Invalid choice. Running example 1...")
            await basic_example()


if __name__ == "__main__":
    print("""
This example demonstrates how to use documents as context with MAKER.

The current version of MAKER already supports document context through
the 'context' parameter in the ask() method. No additional setup needed!

Key points:
✅ Pass document text via options={"context": document_text}
✅ Works with single or multiple documents (combine them)
✅ Context is used in classification, decomposition, voting, and synthesis
✅ For long documents, you may need to chunk them manually

Note: Make sure OPENAI_API_KEY is set in your environment.
    """)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nExamples interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. OPENAI_API_KEY is set")
        print("  2. You have an internet connection")
        print("  3. Your API key is valid")
