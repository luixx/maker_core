"""Example using Azure OpenAI provider."""

import asyncio
import os
from src.maker_core import Maker
from src.maker_core.providers import AzureOpenAIProvider


async def main():
    """Run MAKER example with Azure OpenAI."""
    # Initialize Azure OpenAI provider
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    
    if not all([api_key, azure_endpoint, deployment_name]):
        print("Error: Set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_DEPLOYMENT")
        return
    
    provider = AzureOpenAIProvider(
        api_key=api_key,
        azure_endpoint=azure_endpoint,
        api_version="2024-10-21",
        deployment_name=deployment_name
    )
    
    # Create Maker instance
    maker = Maker(provider=provider)
    
    # Ask a question
    question = "How does machine learning differ from traditional programming?"
    print(f"\nQuestion: {question}\n")
    
    result = await maker.ask(question)
    
    print(f"\nAnswer: {result['answer']}")
    print(f"\nConfidence: {result['confidence'].value}\n")


if __name__ == "__main__":
    asyncio.run(main())
