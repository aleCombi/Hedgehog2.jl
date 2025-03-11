import os
import yaml
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.schema import Document
import time

class ADRAgent:
    def __init__(self, adr_directory: str, retrieval_model_name: str = "mistral", embedding_model_name: str = "mistral"):
        """
        Initializes the ADR Agent by loading all ADR files from the given directory.
        """
        self.adr_directory = adr_directory
        self.retrieval_model_name = retrieval_model_name
        self.embedding_model_name = embedding_model_name
        self.adr_vectorstore = self.index_adrs()
        self.llm_retrieval = ChatOllama(model=self.retrieval_model_name)  # AI model for processing ADRs
        self.design_qa = RetrievalQA.from_chain_type(self.llm_retrieval, retriever=self.adr_vectorstore.as_retriever())
    
    def load_adrs(self) -> List[Document]:
        """
        Loads all ADRs from the specified directory, parses them, and converts them into Documents.
        """
        documents = []
        for file in os.listdir(self.adr_directory):
            if file.endswith(".yaml") and file.startswith("adr-"):
                file_path = os.path.join(self.adr_directory, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    adr_data = yaml.safe_load(f)
                
                adr_text = f"{adr_data['title']}\n\n{adr_data['context']}\n\n{adr_data['decision']}"
                metadata = {
                    "adr_id": adr_data["adr_id"],
                    "title": adr_data["title"],
                    "filename": file_path
                }
                documents.append(Document(page_content=adr_text, metadata=metadata))
        return documents
    
    def index_adrs(self):
        """
        Indexes ADRs in ChromaDB for semantic search.
        """
        adr_documents = self.load_adrs()
        embedding_model = OllamaEmbeddings(model=self.embedding_model_name)  # Use specified embedding model
        return Chroma.from_documents(adr_documents, embedding=embedding_model)
    
    def retrieve_relevant_adrs(self, feature_request: str, k=3) -> List[Document]:
        """
        Fetches the most relevant ADRs using vector similarity search.
        """
        retriever = self.adr_vectorstore.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(feature_request)
        return results["documents"]
    
    def generate_high_level_steps(self, feature_request: str) -> List[str]:
        """
        Uses AI-powered RetrievalQA to generate structured high-level implementation steps.
        """
        prompt = f"""
        You are an AI agent helping a developer implement a feature request in Hedgehog2.jl, a Julia derivatives pricing library.
        Your task is to retrieve relevant design decisions (ADRs), referenced by name, and provide structured implementation steps.

        Feature Request: {feature_request}
        
        Respond with clear, step-by-step guidance based on existing design decisions.
        The answer must be a numbered list.
        """
        response = self.design_qa.invoke(prompt)
        return response["result"]


# Main execution
if __name__ == "__main__":
    adr_agent = ADRAgent("./docs/adr", retrieval_model_name="codellama:34b", embedding_model_name="mistral")  # Replace with your ADR directory path and preferred model
    feature_request1 = "Implement put options pricing using Black-Scholes analytical formulas"
    feature_request2 = "Implement options pricing using Cox Ross Rubinstein Binomial Tree method."
    feature_request3 = "Implement options pricing using Heston Model."
    feature_request4 = "Implement interest rate forward pricing using discount curves."

    start_time = time.time()
    steps = adr_agent.generate_high_level_steps(feature_request1)
    end_time = time.time()

    print(steps)
    print(f"Elapsed time: {(end_time - start_time):.4f} seconds")
