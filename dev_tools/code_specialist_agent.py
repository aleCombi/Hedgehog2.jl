import os
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.schema import Document
import julia_chunks

class CodingExpertAgent:
    def __init__(self, julia_directory: str, retrieval_model_name: str = "codellama:34b", embedding_model_name: str = "mistral"):
        """
        Initializes the Coding Expert Agent by loading and indexing Julia code from the given directory.
        """
        self.julia_directory = julia_directory
        self.retrieval_model_name = retrieval_model_name
        self.embedding_model_name = embedding_model_name
        self.code_vectorstore = self.index_julia_code()
        self.llm_retrieval = ChatOllama(model=self.retrieval_model_name)  # AI model for answering code questions
        self.code_qa = RetrievalQA.from_chain_type(self.llm_retrieval, retriever=self.code_vectorstore.as_retriever())
    
    def load_julia_code(self) -> List[Document]:
        """
        Loads all Julia functions and structs with docstrings from the specified directory, parses them,
        and converts them into LangChain Documents for indexing.
        """
        chunks = julia_chunks.chunk_by_docstring(self.julia_directory)
        documents = []
        
        for chunk in chunks:
            content = f"""
            Type: {chunk['type']}
            Name: {chunk['name']}
            
            Docstring:
            {chunk['docstring']}
            
            Definition:
            {chunk['definition_code']}
            """
            metadata = {
                "filename": chunk["metadata"]["filename"],
                "start_line": chunk["metadata"]["start_line"],
                "end_line": chunk["metadata"]["end_line"],
                "name": chunk["name"]
            }
            documents.append(Document(page_content=content, metadata=metadata))
        
        return documents
    
    def index_julia_code(self):
        """
        Indexes Julia code chunks in ChromaDB for semantic search.
        """
        julia_documents = self.load_julia_code()
        embedding_model = OllamaEmbeddings(model=self.embedding_model_name)
        return Chroma.from_documents(julia_documents, embedding=embedding_model)
    
    def retrieve_relevant_code(self, query: str, k=3) -> List[Document]:
        """
        Fetches the most relevant code snippets using vector similarity search.
        """
        retriever = self.code_vectorstore.as_retriever(search_kwargs={"k": k})
        results = retriever.invoke(query)
        return results["documents"]
    
    def answer_code_question(self, question: str) -> str:
        """
        Uses AI-powered RetrievalQA to answer a code-related question.
        """
        prompt = f"""
        You are an AI agent assisting with Hedgehog2.jl, a Julia derivatives pricing library.
        Use relevant code snippets and docstrings to provide accurate and well-structured answers with code from the library.
        
        Question: {question}
        """
        response = self.code_qa.invoke(prompt)
        return response["result"]


# Example execution
if __name__ == "__main__":
    coding_agent = CodingExpertAgent("./src", retrieval_model_name="codellama:34b", embedding_model_name="mistral")
    
    question1 = "How can I price a call option with spot price 7, strike 7, rate 0.4, volatility 0.4 using Hedgehog2?"
    question2 = "What are the fields of the DiscountCurve struct?"
    question3 = "How is Monte Carlo simulation implemented in Hedgehog2.jl?"
    
    print(coding_agent.answer_code_question(question1))