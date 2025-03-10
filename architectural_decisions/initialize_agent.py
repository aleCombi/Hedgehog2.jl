from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# Load and prepare text documents
def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

design_text = load_text_file(r"C:\repos\Hedgehog2.jl\architectural_decisions\design.md")
answers = load_text_file(r"C:\repos\Hedgehog2.jl\architectural_decisions\questions_answers.txt")
code = load_text_file(r"C:\repos\Hedgehog2.jl\src\Hedgehog2.jl")

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
design_docs = text_splitter.create_documents([design_text])
answers_docs = text_splitter.create_documents([answers])
code_docs = text_splitter.create_documents([code])

# Use Ollama embeddings
embedding_model = OllamaEmbeddings(model="codellama")

# Create separate vector stores
design_vectorstore = Chroma.from_documents(design_docs, embedding=embedding_model)
answers_vectorstore = Chroma.from_documents(answers_docs, embedding=embedding_model)
code_vectorstore = Chroma.from_documents(code_docs, embedding=embedding_model)

# Create separate retrievers
design_retriever = design_vectorstore.as_retriever(search_kwargs={"k": 3})
answers_retriever = answers_vectorstore.as_retriever(search_kwargs={"k": 3})
code_retriever = code_vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ Design, answers, and code indexed in ChromaDB!")

# Load LLM for answering
llm_retrieval = ChatOllama(model="mistral")  # Fast model for retrieval Q&A
llm_coding = ChatOllama(model="codellama:34b")  # Strong model for coding

# Agents
design_qa = RetrievalQA.from_chain_type(llm_retrieval, retriever=design_retriever)
answers_qa = RetrievalQA.from_chain_type(llm_retrieval, retriever=answers_retriever)
code_qa = RetrievalQA.from_chain_type(llm_retrieval, retriever=code_retriever)

# Interactive session
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "end":
        print("👋 Chat session ended.")
        break

    print("retrieving information to help the coding agent")
    # Retrieve relevant information
    design_response = design_qa.invoke(user_input)
    answer_response = answers_qa.invoke(user_input)
    code_response = code_qa.invoke(user_input)

    # Generate improved response using all retrieved knowledge
    final_prompt = f"""
    User Query: {user_input}
    
    Relevant Design Decisions:
    {design_response}
    
    Past Answers:
    {answer_response}
    
    Relevant Code Snippets:
    {code_response}
    
    Based on this, provide the best possible response.
    """
    print("waiting for the coding agent answer")
    final_response = llm_coding.invoke(final_prompt)

    print("\n🧠 AI Response:", final_response.content)
