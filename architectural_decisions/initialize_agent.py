from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import ConversationalRetrievalChain

# Use Ollama's local embedding model
embedding_model = OllamaEmbeddings(model="codellama")  # You can also try 'llama3' or other Ollama models

# Load design decisions from a file
with open(r"C:\repos\Hedgehog2.jl\architectural_decisions\design.md", "r") as f:
    design_text = f.read()

with open(r"C:\repos\Hedgehog2.jl\architectural_decisions\questions_answers.txt", "r") as f:
    answers = f.read()

with open(r"C:\repos\Hedgehog2.jl\src\Hedgehog2.jl", "r") as f:
    code = f.read()

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
design_docs = text_splitter.split_text(design_text)
code_docs = text_splitter.split_text(code)
answers_docs = text_splitter.split_text(answers)

# Store separately in ChromaDB
vectorstore = Chroma.from_texts(code_docs + design_docs + answers_docs, embedding=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ Design decisions stored in ChromaDB using Ollama embeddings!")

# Load Ollama model for answering
llm = ChatOllama(model="codellama")  # Or replace with your fine-tuned model

# Set up the conversational retrieval chain
qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)

# Example query
query = "Modify the Black-Scholes pricing strategy to support put options in addition to call options."
response = qa_chain.invoke({"question": query, "chat_history": []})
chat_history = [(query, response["answer"])]
print("You:", query)
print("AI:", response["answer"])

while True:
    user_input = input("You: ")

    # End conversation if user types 'end'
    if user_input.lower() == "end":
        print("👋 Chat session ended. Goodbye!")
        break

    # Generate response while maintaining chat history
    response = qa_chain.invoke({"question": user_input, "chat_history": chat_history})

    # Print AI response
    print("AI:", response["answer"])

    # Update chat history
    chat_history.append((user_input, response["answer"]))

print("🧠 AI Response:", response["answer"])
