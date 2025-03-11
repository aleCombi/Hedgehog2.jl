import os
import re

DOCSTRING_DEFINITION_PATTERN = re.compile(
    r'"""(.*?)"""'                       # 1) Capture docstring content
    r'\s*\n+'                             #    Possible newlines/spaces
    r'(function\s+.*?end'                # 2) function ... end
    r'|(?:mutable\s+)?struct\s+.*?end'   #    struct or mutable struct ... end
    r'|abstract\s+type\s+.*?end)',       #    abstract type ... end
    re.DOTALL
)

def find_docstring_definitions(code):
    """
    Finds pairs of (docstring_text, definition_code) using a single regex.
    Yields tuples: (docstring_text, definition_code, start_offset, end_offset).
    """
    matches = DOCSTRING_DEFINITION_PATTERN.finditer(code)
    for match in matches:
        docstring_text = match.group(1).strip()
        definition_code = match.group(2).strip()

        start_offset = match.start()
        end_offset = match.end()

        yield docstring_text, definition_code, start_offset, end_offset

def compute_line_numbers(code, start_offset, end_offset):
    """
    Given the entire file content and the character offsets of a match,
    compute the 1-based start/end line numbers.
    """
    start_line = code[:start_offset].count('\n') + 1
    end_line = code[:end_offset].count('\n') + 1
    return start_line, end_line

def parse_definition_type_and_name(definition_code):
    """
    Inspects the raw definition_code to determine:
      - def_type: one of 'function', 'struct', 'abstract' (or 'unknown')
      - def_name: extracted name, or 'unknown'
    """
    # Defaults
    def_type = "unknown"
    def_name = "unknown"

    # function ...
    if definition_code.startswith("function"):
        def_type = "function"
        # Attempt to parse the function name
        func_name_match = re.search(r'^function\s+([a-zA-Z0-9_!?\']+)', definition_code)
        if func_name_match:
            def_name = func_name_match.group(1)

    # abstract type ...
    elif definition_code.startswith("abstract type"):
        def_type = "abstract type"
        abs_name_match = re.search(r'^abstract\s+type\s+([a-zA-Z0-9_!?\']+)', definition_code)
        if abs_name_match:
            def_name = abs_name_match.group(1)

    # mutable struct ...
    elif definition_code.startswith("mutable struct"):
        def_type = "struct"
        struct_name_match = re.search(r'^mutable\s+struct\s+([a-zA-Z0-9_!?\']+)', definition_code)
        if struct_name_match:
            def_name = struct_name_match.group(1)

    # struct ...
    elif definition_code.startswith("struct"):
        def_type = "struct"
        struct_name_match = re.search(r'^struct\s+([a-zA-Z0-9_!?\']+)', definition_code)
        if struct_name_match:
            def_name = struct_name_match.group(1)

    return def_type, def_name

def chunk_julia_file_by_docstring(file_path):
    """
    Reads a single .jl file and returns a list of chunk dicts, each containing:
      - type
      - name
      - docstring
      - definition_code
      - metadata (filename, start_line, end_line)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    chunks = []
    for docstring_text, definition_code, start_off, end_off in find_docstring_definitions(code):
        start_line, end_line = compute_line_numbers(code, start_off, end_off)
        def_type, def_name = parse_definition_type_and_name(definition_code)

        chunk = {
            "type": def_type,
            "name": def_name,
            "docstring": docstring_text,
            "definition_code": definition_code,
            "metadata": {
                "filename": file_path,
                "start_line": start_line,
                "end_line": end_line
            }
        }
        chunks.append(chunk)

    return chunks

def chunk_by_docstring(dir_path):
    """
    Recursively scans a directory for .jl files, extracts docstring+definition pairs,
    and returns a combined list of chunk dicts.
    """
    all_chunks = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".jl"):
                file_path = os.path.join(root, file)
                file_chunks = chunk_julia_file_by_docstring(file_path)
                all_chunks.extend(file_chunks)
    return all_chunks

from langchain.schema import Document

def transform_chunks_to_documents(chunk_dicts):
    documents = []
    for chunk in chunk_dicts:
        page_content = f"{chunk.get('docstring', '')}\n\n{chunk.get('definition_code', '')}"
        metadata = chunk.get("metadata", {})
        
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
    
    return documents

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA

# Load and prepare text documents
def load_text_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

design_text = load_text_file(r"C:\repos\Hedgehog2.jl\architectural_decisions\design.md")
code = load_text_file(r"C:\repos\Hedgehog2.jl\src\Hedgehog2.jl")

# Split text into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
design_docs = text_splitter.create_documents([design_text])

julia_dir = r"C:\repos\Hedgehog2.jl\src"
code_docs = chunk_by_docstring(julia_dir)

# Use Ollama embeddings
embedding_model = OllamaEmbeddings(model="mistral")

# Create separate vector stores
design_vectorstore = Chroma.from_documents(design_docs, embedding=embedding_model)
code_vectorstore = Chroma.from_documents(transform_chunks_to_documents(code_docs), embedding=embedding_model)

# Create separate retrievers
design_retriever = design_vectorstore.as_retriever(search_kwargs={"k": 3})
code_retriever = code_vectorstore.as_retriever(search_kwargs={"k": 3})

print("✅ Design, answers, and code indexed in ChromaDB!")

# Load LLM for answering
llm_retrieval = ChatOllama(model="mistral")  # Fast model for retrieval Q&A
llm_coding = ChatOllama(model="codellama:34b")  # Strong model for coding

# Agents
design_qa = RetrievalQA.from_chain_type(llm_retrieval, retriever=design_retriever)
code_qa = RetrievalQA.from_chain_type(llm_retrieval, retriever=code_retriever)

def retrieve_design_decisions(user_input):
    return f"You are an AI agent helping a DEV finding design decisions in Hedgehog2.jl to keep in mind while performing its coding tasks. Retrieve the design decisions relevant to help a dev satisfying this feature request: {user_input}. Your answer is part of the prompt for the DEV. It should contain only high level instructions. Don't write any code."

def retrieve_code(user_input):
    return f"""You are an AI agent helping a DEV finding code examples in Hedgehog.jl that can help it write code respecting existing code patterns. Retrieve existig code that can be helpful example for the dev satisfying this feature request: {user_input}. 
    For each example explain which similar request it answers. Your answer is part of the prompt for the DEV. You should not write any code that is not already part of the library."""


# Interactive session
while True:
    user_input = input("\nYou: ")

    if user_input.lower() == "end":
        print("👋 Chat session ended.")
        break

    print("retrieving information to help the coding agent")
    # Retrieve relevant information
    design_response = design_qa.invoke(retrieve_design_decisions(user_input))["result"]
    code_response = code_qa.invoke(retrieve_code(user_input))["result"]
    print(f"The relevant design choices are: {design_response}\n")
    print(f"The relevant code examples are: {code_response}\n")

    # Generate improved response using all retrieved knowledge
    final_prompt = f"""
    You are an AI Quant Developer that helps writing code for the Hedgehog2.jl derivatives pricing library.
    You are expected to answer feature requests by the user with julia code to be added to the library.
    You are provided with a feature request, the relevant design choices you need to follow and the relevant code from the library.
    You need to be consistent with the library design fully.

    Feature request: {user_input}
    
    Relevant Design Decisions:
    {design_response}
    
    Relevant Code Snippets:
    {code_response}
    """

    print(f"FINAL PROMPT \n --------------- \n {final_prompt}")
    final_response = llm_coding.invoke(final_prompt)
    print("waiting for the coding agent answer")

    print("\n🧠 AI Response:", final_response.content)
