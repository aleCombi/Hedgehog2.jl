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

if __name__ == "__main__":
    # Example usage
    julia_dir = r"C:\repos\Hedgehog2.jl\src"
    results = chunk_by_docstring(julia_dir)
    print(f"Found {len(results)} docstring-based definitions.")
    for result in results:
        print("Chunk:")
        print(results[0])
        print(r"\n")
