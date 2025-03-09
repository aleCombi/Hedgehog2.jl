#!/usr/bin/env julia

using FilePathsBase

"""
    ai_with_library(library_path, design_file_path, query)

Reads:
1. All `.jl` files under `library_path` (recursively).
2. The text file at `design_file_path` containing design choices.

Then:
- Concatenates these two sources of text into a single prompt.
- Sends the prompt to the `ollama` command (running the `mistral` model by default).
- Returns the AI's response.

Adjust `max_length` as needed to fit model token limits.
"""
function ai_with_library(library_path::String, design_file_path::String, query::String)
    println("Reading Julia library from: ", library_path)
    println("Reading design choices from: ", design_file_path)

    # 1. Recursively collect all `.jl` files
    files = []
    for (root, _, filenames) in walkdir(library_path)
        jl_files = filter(f -> endswith(f, ".jl"), filenames)
        append!(files, joinpath.(root, jl_files))
    end

    # 2. Read the library code
    all_file_contents = join([read(f, String) for f in files], "\n\n")

    # 3. Enforce maximum length for library code to avoid overshooting token limit
    max_length = 10000
    truncated_library_code = all_file_contents[1:min(end, max_length)]

    # 4. Read design choices (if file exists)
    design_choices_text = ""
    if isfile(design_file_path)
        design_choices_text = read(design_file_path, String)
    else
        @warn "Design file not found at $design_file_path. Using empty design choices."
    end

    # 5. Construct the AI prompt
    prompt = """
Here are the design choices (if any):
$design_choices_text

Here is the Julia library code (truncated if large):
$truncated_library_code

Answer this question about it:
$query
"""

    println("Sending query to Ollama...")

    # 6. Run `ollama` with the constructed prompt
    response = read(`ollama run mistral "$prompt"`, String)

    # 7. Debugging: print the AI response locally
    println("AI Response: ", response)

    return response
end


# Handle command-line arguments only if run directly.
if abspath(PROGRAM_FILE) == @__FILE__
    args = ARGS  # Command-line arguments

    # Set defaults
    default_library_path = "C:\\repos\\Hedgehog2.jl"
    default_design_file_path = "C:\\repos\\Hedgehog2.jl\\architectural_decisions\\julia_pricing_framework.md"

    # Ensure the user provides at least a query
    if length(args) < 1
        println("Usage:  julia ai_with_library.jl \"<query>\" [library_path] [design_file_path]")
        println("Example:")
        println("  julia ai_with_library.jl \"How does option pricing work?\"")
        println("  (Optionally provide library path and design-choices file path.)")
        exit(1)
    end

    # Extract arguments
    query = args[1]
    library_path = length(args) > 1 ? args[2] : default_library_path
    design_file_path = length(args) > 2 ? args[3] : default_design_file_path

    # Run AI function
    println(ai_with_library(library_path, design_file_path, query))
end
