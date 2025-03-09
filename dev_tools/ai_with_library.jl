function ai_with_library(library_path, query)
    println("Reading Julia library from: ", library_path)

    # Recursively collect all `.jl` files in subdirectories
    files = []
    for (root, _, filenames) in walkdir(library_path)
        append!(files, filter(f -> endswith(f, ".jl"), joinpath.(root, filenames)))
    end

    # Read file contents
    file_contents = join([read(f, String) for f in files], "\n\n")

    # Limit content size to avoid exceeding token limit
    max_length = 10000  # Adjust based on model capability
    truncated_contents = file_contents[1:min(end, max_length)]  # Trim if too long

    # Construct AI prompt with code context
    prompt = "Here is a Julia library (truncated if large):\n\n" * truncated_contents * 
             "\n\nAnswer this question about it:\n" * query

    println("Sending query to Ollama...")

    # Run Ollama and capture response
    response = read(`ollama run mistral "$prompt"`, String)

    # Debugging: Show AI response
    println("AI Response: ", response)

    return response
end

# Handle command-line arguments
if abspath(PROGRAM_FILE) == @__FILE__
    args = ARGS  # Get command-line arguments

    # Default library path
    default_library_path = "C:\\repos\\Hedgehog2.jl"

    # Ensure the user provides at least a query
    if length(args) < 1
        println("Usage: julia ai_with_library.jl \"<query>\" [library_path]")
        println("Example: julia ai_with_library.jl \"How does option pricing work?\"")
        exit(1)
    end

    # Extract arguments
    query = args[1]
    library_path = length(args) > 1 ? args[2] : default_library_path

    # Run AI with the provided query and print response
    println(ai_with_library(library_path, query))
end
