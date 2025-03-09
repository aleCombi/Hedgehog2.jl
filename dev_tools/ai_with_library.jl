function ai_with_library(library_path, query)
    println("Sending query to Ollama: ", query)

    # Run Ollama and capture response
    response = read(`ollama run mistral "$query"`, String)

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
