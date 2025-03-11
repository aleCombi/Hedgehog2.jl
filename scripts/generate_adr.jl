using Dates
using Printf

# Define ADR directory
const ADR_DIR = "docs/adr"

# Function to slugify a title
function slugify(title)
    return lowercase(replace(title, r"[^\w\s]" => "", " " => "-"))
end

# Function to get the next ADR number
function get_next_adr_number()
    files = filter(x -> endswith(x, ".yaml"), readdir(ADR_DIR, join=true))
    numbers = [parse(Int, match(r"adr-(\d+)", f).captures[1]) for f in files if occursin(r"adr-\d+", f)]
    return isempty(numbers) ? 1 : maximum(numbers) + 1
end

# Function to create a new ADR file
function create_adr(title)
    # Ensure ADR directory exists
    isdir(ADR_DIR) || mkpath(ADR_DIR)

    # Generate file name
    adr_number = get_next_adr_number()
    slug = slugify(title)
    filename = @sprintf "adr-%03d-%s.yaml" adr_number slug
    filepath = joinpath(ADR_DIR, filename)

    # Generate ADR content
    content = """
    adr_id: $adr_number
    title: \"$title\"
    status: Draft
    date: $(Dates.today())
    context: |
      # Add context for this decision
    decision: |
      # Describe the decision made
    consequences:
      positive:
        - "Describe benefits"
      negative:
        - "Describe drawbacks"
    alternatives:
      - name: "Alternative 1"
        pros: "Pros of this alternative"
        cons: "Cons of this alternative"
      - name: "Alternative 2"
        pros: "Pros of this alternative"
        cons: "Cons of this alternative"
    references: []
    """

    # Write ADR file
    open(filepath, "w") do io
        write(io, content)
    end

    println("✅ ADR Created: $filename")

    # Update index file
    update_adr_index()

    return filename
end

# Function to update ADR index
function update_adr_index()
    index_path = joinpath(ADR_DIR, "index.yaml")
    files = filter(x -> endswith(x, ".yaml") && x != "index.yaml", readdir(ADR_DIR))
    sorted_files = sort(files)

    content = """
    adr_index:
    """
    for file in sorted_files
        content *= "  - \"$file\"\n"
    end

    open(index_path, "w") do io
        write(io, content)
    end

    println("📌 ADR index updated.")
end

# Main Execution
if length(ARGS) < 1
    println("Usage: julia generate_adr.jl \"Your ADR Title Here\"")
    exit(1)
end

title = ARGS[1]
create_adr(title)
