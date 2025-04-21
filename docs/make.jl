using Documenter
using Hedgehog

# Function to generate ADR pages that display raw YAML
function generate_raw_adr_pages()
    adr_dir = joinpath(@__DIR__, "adr")
    output_dir = joinpath(@__DIR__, "src", "adr")
    
    # Create output directory if it doesn't exist
    isdir(output_dir) || mkdir(output_dir)
    
    # Read the index file to get list of ADRs
    index_path = joinpath(adr_dir, "index.yaml")
    index_content = read(index_path, String)
    
    # Extract file names from index content
    adr_files = []
    for line in split(index_content, '\n')
        if occursin(".yaml", line)
            # Extract filename from the line using regex
            m = match(r"\"([^\"]+)\"", line)
            if m !== nothing
                push!(adr_files, m.captures[1])
            end
        end
    end
    
    # Create individual markdown pages for each ADR
    adr_pages = []
    for adr_file in adr_files
        adr_path = joinpath(adr_dir, adr_file)
        
        # Read the raw YAML content
        yaml_content = read(adr_path, String)
        
        # Extract ADR ID and title using regex
        adr_id = match(r"adr_id:\s*(\d+)", yaml_content).captures[1]
        title = match(r"title:\s*\"([^\"]+)\"", yaml_content).captures[1]
        
        # Generate a markdown file for this ADR
        md_filename = replace(adr_file, ".yaml" => ".md")
        md_path = joinpath(output_dir, md_filename)
        
        open(md_path, "w") do io
            write(io, "# ADR-$(adr_id): $(title)\n\n")
            write(io, "```yaml\n")
            write(io, yaml_content)
            write(io, "```\n")
        end
        
        # Add to the list of pages
        push!(adr_pages, "ADR-$(adr_id): $(title)" => "adr/$(md_filename)")
    end
    
    # Create an index page for ADRs
    open(joinpath(output_dir, "index.md"), "w") do io
        write(io, "# Architecture Decision Records\n\n")
        write(io, "This section contains the Architecture Decision Records (ADRs) for Hedgehog.jl.\n\n")
        write(io, "## Index\n\n")
        
        for (i, adr_file) in enumerate(adr_files)
            # Extract ID and title using regex from filename
            md_content = read(joinpath(adr_dir, adr_file), String)
            adr_id = match(r"adr_id:\s*(\d+)", md_content).captures[1]
            title = match(r"title:\s*\"([^\"]+)\"", md_content).captures[1]
            
            md_filename = replace(adr_file, ".yaml" => ".html")
            write(io, "$(i). [ADR-$(adr_id): $(title)](adr/$(md_filename))\n")
        end
        
        # Display the index file itself
        write(io, "\n## Raw Index File\n\n")
        write(io, "```yaml\n")
        write(io, index_content)
        write(io, "```\n")
    end
    
    return [
        "Overview" => "adr/index.md", 
        adr_pages...
    ]
end

# Run the ADR generation function
adr_pages = generate_raw_adr_pages()

# Create a simple wrapper for the roadmap
roadmap_path = joinpath(@__DIR__, "src", "derivatives_pricing_roadmap.md")
roadmap_content = read(joinpath(@__DIR__, "derivatives_pricing_roadmap.md"), String)
open(roadmap_path, "w") do io
    write(io, "# Derivatives Pricing Roadmap\n\n")
    write(io, roadmap_content)
end

makedocs(
    sitename = "Hedgehog.jl",
    modules = [Hedgehog],
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://aleCombi.github.io/Hedgehog.jl/stable/",
        assets = ["assets/favicon.ico"],
        highlights = ["julia", "yaml"],
    ),
    clean = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Pricing Methods" => "pricing_methods.md",
        "Architecture Decisions" => adr_pages,
        "Roadmap" => "derivatives_pricing_roadmap.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/aleCombi/Hedgehog.jl.git",
    devbranch = "master",  # Change to "main" if that's your default branch
    target = "build",
    push_preview = true
)