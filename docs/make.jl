using Documenter
using Hedgehog

makedocs(
    sitename = "Hedgehog.jl",
    modules = [Hedgehog],
    format = Documenter.HTML(),
    clean = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Pricing Methods" => "pricing_methods.md",
        "API Reference" => "api.md"
    ]
)

deploydocs(
    repo = "github.com/aleCombi/Hedgehog.jl.git",
    devbranch = "master",  # Change to "main" if that's your default branch
    target = "build",
    push_preview = true
)