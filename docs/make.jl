using Documenter
using Hedgehog

makedocs(
    sitename = "Hedgehog.jl",
    modules = [Hedgehog],
    format = Documenter.HTML(
        prettyurls = true,
        canonical = "https://aleCombi.github.io/Hedgehog.jl/stable/",
        assets = ["assets/favicon.ico"],
    ),
    clean = true,
    checkdocs = :exports,
    pages = [
        "Home" => "index.md",
        "Pricing Methods" => "pricing_methods.md",
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