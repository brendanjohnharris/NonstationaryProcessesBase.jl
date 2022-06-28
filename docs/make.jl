using NonstationaryProcessesBase
using Documenter

DocMeta.setdocmeta!(NonstationaryProcessesBase, :DocTestSetup, :(using NonstationaryProcessesBase); recursive=true)

makedocs(;
    modules=[NonstationaryProcessesBase],
    authors="brendanjohnharris <brendanjohnharris@gmail.com> and contributors",
    repo="https://github.com/brendanjohnharris/NonstationaryProcessesBase.jl/blob/{commit}{path}#{line}",
    sitename="NonstationaryProcessesBase.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Parameter profiles" => "parameterprofiles.md",
    ],
)
