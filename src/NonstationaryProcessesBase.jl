module NonstationaryProcessesBase

using Requires
using Random
using StaticArrays
using Distributions
using Tullio
using FFTW
using Setfield
using StatsBase
using Reexport
using TimeseriesTools

export seed, ParameterProfiles

function __init__()
    @require BifurcationKit="0f109fa4-8a5d-4b75-95aa-f515264e7665" @eval include("Bifurcations.jl")
    @require DynamicalSystems="61744808-ddfa-5f27-97ff-6e42cc95d634" @eval include("DynamicalSystems.jl")
    @require StatsPlots="f3b207a7-027a-5e70-b257-86293d7955fd" @eval include("Plots/Plotting.jl")
    @require PyPlot="d330b81b-6aea-500a-939a-2ce795aea3ee" @eval include("Plots/PyPlotTools.jl")
end

function seed(theSeed=nothing) # Seed the rng, but return the seed. If no, nothing, or NaN argument, randomly seed rng
    if isnothing(theSeed)
        theSeed = abs(Random.rand(Int64))
    end
    Random.seed!(theSeed)
    return theSeed
end

include("Discontinuous.jl")
include("ParameterProfiles.jl")
include("Process.jl")
include("AMI.jl")

end
