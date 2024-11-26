using DimensionalData
using DelimitedFiles
using Dates
using SciMLBase
import TimeseriesTools.TimeSeries
import TimeseriesTools.times

export Process, process_aliases, fieldguide

Base.@kwdef mutable struct Process
    process = nothing
    parameter_profile::Union{Function,Tuple,Array,NTuple} = x -> 0.0 # Can be a tuple of functions, if the system has more than one parameter
    parameter_profile_parameters = [] # Can be a tuple of tuples
    X0::Vector = [nothing]
    t0::Union{Float64,Int64} = 0.0
    transient_t0::Union{Float64,Int64} = t0 # t0 will always take precedence
    dt::Union{Float64,Int64} = 0.001
    savedt::Union{Float64,Int64} = 0.01
    tmax::Union{Float64,Int64} = 100.0
    alg::Union{SciMLBase.SciMLAlgorithm,Function,Nothing} = nothing
    solver_opts::Dict = Dict(:adaptive => false)
    #parameter_rng::UInt64 = seed()
    solver_rng::Int64 = seed()
    id::Int64 = abs(rand(Int64)) # Just a unique number for this simulation
    date::String = string(Dates.now())
    solution = nothing
    varnames::Vector{Symbol} = defaultvars(length(X0))
end

function subshow(io, P)
    namelen = maximum(length.(string.((fieldnames ∘ typeof)(P)))) + 2
    fillfun = x -> repeat(" ", namelen - length(string(x)))
    showfields = (fieldnames ∘ typeof)(P)
    rows = ["   $x:$(fillfun(x))" * string(getfield(P, x)) * "\n" for x ∈ showfields]
    issolved = !isnothing(getsolution(P))
    rows[findfirst(showfields .== :solution)] = "   solution:$(fillfun(:solution))$issolved\n"
    print(io, reduce(*, rows))
end
function Base.show(io::IO, P::Process)
    print(io, "Process with fields:\n")
    subshow(io, P)
end
function Base.show(io::IO, m::MIME"text/plain", P::Process)
    printstyled(io, "Process", color=:red, bold=true)
    printstyled(io, " with fields:\n", bold=true)
    subshow(io, P)
end

function defaultvars(x)
    vars = Symbol.(Char.([(120:122)..., 119, 117, 118]))
    (length(x) <= length(vars) ? vars[1:x] : 1:x)
end

"""
    Process(; process=nothing, process_kwargs...)
Define a `Process` that simulates a given system, specified by `process::Function`, and any simulation parameters (including control parameters, sampling parameters, solver parameters, and some metadata like the date and a simulation ID).
For a complete list of `process_kwargs`, see the [`fieldguide`](@ref NonstationaryProcessesBase.fieldguide).
A `Process` can also be used to create other `Process`es using the syntax `S = (P::Process)(;process_kwargs...)`, which will first copy `P` then update any fields given in `process_kwargs`
"""
function Process(D::Dict)
    for s ∈ setdiff(keys(D), (:date, :solution))
        if D[s] isa String
            if s == :process
                try
                    D[s] = eval(Meta.parse(D[s])) # Maybe we can't parse the process as a function
                catch
                    D[s] = D[s] # It must be a string. You won't be able to solve this process, so the timeseries better be supplied.
                end
            else
                if s == :gitcommit
                    pop!(D, s) # We don't need this
                else
                    D[s] = eval(Meta.parse(D[s]))
                end
            end
        end
    end
    Process(; D...)
end

function (P::Process)(; kwargs...)
    # Can use field aliases here
    kwargs = Dict{Any,Any}(kwargs)
    repalias!(kwargs, process_aliases)

    # * If the parameter profile is given as a number, assume this is a stationary process
    if :parameter_profile ∈ keys(kwargs) && !(:parameter_profile_parameters ∈ keys(kwargs)) && kwargs[:parameter_profile] isa Union{Number,Vector{<:Number}}
        kwargs[:parameter_profile_parameters] = Tuple([y for y in kwargs[:parameter_profile]])
        kwargs[:parameter_profile] = Tuple([constant for y in kwargs[:parameter_profile]])
    end

    # * Copy kwargs onto a new Process
    P2 = deepcopy(P)
    [setfield!(P2, x, y) for (x, y) in kwargs]

    # * Set new process metadata
    setfield!(P2, :solution, nothing) # You've changed some parameters, so the solution is no longer valid
    setfield!(P2, :id, abs(rand(Int64))) # New id, yeah?
    setfield!(P2, :solver_rng, seed()) # New random seed, yeah?
    setfield!(P2, :date, string(Dates.now())) # New datetime, yeah?
    return P2
end

"""
Field aliases for `Process` constructors, given as a dictionary of `(field=>alias)` pairs.
"""
process_aliases = Dict(
    :process => [:sim, :system, :processes],
    :parameter_profile => [:profile, :profiles, :𝑝, :𝑃, :parameter_profiles],
    :parameter_profile_parameters => [:parameters, :ps, :params, :param, :parameter,
        :profile_parameters, :parameterprofileparameters,
        :profileparameters, :𝔓, :𝔭],
    :X0 => [:initial_conditions, :X, :X_0, :X₀, :𝑥₀, :𝑋₀, :𝑥0,
        :𝑋0],
    :transient_t0 => [:transient, :cutoff, :tₜ, :𝑡ₜ, :tt],
    :t0 => [:tstart, :t₀, :𝑡₀],
    :dt => [:δt, :𝛿t, :δ𝑡, :𝛿𝑡],
    :savedt => [:save_dt, :save_Δt, :save_δt, :save_𝛥t, :save_𝛿t, :save_Δ𝑡,
        :save_δ𝑡, :save_𝛥𝑡, :save_𝛿𝑡, :Δt, :𝛥t, :Δ𝑡, :Δ𝑡],
    :tmax => [:t_max, :T, :Tmax, :T_max, :𝑇],
    :alg => [:algorithm, :solver],
    :solver_opts => [:opts, :solopts, :sol_opts, :solveropts],
    :solver_rng => [:rng, :rngseed, :rng_seed, :solverrng],
    :id => [:identifier, :inventory_id],
    :date => [:time, :datetime],
    :solution => [:sol, :result, :output],
    :varnames => [:variables, :variablenames, :variable_names]
)

formatguide(x, P::Type) = reduce(*, ["- `$key`: $(x[key])\n" for key in fieldnames(P)])

"""
A dictionary containing brief descriptions of each `Process` field:.
$(formatguide(fieldguide, Process))
"""
fieldguide = Dict(
    :process => "A `Function` with a method for `Process` types that performs a particular simulation. See [NonstationaryProcesses.jl](https://github.com/brendanjohnharris/NonstationaryProcesses.jl) for examples.",
    :parameter_profile => "A tuple of `Function`s that describe how each parameter of a system evolves over time. These can be functions, or curried functions of parameters given in `:parameter_profile_parameters`. See e.g. [`constantParameter`](@ref ParameterProfiles.constantParameter), [`Discontinuous`](@ref)",
    :parameter_profile_parameters => "A tuple of parameters (which may also be tuples) for each `:parameter_profile`",
    :X0 => "The initial conditions, given as a vector equal in length to number of variables in the system",
    :transient_t0 => "The length of time to simulate the system, from the initial conditions `:X0` at `:t0`, that is discarded when retrieving the [`timeseries`](@ref)",
    :t0 => "The initial time of the simulation, at which the system has the state `:X0`",
    :dt => "The time step of the simulation and solver",
    :savedt => "The sampling period of the solution's [`timeseries`](@ref), which must be a multiple of `:dt`",
    :tmax => "The final time point of the simulation. The duration of the simulation is then `:tmax - :transient_t0`, and the duration of the returned time series is `:tmax - :t0`. See [`times`](@ref)",
    :alg => "The algorithm used to solve `DifferentialEquations.jl` processes. See, for example, a list of [ODE solvers](https://diffeq.sciml.ai/stable/solvers/ode_solve/)",
    :solver_opts => "A dictionary of additional options passed to the `:alg`. This can include solver tolerances, adaptive timesteps, or the maximum number of iterations. See the [common solver options](https://diffeq.sciml.ai/stable/basics/common_solver_opts/)",
    :solver_rng => "An integer seed for the random number generator, set to a random number by default",
    :id => "A unique integer ID for a given [`Process`](@ref)",
    :date => "The date at which the [`Process`](@ref) was created",
    :solution => "The solution of the simulation in its native format (e.g. an [ODE solution](https://diffeq.sciml.ai/stable/types/ode_types/#SciMLBase.ODESolution))",
    :varnames => "Dummy names for the variables of a [`Process`](@ref), defaulting to `[:x, :y, :z, ...]`"
)



function repalias!(D, aliai::Dict)
    for d ∈ keys(D)
        for a ∈ keys(aliai)
            if d ∈ aliai[a]
                D[a] = pop!(D, d)
            end
        end
    end
end

"""
Solve a [`Process`](@ref) by calling the `(::Process).process` method.
"""
function solution!(P::Process) # vars::Tuple=Tuple(1:size(P.X0)[1])
    if isnothing(P.solution)
        @debug "Solving for the $(getprocess(P)) process ($(getid(P)))"
        P.solution = P.process(P)
    end
    return P.solution
end
export solution!

"""
Copy then solve a [`Process`](@ref)
"""
function simulate(P::Process)
    P2 = deepcopy(P)
    solution!(P2)
    return P2
end
simulate!(P::Process) = (P.solution = nothing; solution!(P); nothing)
export simulate
export simulate!

# * Might have various solution types, so timeseries gets all of them as an array
TimeSeries(s::SciMLBase.AbstractTimeseriesSolution, dim::Real) = s[dim, :]
TimeSeries(s::SciMLBase.AbstractTimeseriesSolution, dim::Union{Vector,UnitRange}=1:size(s.u[1], 1)) = copy(s[dim, :]')
function TimeSeries(s::AbstractArray, dim::Union{Vector,UnitRange,Real}=1:size(s, 2))
    if s isa Vector
        if length(dim) != 1 || dim[1] != 1
            error("Cannot index the second dimension of the input, which is a vector")
        end
        s[:]
    else
        s[:, dim]
    end
end
timeseries = TimeseriesTools.TimeSeries

"""
Return the `:solution` of a [`Process`](@ref) as a formatted time series. cf. [`timeseries`](@ref)
"""
function timeseries!(P::Process, dim=1:length(getX0(P)); transient::Bool=false)
    # P.solution = nothing
    x = timeseries(solution!(P), dim)
    if transient
        idxs = 1:length(times(P, transient=true))
    else
        idxs = (length(P.transient_t0:P.savedt:P.t0)):1:length(times(P, transient=true))
    end
    saveTimes = (P.transient_t0:P.savedt:P.tmax)[idxs]
    namevars = P.varnames[dim]
    if size(x, 2) > 1
        x = ToolsArray(x[idxs, :], (𝑡(saveTimes), TimeseriesTools.Var(namevars)))
    else
        x = ToolsArray(x[idxs], (𝑡(saveTimes),))
    end
end
export timeseries

timeDims(T::ToolsArray) = dims(T, 𝑡).val
variableDims(T::ToolsArray) = dims(T, :Variable).val
# function timeseries(s::Tuple, dim::Union{Real, Vector, Tuple}=1)
#     timeseries(s[1], dim) # You gave the metadata as well
# end

"""
    times(P::Process; transient::Bool=false)
Return the time indices of a [`Process`](@ref), optionally including the transient.
"""
function TimeseriesTools.times(P::Process; transient::Bool=false)
    if transient
        P.transient_t0:P.savedt:P.tmax
    else
        P.t0:P.savedt:P.tmax
    end
end
export times

"""
Return the profiles of a [`Process`](@ref) as a vector-outputting function with the form `f(t::Real) -> ::Vector`
"""
parameter_function(P::Process) = tuplef2ftuple(P.parameter_profile, P.parameter_profile_parameters)
export parameter_function

"""
Return the profiles of a [`Process`](@ref) as a vector of scalar-valued functions each with the form `f(t::Real) -> ::Real`
"""
function parameter_functions(P::Process)
    if all(isempty.(getps(P)))
        getprofiles(P)
    else
        if P.parameter_profile isa Union{Tuple,Vector}
            [P.parameter_profile[x](P.parameter_profile_parameters[x]...) for x in 1:length(P.parameter_profile)]
        else
            P.parameter_profile(P.parameter_profile_parameters...)
        end
    end
end
export parameter_functions

"""
    parameterseries(P::Process; p=nothing, times_kwargs...)
Return the profiles of a [`Process`](@ref) as parameter values at time indices given by [`times`](@ref).
`p` specified the indices of the parameters to return (e.g. 1 or [2, 3]).
If `p` is a vector, the values will be returned in an nₚ×nₜ matrix.
"""
function parameterseries(P::Process; p=nothing, kwargs...)
    ps = hcat(parameter_function(P).(times(P; kwargs...))...)
    if size(ps, 1) == 1 # This 1 × N array, which should be a vector
        ps = ps[:]
    end
    if isnothing(p) || (ps isa Vector)
        return ps
    else
        return ps[p, :]
    end
end
export parameterseries

# Access the fields of a process with functions
for field ∈ keys(process_aliases)
    f = Symbol(:get, field)
    eval(quote
        $f(P::Process) = P.$field
        export $f
    end)
    for field_alias ∈ process_aliases[field]
        fa = Symbol(:get, field_alias)
        eval(quote
            $fa = $f #export fa
        end)
    end
end

"""
Return the indices of parameter profiles that are not [`constant`](@ref ParameterProfiles.constantParameter)
"""
function getvaryingparameters(P::Process)
    fs = getparameter_profile(P)
    ps = typeof.(fs) .!= typeof(constantParameter)
    findall(ps)
end
export getvaryingparameters

function forcevec(x)
    if !(x isa Union{AbstractArray,Tuple})
        x = [x]
    else
        x
    end
end
export forcevec

function forcemat(x)
    if x isa AbstractVector
        x = reshape(x, :, 1)
    end
    return x
end
export forcemat

function trimtransient(P::Process)
    if !isempty(P.solution)
        P.solution = parent(timeseries(P, transient=false))
        P.X0 = P.solution[1, :] # If you want to resimulate, start at this point. Assumes the entire description of the system is contained in time and position, which is not unreasonable
    end
    P.transient_t0 = P.t0
    return P
end

"""
Remove a transient from a [`Process`](@ref) by dropping any time-series values between `:transient_t0` and `:t0`, as well as setting the intitial condition to the value of the solution at `:t0`.
"""
trimtransient! = trimtransient;
export trimtransient!

"""
Save the solution of a [`Process`](@ref) in a given folder. This replaces the `:solution` field of the [`Process`](@ref) with the location of the saved data, and subsequent calls of [`timeseries`](@ref) read from this file.
"""
function saveTimeseries!(P::Process, folder::String="./", delim::Char=','; transient::Bool=false, fileroot="timeseries")
    X = timeseries(P, transient=transient)
    mkpath(folder)
    if !transient
        P = trimtransient(P)
    end
    filename = joinpath(folder, fileroot * "_" * string(getid(P)) * ".csv")
    P.solution = abspath(folder)
    #P.solution = nothing
    # @info "Saving time-series data to $filename"
    writedlm(filename, X, delim)
end
export saveTimeseries!

function gettimeseriesfile(P::Process, folder::String)
    filename = filter(x -> occursin("timeseries_" * string(getid(P)), x), readdir(folder))
end

"""
Retrieve the solution of a [`Process`](@ref) as a [`ToolsArray`](https://rafaqz.github.io/DimensionalData.jl/stable/api/#DimensionalData.ToolsArray), starting from `:t0`, at a sampling period of `:save_dt`. This function will solve the [`Process`](@ref) and populate the `:solution` only if the [`Process`](@ref) has not yet been simulated.
"""
function TimeSeries(P::Process, dim=1:length(getX0(P)); folder::Union{String,Bool}=(getsolution(P) isa String), kwargs...)
    if folder isa Bool && folder
        if getsolution(P) isa String
            folder = getsolution(P)
        else
            folder = "./"
        end
    end
    if folder isa String
        filename = gettimeseriesfile(P, folder)
    else
        return timeseries!(P, dim; kwargs...)
    end
    if isempty(filename)
        return timeseries!(P, dim; kwargs...)
    end
    filename = filename[1]
    @debug "Loading time-series data from $filename"
    filename = joinpath(folder, filename)
    P.solution = readdlm(filename, ',', Float64)
    return timeseries!(P, dim; kwargs...)
end

flat_tuple(x::Tuple)::Tuple = x
flat_tuple(x::AbstractVector)::Tuple = Tuple(x)
flat_tuple(x::Function)::Tuple{Function} = (x,)


"""
    updateparam(P::Process, p::Integer, profile::Function)
    updateparam(P, p, value::Union{Number, Tuple, Vector})
    updateparam(P, p, profile, value)
Copy a [`Process`](@ref) with a new set of `:parameter_profiles` and `:parameter_profile_parameters`.
`p` is an integer specifying which parameter to update, `profile` is the new profile, and `value` contains the new `:parameter_profile_parameters`.
"""
function updateparam(P::Process, p::Integer, profile, value)
    profiles = getparameter_profile(P) |> flat_tuple
    values = getparameter_profile_parameters(P) |> flat_tuple
    if length(profiles) == 1
        @assert p == 1
        profiles = [profile]
        values = value
    else
        profiles = [profiles[1:p-1]..., profile, profiles[p+1:end]...] # 🤮
        values = [values[1:p-1]..., value, values[p+1:end]...]
    end
    return P(parameter_profile=Tuple(profiles), parameter_profile_parameters=values)
end
function updateparam(P::Process, p::Integer, profile::Function)
    value = getparameter_profile_parameters(P)
    profiles = [getparameter_profile(P)...]
    value = (length(profiles) == 1 && p == 1) ? value : value[p]
    return updateparam(P, p, profile, value)
end
function updateparam(P::Process, p::Integer, value::Union{Number,Tuple,Vector})
    profile = getparameter_profile(P)
    profile = (length(profile) == 1 && p == 1) ? profile : profile[p]
    return updateparam(P, p, profile, value)
end
export updateparam
