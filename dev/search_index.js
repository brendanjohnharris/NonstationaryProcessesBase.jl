var documenterSearchIndex = {"docs":
[{"location":"parameterprofiles/","page":"Parameter profiles","title":"Parameter profiles","text":"CurrentModule = NonstationaryProcessesBase.ParameterProfiles","category":"page"},{"location":"parameterprofiles/","page":"Parameter profiles","title":"Parameter profiles","text":"Modules = [ParameterProfiles]","category":"page"},{"location":"parameterprofiles/#NonstationaryProcessesBase.ParameterProfiles.constantParameter","page":"Parameter profiles","title":"NonstationaryProcessesBase.ParameterProfiles.constantParameter","text":"A function for creating a constant parameter profile, such that if f = constantParamter(x) then f(x)(t) = x\n\n\n\n\n\n","category":"function"},{"location":"parameterprofiles/#NonstationaryProcessesBase.ParameterProfiles.rampInterval-NTuple{4, Real}","page":"Parameter profiles","title":"NonstationaryProcessesBase.ParameterProfiles.rampInterval","text":"Define a line over an interval. Saturate before and after this interval\n\n\n\n\n\n","category":"method"},{"location":"parameterprofiles/#NonstationaryProcessesBase.ParameterProfiles.rampOff-NTuple{4, Real}","page":"Parameter profiles","title":"NonstationaryProcessesBase.ParameterProfiles.rampOff","text":"Define a line over an interval. After this interval, saturate, but before, extrapolate.\n\n\n\n\n\n","category":"method"},{"location":"parameterprofiles/#NonstationaryProcessesBase.ParameterProfiles.rampOn-NTuple{4, Real}","page":"Parameter profiles","title":"NonstationaryProcessesBase.ParameterProfiles.rampOn","text":"Define a line over an interval. Before this interval, saturate, but after, extrapolate.\n\n\n\n\n\n","category":"method"},{"location":"parameterprofiles/#NonstationaryProcessesBase.ParameterProfiles.stepNoise","page":"Parameter profiles","title":"NonstationaryProcessesBase.ParameterProfiles.stepNoise","text":"stepNoise()\n\nCompose a discontinuous function of many square bumps and troughs, following a white noise process.     'stepHeight':   the standard deviation of the step height     'T':            a tuple, (t0, tmax)\n\nExamples\n\njulia> D = stepNoise();\nD([-1, 0, 1])\n\n\n\n\n\n","category":"function"},{"location":"parameterprofiles/#NonstationaryProcessesBase.ParameterProfiles.unitStep","page":"Parameter profiles","title":"NonstationaryProcessesBase.ParameterProfiles.unitStep","text":"unitStep(threshold::Real=0.0, baseline::Real=0.0, stepHeight::Real=1.0, stepOpt::Real=1.0)\n\nConstruct a constant function of 'x' that undergoes a step of size 'stepHeight' from a 'baseline' at 'threshold'. 'stepOpt' sets the value at 0. A Discontinuous type is returned, which can be called like a normal function but also contains the x-value for which there is a discontinuity.\n\nExamples\n\njulia> D = unitStep();\nD([-1, 0, 1])\n\n\n\n\n\n","category":"function"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = NonstationaryProcessesBase","category":"page"},{"location":"#NonstationaryProcessesBase","page":"Home","title":"NonstationaryProcessesBase","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for NonstationaryProcessesBase.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [NonstationaryProcessesBase]","category":"page"},{"location":"#NonstationaryProcessesBase.fieldguide","page":"Home","title":"NonstationaryProcessesBase.fieldguide","text":"A dictionary containing brief descriptions of each Process field:.\n\nprocess: A Function with a method for Process types that performs a particular simulation. See NonstationaryProcesses.jl for examples.\nparameter_profile: A tuple of Functions that describe how each parameter of a system evolves over time. These can be functions, or curried functions of parameters given in :parameter_profile_parameters. See e.g. constantParameter, Discontinuous\nparameter_profile_parameters: A tuple of parameters (which may also be tuples) for each :parameter_profile\nX0: The initial conditions, given as a vector equal in length to number of variables in the system\nt0: The initial time of the simulation, at which the system has the state :X0\ntransient_t0: The length of time to simulate the system, from the initial conditions :X0 at :t0, that is discarded when retrieving the timeseries\ndt: The time step of the simulation and solver\nsavedt: The sampling period of the solution's timeseries, which must be a multiple of :dt\ntmax: The final time point of the simulation. The duration of the simulation is then :tmax - :transient_t0, and the duration of the returned time series is :tmax - :t0. See times\nalg: The algorithm used to solve DifferentialEquations.jl processes. See, for example, a lsit of ODE solvers\nsolver_opts: A dictionary of additional options passed to the :alg. This can include solver tolerances, adaptive timesteps, or the maximum number of iterations. See the common solver options\nsolver_rng: An integer seed for the random number generator, set to a random number by default\nid: A unique integer ID for a given Process\ndate: The date at which the Process was created\nsolution: The solution of the simulation in its native format (e.g. an ODE solution)\nvarnames: Dummy names for the variables of a Process, defaulting to [:x, :y, :z, ...]\n\n\n\n\n\n","category":"constant"},{"location":"#NonstationaryProcessesBase.process_aliases","page":"Home","title":"NonstationaryProcessesBase.process_aliases","text":"Field aliases for Process constructors, given as a dictionary of (field=>alias) pairs.\n\n\n\n\n\n","category":"constant"},{"location":"#NonstationaryProcessesBase.Discontinuous","page":"Home","title":"NonstationaryProcessesBase.Discontinuous","text":" Discontinuous\n\nA type that contains a discontinuous function and an index of any of its discontinuities. This allows the construction of step parameter profiles that retain discontinuity information. You can also call the Discontinuous` type like a normal function, if you want to.\n\n\n\n\n\n","category":"type"},{"location":"#NonstationaryProcessesBase.Process-Tuple{Dict}","page":"Home","title":"NonstationaryProcessesBase.Process","text":"Process(; process=nothing, process_kwargs...)\n\nDefine a Process that simulates a given system, specified by process::Function, and any simulation parameters (including control parameters, sampling parameters, solver parameters, and some metadata like the date and a simulation ID). For a complete list of process_kwargs, see the fieldguide. A Process can also be used to create other Processes using the syntax S = (P::Process)(;process_kwargs...), which will first copy P then update any fields given in process_kwargs\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.getvaryingparameters-Tuple{Process}","page":"Home","title":"NonstationaryProcessesBase.getvaryingparameters","text":"Return the indices of parameter profiles that are not constant\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.parameter_function-Tuple{Process}","page":"Home","title":"NonstationaryProcessesBase.parameter_function","text":"Return the profiles of a Process as a vector-outputting function with the form f(t::Real) -> ::Vector\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.parameter_functions-Tuple{Process}","page":"Home","title":"NonstationaryProcessesBase.parameter_functions","text":"Return the profiles of a Process as a vector of scalar-valued functions each with the form f(t::Real) -> ::Real\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.parameterseries-Tuple{Process}","page":"Home","title":"NonstationaryProcessesBase.parameterseries","text":"parameterseries(P::Process; p=nothing, times_kwargs...)\n\nReturn the profiles of a Process as parameter values at time indices given by times. p specified the indices of the parameters to return (e.g. 1 or [2, 3]). If p is a vector, the values will be returned in an nₚ×nₜ matrix.\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.saveTimeseries!","page":"Home","title":"NonstationaryProcessesBase.saveTimeseries!","text":"Save the solution of a Process in a given folder. This replaces the :solution field of the Process with the location of the saved data, and subsequent calls of timeseries read from this file.\n\n\n\n\n\n","category":"function"},{"location":"#NonstationaryProcessesBase.simulate-Tuple{Process}","page":"Home","title":"NonstationaryProcessesBase.simulate","text":"Copy then solve a Process\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.solution!-Tuple{Process}","page":"Home","title":"NonstationaryProcessesBase.solution!","text":"Solve a Process by calling the (::Process).process method.\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.timeseries!","page":"Home","title":"NonstationaryProcessesBase.timeseries!","text":"Return the :solution of a Process as a formatted time series. cf. timeseries\n\n\n\n\n\n","category":"function"},{"location":"#NonstationaryProcessesBase.trimtransient!","page":"Home","title":"NonstationaryProcessesBase.trimtransient!","text":"Remove a transient from a Process by dropping any time-series values between :transient_t0 and :t0, as well as setting the intitial condition to the value of the solution at :t0.\n\n\n\n\n\n","category":"function"},{"location":"#NonstationaryProcessesBase.tuplef2ftuple-Tuple{Any, Any}","page":"Home","title":"NonstationaryProcessesBase.tuplef2ftuple","text":"tuplef2ftuple(f::Tuple{Function}, params::Tuple)\n\nConvert a vector of curried functions f that each accept an element of the parameter tuple params into a function of the form f(t) -> x::Vector, where xᵢ = fᵢ(paramsᵢ)(t) for i = 1:length(f)\n\n\n\n\n\n","category":"method"},{"location":"#NonstationaryProcessesBase.updateparam-Tuple{Process, Integer, Any, Any}","page":"Home","title":"NonstationaryProcessesBase.updateparam","text":"updateparam(P::Process, p::Integer, profile::Function)\nupdateparam(P, p, value::Union{Number, Tuple, Vector})\nupdateparam(P, p, profile, value)\n\nCopy a Process with a new set of :parameter_profiles and :parameter_profile_parameters. p is an integer specifying which parameter to update, profile is the new profile, and value contains the new :parameter_profile_parameters.\n\n\n\n\n\n","category":"method"},{"location":"#TimeseriesTools.TimeSeries","page":"Home","title":"TimeseriesTools.TimeSeries","text":"Retrieve the solution of a Process as a ToolsArray, starting from :t0, at a sampling period of :save_dt. This function will solve the Process and populate the :solution only if the Process has not yet been simulated.\n\n\n\n\n\n","category":"function"},{"location":"#TimeseriesTools.times-Tuple{Process}","page":"Home","title":"TimeseriesTools.times","text":"times(P::Process; transient::Bool=false)\n\nReturn the time indices of a Process, optionally including the transient.\n\n\n\n\n\n","category":"method"}]
}
