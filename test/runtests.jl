using NonstationaryProcessesBase
using TimeseriesTools
using Test

@testset "NonstationaryProcessesBase" begin
    function noisySine(P::Process)
        seed(P.solver_rng)
        sol = [sin(t) + parameter_function(P)(t)[1] * randn() for t in P.transient_t0:P.savedt:P.tmax]
    end

    sim = @test_nowarn Process(
        process=noisySine,
        X0=[0.0],
        parameter_profile=unitStep,
        parameter_profile_parameters=(100.0, 0.0, 0.1), # (threshold, baseline, stepHeight)
        transient_t0=-10.0,
        dt=0.01,
        savedt=0.01,
        tmax=200.0,
        alg=nothing,
        solver_opts=Dict())

    @test getvaryingparameters(sim) == [1]
    @test parameter_function(sim).(0:0.1:200) == unitStep(100.0, 0.0, 0.1).(0:0.1:200)
    @test parameter_functions(sim).(0:0.1:200) == unitStep(100.0, 0.0, 0.1).(0:0.1:200)
    @test parameterseries(sim) == unitStep(100.0, 0.0, 0.1).(0:0.01:200)
    @test timeseries(simulate(sim)) == timeseries(sim)
    @test_nowarn saveTimeseries!(sim, tempdir(); transient=true)
    @test times(sim; transient=true) == -10.0:0.01:200.0
    @test_nowarn trimtransient!(sim)
    @test times(sim; transient=true) == 0:0.01:200.0
    @test_nowarn getparameter_profile(sim)
    @test_nowarn updateparam(sim, 1, unitBump, ((1, 2), 0.0, 1.0, 1.0))

    x = @test_nowarn timeseries(sim)
    @test x isa RegularTimeSeries

end
