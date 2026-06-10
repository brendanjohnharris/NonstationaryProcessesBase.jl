using NonstationaryProcessesBase
using TimeseriesBase
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
    @test x isa RegularTimeseries

end

@testset "parameter profiles" begin
    @test constant === constantParameter
    @test constantParameter(5.0)(0.0) == 5.0
    @test constantParameter(5.0)(123.0) == 5.0
    @test constantParameter()(1.0) == 0.0

    @test heaviside(-1.0) == 0
    @test heaviside(1.0) == 1
    @test heaviside(0.0, 0.5) == 0.5
    @test sigmoid(0.0) == 0.5

    # unitStep: baseline before the threshold, baseline+stepHeight after
    s = unitStep(10.0, 2.0, 3.0)
    @test s(5.0) == 2.0
    @test s(15.0) == 5.0
    @test s.d == Set([10.0])

    # unitBump: bumpHeight inside (t1, t2), baseline outside
    b = unitBump((0.0, 10.0), 0.0, 1.0)
    @test b(-1.0) == 0.0
    @test b(5.0) == 1.0
    @test b(15.0) == 0.0

    # ramp(gradient, p0, t0): p0 + gradient*(t - t0)
    @test ramp(2.0, 1.0, 0.0)(3.0) == 7.0
    @test ramp(2.0, 1.0, 0.0)(0.0) == 1.0
    # ramp(p1, p2, t1, t2): line through (t1, p1) and (t2, p2)
    @test ramp(0.0, 10.0, 0.0, 10.0)(5.0) == 5.0

    # rampInterval: ramps p1 -> p2 over [t1, t2] and saturates outside
    ri = rampInterval(2.0, 8.0, 0.0, 10.0)
    @test ri(-1.0) == 2.0
    @test ri(5.0) == 5.0
    @test ri(15.0) == 8.0

    # sineWave(period, amplitude, t0, baseline): amplitude*sin(2π/period*(t-t0)) + baseline
    sw = sineWave(4.0, 2.0, 0.0, 5.0)
    @test sw(0.0) == 5.0
    @test sw(1.0) ≈ 7.0
    @test sw(1.0) ≈ sw(5.0)          # periodic with period 4
end

@testset "Discontinuous" begin
    D = unitStep(5.0)
    @test D isa Discontinuous
    @test D.d == Set([5.0])
    @test D(0.0) == 0.0
    @test D(10.0) == 1.0
    @test D([0.0, 10.0]) == [0.0, 1.0]          # callable like a vectorised function

    # arithmetic with a scalar keeps it a Discontinuous
    @test (2.0 * unitStep(5.0))(10.0) == 2.0
    @test (unitStep(5.0) + 3.0)(0.0) == 3.0

    # arithmetic between Discontinuous merges the discontinuity sets
    D2 = unitStep(5.0) + unitStep(8.0)
    @test D2 isa Discontinuous
    @test D2.d == Set([5.0, 8.0])
    @test D2(0.0) == 0.0
    @test D2(10.0) == 2.0
end

@testset "tuplef2ftuple" begin
    # tuple of profile constructors + their params -> single vector-valued function
    f = tuplef2ftuple((constantParameter, constantParameter), ((1.0,), (2.0,)))
    @test f(0.0) == [1.0, 2.0]
    @test f(99.0) == [1.0, 2.0]

    # mixing a constant with a ramp
    g = tuplef2ftuple((constantParameter, ramp), ((1.0,), (2.0, 1.0, 0.0)))
    @test g(3.0) == [1.0, 7.0]                  # ramp(2,1,0)(3) == 7

    # a Discontinuous component propagates its discontinuity into the combined profile
    h = tuplef2ftuple((constantParameter, unitStep), ((1.0,), (5.0,)))
    @test 5.0 in h.d
end
