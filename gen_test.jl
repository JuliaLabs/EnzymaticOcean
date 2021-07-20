include("convective_adjustment_utils.jl")

using Gen

# Initial Conditions
grid = RegularGrid(Nz=32, Lz=128)
const stop_time = 1800
if USE_CPU
    T0 = zeros(Float64, grid.Nz)
else
    T0 = CUDA.zeros(Float64, grid.Nz)
end
z = zᶜ(grid)
temperature_gradient = 1e-4
surface_flux = 1e-4
T0 .= 20 .+ temperature_gradient .* z


# Draw up a perfect model
function model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
    # Calculate Δt & Nt
    Δt = 0.2 * grid.Δz^2 / convective_diffusivity
    Nt = ceil(Int, stop_time / Δt)

    # Wrap into arrays for Enzyme
    convective_diffusivity = adapt(typeof(T), [convective_diffusivity])
    background_diffusivity = adapt(typeof(T), [background_diffusivity])

    for _ in 2:Nt
        T = convect!(T, grid, background_diffusivity, convective_diffusivity, surface_flux, Δt)
    end
    return T
end


# Convective adjustment function
@gen function convective_adjustment(grid, surface_flux, T)
    # Construct priors (debug priors for now) -> Do they also need to be local?
    convective_diffusivity = @trace(normal(10, 2), :convective_diffusivity)
    background_diffusivity = @trace(normal(1e-4, 3e-5), :background_diffusivity)

    # Do I need to genify this at this point?
    model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
end


# Generate the required number of datapoints using the perfect model
# while varying the temperature gradient, by sampling from a normal
# distribution
function dataset_generation(datapoints::Int)

    # Define local convective diffusivity & background diffusivity
    local true_convective_diffusivity = 10
    local true_background_diffusivity = 1e-4

    # Vary T0 by sampling from a normal distribution over the temperature gradient
    function sample_TStart()
        T_start = 20 .+ normal(1e-4, 2e-4) .* z
        return T_start
    end
    
    # Generate the data with a list comprehension
    test_data = [
        model(grid, surface_flux, sample_TStart(), true_convective_diffusivity, true_background_diffusivity) for _ in 1:datapoints
    ]
    return test_data
end

# Generate test set for importance sampling
ys = dataset_generation(500)


#=
Clean code aboce this point
=#

# Trace test
#planner_params = PlannerParams(300, 3.0, 2000, 1.)
#(trace, _) = Gen.generate(agent_model, (scene, dt, num_ticks, planner_params));
#choices = Gen.get_choices(trace)

# Might want a trace to dict func, i.e. to arrive at a syntax akin to
#for i=1:1000
#    trace = do_inference_agent_model(scene, dt, num_ticks, planner_params, start, measurements, 50)
#    putTrace!(viz, i, trace_to_dict(trace))
#end


# Generate the perfect data -> QMC/LHS across the space of temperature gradient
#for i in 1:Nt
#    # Fixed real value
#    local convective_diffusivity = 10
#    local background_diffusivity = 1e-4
#
#    # LHS of temperature_gradient
#    T_start = 20 .+ temporary_temp_gradient .* z
#    t_data[i] = model(grid, surface_flux, T_start, convective_diffusivity, background_diffusivity)
#end


# Train custom data-driven proposal on the generated data.
# See: https://github.com/probcomp/gen-quickstart/blob/master/tutorials/Data-Driven%20Proposals%20in%20Gen.ipynb


#function importance_sampling_inference(model, grid, surface_flux, T, ys, amount_of_computation)
#
#    observations = Gen.choicemap()
#    for (i, y) in enumerate(ys)
#        observations[(:y, i)] = y
#    end
#
#    # Perform importance sampling to find the most likely simulation trace
#    # consistent with our observations
#    (trace, lml_est) = Gen.importance_sampling(model, (grid, surface_flux, T), observations, amount_of_computation)
#    return trace, lml_est
#end