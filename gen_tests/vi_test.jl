#=
Variational Inference Test for the Integration with Gen.
=#

include("../convective_adjustment_utils.jl")

using Gen

# Initial Conditions & Global Parameters
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


# Custom gradient definition to use Enzyme
struct ConvectGF <: CustomGradientGF{Vector{Float64}} end

Gen.apply(::ConvectGF, args) = model(args...)
function Gen.gradient(::ConvectGF, args, retval, retgrad)
    dT = retgrad
    _, pullback = rrule(model, args...)  # Possible Hack: Run this in apply
    _, _, dTgrad, dκᶜ, dκᵇ = pullback(dT)

    # Debugging print-out
    @show dkᶜ, dκᵇ
    return (nothing, nothing, dTgrad, dκᶜ, dκᵇ)
end
Gen.has_argument_grads(::ConvectGF) = (false, false, true, true, true)


# Define the base model for the convective adjustment
function model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
    
    # Calculate Δt & Nt
    # TODO: Use the max_convective_diffusivity here to stabilize the time-stepping?
    Δt = 0.2 * grid.Δz^2 / convective_diffusivity  # TODO: Implement a fallback here to see whether this fixes the instabilities

    # Debugging print-out
    @show Δt, convective_diffusivity
    Nt = ceil(Int, stop_time / Δt)

    # Wrap into arrays for Enzyme
    convective_diffusivity = adapt(typeof(T), [convective_diffusivity])
    background_diffusivity = adapt(typeof(T), [background_diffusivity])

    for _ in 2:Nt
        T = convect!(T, grid, background_diffusivity, convective_diffusivity, surface_flux, Δt)
    end
    return T
end

const convectgf = ConvectGF()

# Generative model in the probabilistic programming sense
# for the convective adjustment model
@gen function convective_adjustment(grid, surface_flux, T)

    # Construct priors (random debug priors for now)
    convective_diffusivity = @trace(uniform(6.0, 14.0), :convective_diffusivity)
    background_diffusivity = @trace(normal(1e-4, 3e-5), :background_diffusivity)

    # Debugging print-outs
    #@show convective_adjustment, background_diffusivity

    T = @trace(convectgf(grid, surface_flux, T, convective_diffusivity, background_diffusivity), :model)
    return T
end


# Approximation of the model
@gen function approx()
    @param convective_diffusivity_lb::Float64
    @param convective_diffusivity_ub::Float64
    @param background_diffusivity_mu::Float64
    @param background_diffusivity_std::Float64
    @trace(uniform(convective_diffusivity_lb, convective_diffusivity_ub), :convective_diffusivity)
    @trace(normal(background_diffusivity_mu, background_diffusivity_std), :background_diffusivity)
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

# Generate test set for VI
ys = dataset_generation(50)  # NOTE: Used to use 500 here, but downgraded this to 50 for debugging purposes


# Inference program to perform variational inference
# on the convective adjustment generative model
function variational_inference(model, grid, surface_flux, T, ys)

    # Initialize the black-box variational inference parameters
    init_param!(approx, :convective_diffusivity_lb, 4.)
    init_param!(approx, :convective_diffusivity_ub, 16.)
    init_param!(approx, :background_diffusivity_mu, 2e-4)
    init_param!(approx, :background_diffusivity_std, 8e-5)

    # Create the choice map to model addresses to observed
    # values ys[i]
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Input arguments
    args = (grid, surface_flux, T)

    # Perform gradient descent updates
    # See: https://github.com/JuliaLabs/EnzymaticOcean/pull/7/commits/eb28c4a1b2e3f5bb63f5349ec467d6c82ecce64e
    # for further details on the custom gradient descent update
    update = ParamUpdate(GradientDescent(0.5, 900), approx)

    # Debug function
    function print_traces(iter, traces, elbo_estimate)
        @info "" traces[end] Gen.get_score(traces[end]) iter elbo_estimate
    end

    # Perform variational inference to find the most likely simulation trace
    # consistent with our observations
    #(log_weight, var_trace, model_trace) = Gen.single_sample_gradient_estimate!(
    #            approx, (),
    #            convective_adjustment, args, observations, 1/100)

    var_trace = Gen.simulate(approx, ())
    constraints = Gen.merge(observations, get_choices(var_trace))
    (model_trace, model_log_weight) = Gen.generate(convective_adjustment, args, constraints)

    @show model_trace
    @show model_log_weight


    #(elbo_estimate, traces, elbo_history) = Gen.black_box_vi!(convective_adjustment, args, observations, approx, (), update;
    #    iters=2000, samples_per_iter=100, verbose=true, callback=print_traces)
    ##for trace in traces
    ##    @show trace, Gen.get_score(trace)
    ##end
    #return traces
end

# Run the inference routine
traces = variational_inference(model, grid, surface_flux, T0, ys)

# Further analysis does yet have to be finalized