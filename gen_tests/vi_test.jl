#=
Variational Inference Test for the Integration with Gen.
=#

include("../convective_adjustment_utils.jl")

using Gen
using Zygote

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
    _, pullback = ChainRulesCore.rrule_via_ad(Zygote.ZygoteRuleConfig(), model, args...)
    #_, pullback = rrule(model, args...)  # Possible Hack: Run this in apply
    _, _, dTgrad, dκᶜ, dκᵇ = pullback(dT)
    return (nothing, nothing, dTgrad, dκᶜ, dκᵇ)
end
Gen.has_argument_grads(::ConvectGF) = (false, false, true, true, true)


# Define the base model for the convective adjustment
function model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
    
    # Calculate Δt & Nt
    max_convective_diffusivity = 14  # NOTE: To stabilize the time-stepping
    Δt = 0.2 * grid.Δz^2 / max_convective_diffusivity
    Nt = ceil(Int, stop_time / Δt)

    # Wrap into arrays for Enzyme
    convective_diffusivity = adapt(typeof(T), [convective_diffusivity])
    background_diffusivity = adapt(typeof(T), [background_diffusivity])

    prev_T = copy(T)

    for _ in 2:Nt
        T = convect!(prev_T, grid, background_diffusivity, convective_diffusivity, surface_flux, Δt)
        prev_T = copy(T)
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

    T = @trace(convectgf(grid, surface_flux, T, convective_diffusivity, background_diffusivity), :model)
    return T
end


# Debugging
Gen.gradient(convectgf, (grid, surface_flux, T0, 10.0, 1e-4), nothing, T0)


# Approximation of the model
@gen function approx()
    @param convective_diffusivity_mu::Float64
    @param convective_diffusivity_log_std::Float64
    @param background_diffusivity_mu::Float64
    @param background_diffusivity_log_std::Float64
    @trace(normal(convective_diffusivity_mu, exp(convective_diffusivity_log_std)), :convective_diffusivity)
    @trace(normal(background_diffusivity_mu, exp(background_diffusivity_log_std)), :background_diffusivity)
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
ys = dataset_generation(50)  # NOTE: Usage of 500 sample points is arbitrary here


# Inference program to perform variational inference
# on the convective adjustment generative model
function variational_inference(model, grid, surface_flux, T, ys)

    # Initialize the black-box variational inference parameters
    init_param!(approx, :convective_diffusivity_mu, 0.)
    init_param!(approx, :convective_diffusivity_log_std, 0.)
    init_param!(approx, :background_diffusivity_mu, 0.)
    init_param!(approx, :background_diffusivity_log_std, 0.)

    # Create the choice map to model addresses to observed
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Input arguments
    args = (grid, surface_flux, T)

    # Perform gradient descent updates
    model_update = ParamUpdate(FixedStepGradientDescent(0.002), convective_adjustment)
    approx_update = ParamUpdate(FixedStepGradientDescent(0.0001), approx)

    # Run Black-Box Variational Inference (BBVI)
    (elbo_estimate, traces, elbo_history) = Gen.black_box_vi!(convective_adjustment, args, model_update, observations, approx, (), approx_update;
        iters=1, samples_per_iter=100, verbose=true)
    return traces
end

# Run the inference routine
#traces = variational_inference(model, grid, surface_flux, T0, ys)

# Further analysis does yet have to be finalized