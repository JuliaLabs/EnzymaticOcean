#=
Variational Inference and Variational Inference with Monte Carlo
    Objective Tests for the Integration with Gen.
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
const temperature_gradient = 1e-4
const surface_flux = 1e-4
const N = 50
T0 .= 20 .+ temperature_gradient .* z

# Define the base model for the convective adjustment
function model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
    @assert convective_diffusivity >= 0.0
    @assert background_diffusivity >= 0.0

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

# Custom gradient definition to use Enzyme
struct ConvectGF <: CustomDetermGF{Vector{Float64}, Function} end
const convectgf = ConvectGF()

accepts_output_grad(::ConvectGF) = true

function Gen.apply_with_state(::ConvectGF, args)
    return ChainRulesCore.rrule_via_ad(Zygote.ZygoteRuleConfig(), model, args...) # TODO: Use Diffractor
end

function Gen.gradient_with_state(::ConvectGF, pullback, args, retgrad)
    dT = retgrad
    _, _, _, dTgrad, dκᶜ, dκᵇ = pullback(dT)
    return (nothing, nothing, dTgrad, dκᶜ, dκᵇ)
end
Gen.has_argument_grads(::ConvectGF) = (false, false, true, true, true)


# Dataset Generation with different priors
function dataset_generation(datapoints::Int)

    # Initialize the dataset Vector
    data = Vector{Vector{Float64}}(undef, datapoints)

    # Generate the requisite number of datapoints
    for i in 1:datapoints

        # Probability distributions over the two parameters
        local true_convective_diffusivity = normal(10, 2)
        local true_background_diffusivity = normal(1e-4, 2e-5)

        # Generate the datapoint and append to the dataset
        data[i] = model(grid, surface_flux, T0, true_convective_diffusivity, true_background_diffusivity)

    end
    return data
end

# Generate test set for VI
ys = dataset_generation(N)

# Generative model in the probabilistic programming sense
# for the convective adjustment model
@gen function convective_adjustment(grid, surface_flux, T0)
    convective_diffusivity = @trace(gamma(10.0, 1.0), :convective_diffusivity)
    background_diffusivity = @trace(gamma(2.0, 0.0001), :background_diffusivity)

    T = @trace(convectgf(grid, surface_flux, T0, convective_diffusivity, background_diffusivity), :T)
    for i in 1:N
        {(:y, i)} ~ broadcasted_normal(T, 0.01)
    end
end

# Approximation of the model, i.e. the proposal distribution of
# the variational inference approximates the model's posterior by
# minimizing the Karhunen-Loeve distance between the two
# distributions. 
@gen function approx()
    @param convective_diffusivity_log_shape::Float64
    @param convective_diffusivity_log_scale::Float64
    @param background_diffusivity_log_shape::Float64
    @param background_diffusivity_log_scale::Float64
    convective_diffusivity_shape = exp(convective_diffusivity_log_shape) + eps()
    convective_diffusivity_scale = exp(convective_diffusivity_log_scale) + eps()
    background_diffusivity_shape = exp(background_diffusivity_log_shape) + eps()
    background_diffusivity_scale = exp(background_diffusivity_log_scale) + eps()

    @trace(gamma(convective_diffusivity_shape, convective_diffusivity_scale), :convective_diffusivity)
    @trace(gamma(background_diffusivity_shape, background_diffusivity_scale), :background_diffusivity)
end


# Inference program to perform variational inference
# on the convective adjustment generative model
function vi(grid, surface_flux, T, ys)

    # Initialize the black-box variational inference parameters
    init_param!(approx, :convective_diffusivity_log_shape, 0.0)
    init_param!(approx, :convective_diffusivity_log_scale, 0.0)
    init_param!(approx, :background_diffusivity_log_shape, 0.0)
    init_param!(approx, :background_diffusivity_log_scale, 0.0)

    # Create the choice map to model addresses to observed
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Input arguments
    args = (grid, surface_flux, T)

    # Perform gradient descent updates
    model_update = ParamUpdate(FixedStepGradientDescent(0.0001), convective_adjustment)
    approx_update = ParamUpdate(FixedStepGradientDescent(0.0001), approx)

    # Run Black-Box Variational Inference (BBVI)
    (elbo_estimate, traces, elbo_history) = 
        Gen.black_box_vi!(convective_adjustment, args, model_update,
                          observations,
                          approx, (), approx_update;
        iters=10, samples_per_iter=10, verbose=true)
    return traces
end


# Inference program to perform variational inference
# with monte carlo objectives (VIMCO)
# on the convective adjustment generative model
function vimco(grid, surface_flux, T, ys)

    # Initialize the black-box variational inference parameters
    init_param!(approx, :convective_diffusivity_log_shape, 0.0)
    init_param!(approx, :convective_diffusivity_log_scale, 0.0)
    init_param!(approx, :background_diffusivity_log_shape, 0.0)
    init_param!(approx, :background_diffusivity_log_scale, 0.0)

    # Create the choice map to model addresses to observed
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # Input arguments
    args = (grid, surface_flux, T)

    # Perform gradient descent updates
    model_update = ParamUpdate(FixedStepGradientDescent(0.0001), convective_adjustment)
    approx_update = ParamUpdate(FixedStepGradientDescent(0.0001), approx)

    # Run Black-Box Variational Inference (BBVI)
    (elbo_estimate, traces, elbo_history) = 
        Gen.black_box_vimco!(convective_adjustment, args, model_update,
                             observations,
                             approx, (), approx_update, 10;
        iters=10, samples_per_iter=10, verbose=true)
    return traces
end

# Run the inference routines
#vi_traces = vi(grid, surface_flux, T0, ys)
#vimco_traces = vimco( grid, surface_flux, T0, ys)
