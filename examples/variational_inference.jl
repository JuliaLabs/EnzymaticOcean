#=
Variational Inference and Variational Inference with Monte Carlo
    Objective Tests for the Integration with Gen.
=#

using EnzymaticOcean
using Gen
using CUDA

const USE_CPU = !CUDA.has_cuda_gpu()

# Initial Conditions & Global Parameters
grid = RegularGrid(Nz=32, Lz=128)
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

# Generate test set for VI
xs, ys = dataset_generation(N, grid, surface_flux, T0)

# Generative model in the probabilistic programming sense
# for the convective adjustment model
@gen function convective_adjustment(grid, surface_flux, xs)
    convective_diffusivity = @trace(gamma(10.0, 1.0), :convective_diffusivity)
    background_diffusivity = @trace(gamma(2.0, 0.0001), :background_diffusivity)

    for i in 1:N
        T = @trace(convectgf(grid, surface_flux, xs[i], convective_diffusivity, background_diffusivity), :T)
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
function vi(grid, surface_flux, xs, ys)

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
    args = (grid, surface_flux, xs)

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
function vimco(grid, surface_flux, xs, ys)

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
    args = (grid, surface_flux, xs)

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
