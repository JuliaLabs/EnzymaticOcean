#=
Importance Sampling Test for the Integration with the Probabilistic
    Programming System Gen.
=#

include("convective_adjustment_utils.jl")

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


# Define the base model for the convective adjustment
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


# Generative model in the probabilistic programming sense
# for the convective adjustment model
@gen function convective_adjustment(grid, surface_flux, T)

    # Construct priors (random debug priors for now)
    convective_diffusivity = @trace(normal(10, 2), :convective_diffusivity)
    background_diffusivity = @trace(normal(1e-4, 3e-5), :background_diffusivity)

    # Do I need to genify this at this point?
    T = model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
    return T
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


# Inference program to perform importance sampling
# on the convective adjustment generative model
# ---> We are not using gradients here!! <---
function importance_sampling_inference(model, grid, surface_flux, T, ys, amount_of_computation)

    # Create the choice map to model addresses to observed
    # values ys[i]
    observations = Gen.choicemap()
    for (i, y) in enumerate(ys)
        observations[(:y, i)] = y
    end

    # In line with Gen's nomenclature we write our inputs as xs
    xs = (grid, surface_flux, T)

    # Perform importance sampling to find the most likely simulation trace
    # consistent with our observations
    (trace, _) = Gen.importance_resampling(model, xs, observations, amount_of_computation)
    return trace
end

# Run the inference routine
trace = importance_sampling_inference(convective_adjustment, grid, surface_flux, T0, ys, 200)


# Further analysis does yet have to be finalized