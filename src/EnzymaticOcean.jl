module EnzymaticOcean

export convective_model, convectgf, RegularGrid, zᶜ, dataset_generation

include("convective_adjustment.jl")

const stop_time = 1800 # pass in

# Define the base model for the convective adjustment
function convective_model(grid, surface_flux, T, convective_diffusivity, background_diffusivity)
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

# Dataset Generation with different priors
function dataset_generation(datapoints::Int, grid, surface_flux, T)

    # Initialize the dataset Vector
    data = Vector{Vector{Float64}}(undef, datapoints)
    xs = Vector{Vector{Float64}}(undef, datapoints)

    # Generate the requisite number of datapoints
    for i in 1:datapoints

        # Probability distributions over the two parameters
        local true_convective_diffusivity = normal(10, 2)
        local true_background_diffusivity = normal(1e-4, 2e-5)

        # Generate the datapoint and append to the dataset
        xs[i] = T
        data[i] = convective_model(grid, surface_flux, T, true_convective_diffusivity, true_background_diffusivity)
    end
    return xs, data
end

include("gen.jl")

end