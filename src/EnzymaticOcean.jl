module EnzymaticOcean

export convective_model, convectgf, RegularGrid, zᶜ

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

include("gen.jl")

end