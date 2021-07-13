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


# Convective adjustment function
@gen function convective_adjustment(grid, surface_flux, T)
    # Construct priors (debug priors for now)
    convective_diffusivity = @trace(normal(10, 2), :convective_diffusivity)
    background_diffusivity = @trace(normal(1e-4, 3e-5), :background_diffusivity)

    # Calculate Δt & Nt -> Keeps adjusting all the time
    Δt = 0.2 * grid.Δz^2 / convective_diffusivity
    Nt = ceil(Int, stop_time / Δt)

    # Wrap priors into arrays for Enzyme
    convective_diffusivity = adapt(typeof(T), [convective_diffusivity])
    background_diffusivity = adapt(typeof(T), [background_diffusivity])

    for _ in 2:Nt
        T = convect!(T, grid, background_diffusivity, convective_diffusivity, surface_flux, Δt)
    end
    return T

end

convective_adjustment(grid, surface_flux, T0)



#function infer()


#end