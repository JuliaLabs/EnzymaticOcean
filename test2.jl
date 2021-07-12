include("convective_adjustment_utils.jl")

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


# Function test
convective_diffusivity = 10
background_diffusivity = 1e-4

Δt = 0.2 * grid.Δz^2 / convective_diffusivity
Nt = ceil(Int, stop_time / Δt)

convective_diffusivity = adapt(typeof(T0), [convective_diffusivity])
background_diffusivity = adapt(typeof(T0), [background_diffusivity])

T = convect!(T0, grid, background_diffusivity, convective_diffusivity, surface_flux, Δt)
