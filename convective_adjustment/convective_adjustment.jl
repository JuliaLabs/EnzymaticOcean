using UnicodePlots

include("convective_adjustment_utils.jl")

# Problem parameters
grid = RegularGrid(Nz=32, Lz=128)

convection_diffusivity = 10.0  # [m² s⁻¹]
background_diffusivity = 1e-4 # [m² s⁻¹]
surface_flux = 1e-4 # [m s⁻¹ ᶜC]
@show time_step = 0.2 * grid.Δz^2 / convection_diffusivity # s
stop_time = 1800 # seconds
@show Nt = ceil(Int, stop_time / time_step)

# Problem data
temperature = zeros(grid.Nz, Nt)

# Set initial condition
z = zᶜ(grid)
temperature_gradient = 1e-4
temperature[:, 1] .= 20 .+ temperature_gradient .* z
lineplot(z, temperature[:, 1])

let kernel = convect!(CPU())
    event = kernel(temperature,
                   grid,
                   background_diffusivity,
                   convection_diffusivity,
                   surface_flux,
                   time_step,
                   Nt,
                   ndrange = size(temperature, 1))

    wait(event)

end

pl = lineplot(temperature[:, 1], z)
lineplot!(pl, temperature[:, end], z)

# Gradient
mean_error = [0.0]
pointwise_error = zeros(grid.Nz, Nt)
temperature_data = deepcopy(temperature)

# Parameter estimation: estimate free parameters by
# calibrading to "truth" or "data":

# Perturb free parameters
perturbed_background_diffusivity = background_diffusivity * 1.1
perturbed_convection_diffusivity = convection_diffusivity * 0.8

#=
# State estimation: produce an estimate temperature *state* (in space and time)
# by optimizing initial condition and boundary conditions.

# Corrupt initial condition
temperature[:, 1] .+= temperature_gradient * grid.Lz * 1e-1 * rand()

# Corrupt boundary condition
surface_flux = surface_flux * 1.1
=#

let kernel = uncertain_convect!(CPU())

    event = kernel(mean_error,
                   pointwise_error,
                   grid,
                   perturbed_background_diffusivity,
                   perturbed_convection_diffusivity,
                   surface_flux,
                   time_step,
                   Nt,
                   temperature_data,
                   temperature,
                   ndrange = size(temperature, 1))

    wait(event)

    @show mean_error
end
