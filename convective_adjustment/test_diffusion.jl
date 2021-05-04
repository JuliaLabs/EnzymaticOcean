#####
##### Eyeball test for convective adjustment code: diffusion of a Gaussian
#####

using UnicodePlots

include("convective_adjustment_utils.jl")

# Problem parameters
grid = RegularGrid(Nz=256, Lz=1)

surface_flux = 0
convection_diffusivity = background_diffusivity = 1.0
time_step = 0.2 * grid.Δz^2
stop_time = 1e-2
@show Nt = ceil(Int, stop_time / time_step)

# Problem data
temperature = zeros(grid.Nz, Nt)

# Set initial condition
z = zᶜ(grid)
z₀ = mean(z)
δz = 0.05

initial_temperature = @. exp(-(z - z₀)^2 / 2δz^2)

temperature[:, 1] .= initial_temperature
lineplot(temperature[:, 1], z, ylim=(-grid.Lz, 0))

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

pl = lineplot(temperature[:, 1], z, ylim=(-grid.Lz, 0), name="initial condition")
lineplot!(pl, temperature[:, end], z, name="final state")
