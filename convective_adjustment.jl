include("convective_adjustment_utils.jl")
using Zygote

# Problem parameters
grid = RegularGrid(Nz=32, Lz=128)

convection_diffusivity = 10.0  # [m² s⁻¹]
background_diffusivity = 1e-4 # [m² s⁻¹]
surface_flux = 1e-4 # [m s⁻¹ ᶜC]
@show time_step = 0.2 * grid.Δz^2 / convection_diffusivity # s
stop_time = 1800 # seconds
@show Nt = ceil(Int, stop_time / time_step)

# Problem data
if USE_CPU 
    temperature = zeros(Float64, grid.Nz, Nt)
else
    temperature = CUDA.zeros(Float64, grid.Nz, Nt)
end

# Set initial condition
z = zᶜ(grid)
temperature_gradient = 1e-4
temperature[:, 1] .= 20 .+ temperature_gradient .* z

function dothething(T, grid, κᵇ, κᶜ, surface_flux, Δt, Nt)
    kb = adapt(typeof(T), [κᵇ])
    kc = adapt(typeof(T), [κᶜ])
    prev=nothing
    for t in 2:Nt
        prev = convect!(T, grid, kb, kc, surface_flux, Δt, t; prev)
    end
    wait(prev)
end

function reversethething(T, dT, grid, κᵇ, κᶜ, surface_flux, Δt, Nt)
    kb  = adapt(typeof(T), [κᵇ])
    dkb = similar(kb)
    dkb .= 0

    kc  = adapt(typeof(T), [κᶜ])
    dkc = similar(kc)
    dkc .= 0
    
    prev=nothing
    for t in Nt:-1:2
       prev = gradconvect!(T, dT, grid, kb, dkb, kc, dkc, surface_flux, Δt, t; prev)
    end
    wait(prev)
    dkb[1], dkc[1]
end

dothething(temperature, grid, background_diffusivity, convection_diffusivity, surface_flux, time_step, Nt)

@show temperature[3, Nt]

# Gradient
temperature_data = temperature .+ adapt(typeof(temperature), randn(size(temperature)...))

# Calculate error using Zygote
dt, = gradient(temperature) do temperature
    mean((temperature .- temperature_data).^2)
end

# Enzyme.API.printall!(true)
# Enzyme.API.printtype!(true)

dkb, dkc = reversethething(temperature, dt, grid, background_diffusivity, convection_diffusivity, surface_flux, time_step, Nt)

@show dkb
@show dkc