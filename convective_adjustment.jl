include("convective_adjustment_utils.jl")
using Flux

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
    T0 = zeros(Float64, grid.Nz)
else
    T0 = CUDA.zeros(Float64, grid.Nz)
end

# Set initial condition
z = zᶜ(grid)
temperature_gradient = 1e-4
T0 .= 20 .+ temperature_gradient .* z

struct Model{AT}
    κᵇ::AT
    κᶜ::AT
end
Model(::Type{T}, κᵇ, κᶜ) where T = Model(adapt(T, [κᵇ]), adapt(T, [κᶜ]))
Flux.@functor Model

function (m::Model)(T, grid, surface_flux, Δt, Nt)
    for _ in 2:Nt
        T = convect!(T, grid, m.κᵇ, m.κᶜ, surface_flux, Δt)
    end
    T
end

model = Model(typeof(T0), background_diffusivity, convection_diffusivity) 

# Perfect model
Tdata = model(T0, grid, surface_flux, time_step, Nt)
@show T0
@show Tdata


function fit(model, Tdata; N=10)
    opt = ADAM()
    ps = params(model)
    for _ in 1:N
        # Calculate error using Zygote
        gs = gradient(ps) do 
            T = model(T0, grid, surface_flux, time_step, Nt)
            mean((T .- Tdata).^2)
        end
        Flux.Optimise.update!(opt, ps, gs)
    end
    return model.κᵇ[1], model.κᶜ[1]
end

model = Model(typeof(T0), background_diffusivity * 1.2, convection_diffusivity * 0.7)
@show fit(model, Tdata; N = 100)

# κᵇ, κᶜ = background_diffusivity * 100, convection_diffusivity * 0.7
# o1 = mean( (model(T0, grid, κᵇ, κᶜ, surface_flux, time_step, Nt) .- Tdata).^2 )
# κᵇ += 1e-6
# o2 = mean( (model(T0, grid, κᵇ, κᶜ, surface_flux, time_step, Nt) .- Tdata).^2 )
# @show (o2 - o1) / 1e-6
