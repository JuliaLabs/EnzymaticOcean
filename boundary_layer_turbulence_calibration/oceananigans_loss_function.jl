using Plots
using Statistics
using Printf
using JLD2

using Oceananigans
using Oceananigans.Units
using Oceananigans.TurbulenceClosures: VerticallyImplicitTimeDiscretization, TKEBasedVerticalDiffusivity
using Oceananigans.TurbulenceClosures: TKESurfaceFlux, RiDependentDiffusivityScaling

function loss_function(Cᴰ,
                       Cᵇ, 
                       Cᴷu⁻,
                       Cᴷu⁺,
                       Cᴷc⁻,
                       Cᴷc⁺,
                       Cᴷe⁻,
                       Cᴷe⁺,
                       CᴷRiʷ,
                       CᴷRiᶜ,
                       Cᵂu★,
                       CᵂwΔ)

    surface_model = TKESurfaceFlux(; Cᵂu★, CᵂwΔ)
    diffusivity_scaling = RiDependentDiffusivityScaling(; Cᴷu⁻, Cᴷu⁺, Cᴷc⁻, Cᴷc⁺, Cᴷe⁻, Cᴷe⁺, CᴷRiʷ, CᴷRiᶜ)

    closure = TKEBasedVerticalDiffusivity(dissipation_parameter = Cᴰ,
                                          mixing_length_parameter = Cᵇ,
                                          surface_model = surface_model,
                                          diffusivity_scaling = diffusivity_scaling,
                                          time_discretization = VerticallyImplicitTimeDiscretization())

    # Some important parameters of the model-data comparison
    initial_save_point = 10 # save point from data to use for initial condition
    target_save_time_ish = 24hour # final time for model run + time for model/data comparison

    #####
    ##### File wrangling / data preparation
    #####
    
    filename = "boundary_layer_turbulence_LES_data.jld2"

    file = jldopen(filename)

    Qᵀ = file["parameters/temperature_flux"]
    Qᵘ = file["parameters/momentum_flux"]
     f = file["parameters/coriolis_parameter"]
     g = file["parameters/gravitational_acceleration"]
     α = file["parameters/thermal_expansion_coefficient"]

    Nz = file["grid/Nz"]
    Lz = file["grid/Lz"]

    iterations = parse.(Int, keys(file["timeseries/t"]))
    times = [file["timeseries/t/$iter"] for iter in iterations]

    # Extract initial condition
    initial_iter = iterations[initial_save_point]
    initial_time = times[initial_save_point]

    T_initial = file["timeseries/T/$initial_iter"][:, :, :] 
    u_initial = file["timeseries/u/$initial_iter"][:, :, :] 
    v_initial = file["timeseries/v/$initial_iter"][:, :, :] 
    e_initial = file["timeseries/e/$initial_iter"][:, :, :] 

    # Extract "target" data for comparing with model output
    target_save_point = findfirst(t -> t > target_save_time_ish, times)
    target_iter = iterations[target_save_point]
    target_time = times[target_save_point]

    T_target = file["timeseries/T/$target_iter"][:, :, :]
    u_target = file["timeseries/u/$target_iter"][:, :, :]
    v_target = file["timeseries/v/$target_iter"][:, :, :]
    e_target = file["timeseries/e/$target_iter"][:, :, :] 

    close(file)

    #####
    ##### Model setup
    #####

    grid = RegularRectilinearGrid(size=Nz, z=(-Lz, 0), topology=(Flat, Flat, Bounded))

    u_bcs = UVelocityBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵘ))
    T_bcs = TracerBoundaryConditions(grid; top = FluxBoundaryCondition(Qᵀ))

    coriolis = FPlane(f=f)

    buoyancy = SeawaterBuoyancy(constant_salinity=true, gravitational_acceleration=g,
                                equation_of_state=LinearEquationOfState(α=α))

    model = HydrostaticFreeSurfaceModel(grid = grid,
                                        tracers = (:T, :e),
                                        buoyancy = buoyancy,
                                        coriolis = coriolis,
                                        boundary_conditions = (u=u_bcs, T=T_bcs),
                                        closure = closure)

    # Initialize model
    model.clock.time = initial_time
    model.velocities.u .= u_initial
    model.velocities.v .= v_initial
    model.tracers.T .= T_initial
    model.tracers.e .= e_initial

    simulation = Simulation(model, Δt = 10.0, stop_time = target_time)

    # The following line quits julia and emits
    # FATAL ERROR: Symbol "__nv_fmin"not found
    run!(simulation)

    z = znodes(model.tracers.T)

    # Visualize...
    u_view = view(interior(model.velocities.u), 1, 1, :)
    v_view = view(interior(model.velocities.v), 1, 1, :)
    T_view = view(interior(model.tracers.T), 1, 1, :)
    e_view = view(interior(model.tracers.e), 1, 1, :)

    u_plot = plot(u_initial, z, linewidth = 2, label = "u, t = 0", xlabel = "Velocity (m s⁻¹)", ylabel = "z (m)", legend=:bottomright)
    plot!(u_plot, v_initial, z, linewidth = 1, label = "v, t = 0")

    T_plot = plot(T_inital, z, linewidth = 2, label = "t = 0", xlabel = "Temperature (ᵒC)", ylabel = "z (m)", legend=:bottomright)
    e_plot = plot(e_inital, z, linewidth = 2, label = "t = 0", xlabel = "TKE (m² s⁻²)", ylabel = "z (m)", legend=:bottomright)

    plot!(u_plot, u_view,   z, linewidth = 2, linestyle=:dash, label = @sprintf("Model, u, t = %s", prettytime(model.clock.time)))
    plot!(u_plot, v_view,   z, linewidth = 1, linestyle=:dash, label = @sprintf("Model, v, t = %s", prettytime(model.clock.time)))
    plot!(u_plot, u_target, z, linewidth = 3, alpha = 0.6,     label = @sprintf("LES data, u, t = %s", prettytime(model.clock.time)))
    plot!(u_plot, v_target, z, linewidth = 2, alpha = 0.6,     label = @sprintf("LES data, v, t = %s", prettytime(model.clock.time)))

    plot!(T_plot, T_view,   z, linewidth = 2, linestyle=:dash, label = @sprintf("Model, t = %s", prettytime(model.clock.time)))
    plot!(T_plot, T_target, z, linewidth = 3, alpha = 0.6, label = @sprintf("LES data, t = %s", prettytime(model.clock.time)))

    plot!(e_plot, e_view,   z, linewidth = 2, linestyle=:dash, label = @sprintf("Model, t = %s", prettytime(model.clock.time)))
    plot!(e_plot, e_target, z, linewidth = 3, alpha = 0.6, label = @sprintf("LES data, t = %s", prettytime(model.clock.time)))

    eT_plot = plot(T_plot, e_plot, layout=(1, 2), size=(1200, 600))

    error = mean((T_view .- T_target).^2)

    return error, eT_plot
end

# Defaults
Cᴰ    = 2.91
Cᵇ    = 1.16
Cᴷu⁻  = 0.15
Cᴷu⁺  = 0.73
Cᴷc⁻  = 0.40
Cᴷc⁺  = 1.77
Cᴷe⁻  = 0.13
Cᴷe⁺  = 1.22
CᴷRiʷ = 0.72
CᴷRiᶜ = 0.76
Cᵂu★  = 3.62
CᵂwΔ  = 1.31

error, eT_plot = loss_function(Cᴰ, Cᵇ, Cᴷu⁻, Cᴷu⁺, Cᴷc⁻, Cᴷc⁺, Cᴷe⁻, Cᴷe⁺, CᴷRiʷ, CᴷRiᶜ, Cᵂu★, CᵂwΔ)  

@show error
display(eT_plot)
