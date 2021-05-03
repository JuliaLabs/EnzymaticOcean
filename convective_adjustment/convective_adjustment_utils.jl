using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using CUDAKernels
using Statistics

# Grid
struct RegularGrid
    Nz :: Int
    Lz :: Float64
    Δz :: Float64
end

RegularGrid(; Nz, Lz) = RegularGrid(Nz, Lz, Lz / Nz)   

zᶜ(grid::RegularGrid) = (-grid.Nz+1/2:-1/2) * grid.Δz

# Derivatives
@inline ∂zᶠ(i, grid, T::AbstractArray) = @inbounds (T[i] - T[i-1]) / grid.Δz
@inline ∂zᶜ(i, grid, F::AbstractArray) = @inbounds (F[i+1] - F[i]) / grid.Δz

@inline ∂zᶠ(i, grid, f::F, args...) where F<:Function = (f(i, grid, args...) - f(i-1, grid, args...)) / grid.Δz
@inline ∂zᶜ(i, grid, f::F, args...) where F<:Function = (f(i+1, grid, args...) - f(i, grid, args...)) / grid.Δz

diffusivity(i, grid, κᵇ, κᶜ, T) = ifelse(∂zᶠ(i, grid, T) < 0, κᶜ, κᵇ)

function diffusive_flux(i, grid, κᵇ, κᶜ, surface_flux, T)
    if i < 2 # bottom at i = 1
        return 0
    elseif i < grid.Nz + 1 # interior
        return - diffusivity(i, grid, κᵇ, κᶜ, T) * ∂zᶠ(i, grid, T)
    else # top boundary
        return surface_flux
    end
end

"""
Conduct forward run with no error / loss calculation.
The initial state must be stored in T[:, 1]. """
@kernel function convect!(T, grid, κᵇ, κᶜ, surface_flux, Δt, Nt)
    i = @index(Global)

    Tⁿ⁻¹(n) = view(T, :, n-1)

    @unroll for n = 2:Nt
        T[i, n] = T[i, n-1] - Δt * ∂zᶜ(i, grid, diffusive_flux, κᵇ, κᶜ, surface_flux, view(T, :, n-1))
        @synchronize
    end
end

"""Conduct forward run and assess error against T_truth."""
@kernel function uncertain_convect!(mean_error,      # Output (a 0D array containing a scalar)
                                    pointwise_error, # Output (a 0D array containing a scalar)
                                    grid,            # Represents the physical domain
                                    κᵇ,              # Free parameter: "background" diffusivity
                                    κᶜ,              # Free parameter: diffusiity in convective plumes
                                    surface_flux,    # Fixed parameter: surface boundary condition
                                    Δt,              # Fixed parameter: time increment
                                    Nt,              # Fixed parameter: of time-steps
                                    T_truth,         # Data: Nz by Nt matrix of vertical-temporal temperature profile
                                    T)               # Scratch vector that stores the "current" temperature profile
    i = @index(Global)

    @unroll for n = 2:Nt
        T[i, n] = T[i, n-1] - Δt * ∂zᶜ(i, grid, diffusive_flux, κᵇ, κᶜ, surface_flux, view(T, :, n-1))
        @synchronize
    end

    @unroll for n = 1:Nt
        pointwise_error[i, n] = (T[i, n] - T_truth[i, n])^2
    end

    @synchronize

    if i == 1
        mean_error[1] = mean(pointwise_error)
    end
end
