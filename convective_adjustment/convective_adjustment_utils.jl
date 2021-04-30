using KernelAbstractions
using CUDAKernels

# Grid
struct RegularGrid
    Nz :: Int
    Lz :: Float64
    Δz :: Float64
end

RegularGrid(; Nz, Lz) = RegularGrid(Nz, Lz, Lz / Nz)   

# Derivatives
∂zᶠ(i, grid, T) = (T[i] - T[i-1]) / grid.Δz
∂zᶜ(i, grid, F) = (F[i+1] - F[i]) / grid.Δz
∂zᶜ(i, grid, F::Function, args...) = (F(i+1, grid, args...) - F(i, grid, args...)) / grid.Δz

diffusivity(i, grid, κᵇ, κᶜ, T) = ifelse(∂zᶠ(i, grid, T) < 0, κᶜ, κᵇ)

diffusive_flux(i, grid, κᵇ, κᶜ, surface_flux, T) = ifelse(i == 1, 0,
                                                   ifelse(i == grid.Nz+1, surface_flux, diffusivity(i, grid, κᵇ, κᶜ, T) * ∂zᶠ(i, grid, T))

"""Conduct forward run with no error / loss calculation."""
@kernel function convect!(T_out, κᵇ, κᶜ, surface_flux, T, grid, dt, Nt)

    i = @index(Global)

    @unroll for n = 1:Nt
        T_out[i, n] = dt * ∂zᶜ(i, grid, diffusive_flux, κᵇ, κᶜ, surface_flux, T)

        @synchronize

        T[i] = T_out[i, n]
    end
end

"""Conduct forward run and assess error against T_truth."""
@kernel function uncertain_convect!(error,                  # Output (a 0D array containing a scalar)
                                    κᵇ,  # Parameter (we want to differentiate with respect to this)
                                    κᶜ, # Parameter (we want to differentiate with respect to this)
                                    surface_flux,           # Surface boundary condition
                                    scratch,                # scratch space
                                    @Const(T_truth),        # "Truth" (used to evaluate model fidelity)
                                    T_out,                  # Stores the model output
                                    T,                      # Stores the "current" state
                                    grid,                   # Represents the physical domain
                                    dt,                     # Time increment
                                    Nt)                     # Number of time-steps
    i = @index(Global)

    @unroll for n = 1:Nt
        T_out[i, n] = dt * ∂zᶜ(i, grid, diffusive_flux, κᵇ, κᶜ, T)

        @synchronize

        T[i] = T_out[i, n]
    end

    @synchronize

    @unroll for n = 1:Nt
        scratch[i, n] = (T_out[i, n] - T_truth[i, n])^2
    end

    @synchronize

    if i == 1
        error[] = mean(error_scratch)
    end
end
