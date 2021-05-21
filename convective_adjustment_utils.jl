include("setup.jl")

using CUDA
const USE_CPU = !CUDA.has_cuda_gpu()

using KernelAbstractions.Extras: @unroll
using Statistics
using InteractiveUtils
using Adapt

using Enzyme

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
        return zero(surface_flux)
    elseif i < grid.Nz + 1 # interior
        return - diffusivity(i, grid, κᵇ, κᶜ, T) * ∂zᶠ(i, grid, T)
    else # top boundary
        return surface_flux
    end
end

"""
Conduct forward run with no error / loss calculation.
The initial state must be stored in T[:, 1]. """
@kernel function kern_convect!(T, grid, κᵇ, κᶜ, surface_flux, Δt, t)
    i = @index(Global, Linear)
    T[i, t] = T[i, t-1] - Δt * ∂zᶜ(i, grid, diffusive_flux, κᵇ[1], κᶜ[1], surface_flux, view(T, :, t-1))
    nothing
end

@kernel function gradkern_convect!(T, dT, grid, κᵇ, dκᵇ, κᶜ, dκᶜ, surface_flux, Δt, t)
    @static if USE_CPU
        func = cpu_kern_convect!
    else
        func = gpu_kern_convect!
    end
    Enzyme.autodiff_no_cassette(func, __ctx__,
        Duplicated(T, dT), grid, Duplicated(κᵇ, dκᵇ), Duplicated(κᶜ, dκᶜ), surface_flux, Δt, t)
end

function convect!(T, grid, κᵇ, κᶜ, surface_flux, Δt, t; prev=nothing)
    if T isa Array
        kern = kern_convect!(CPU(), 256)
    else
        kern = kern_convect!(CUDADevice())
        if prev === nothing
            prev = Event(CUDADevice())
        end
    end
    kern(T, grid, κᵇ, κᶜ, surface_flux, Δt, t; ndrange = size(T, 1), dependencies=prev)
end

Base.size(x::Enzyme.Duplicated, i) = size(x.val, i)

function gradconvect!(T, dT, grid, κᵇ, dκᵇ, κᶜ, dκᶜ, surface_flux, Δt, t; prev=nothing)
    if T isa Array
        kern = gradkern_convect!(CPU(), 256)
    else
        kern = gradkern_convect!(CUDADevice())
        if prev === nothing
            prev = Event(CUDADevice())
        end
    end
    kern(T, dT, grid, κᵇ, dκᵇ, κᶜ, dκᶜ, surface_flux, Δt, t; ndrange = size(T, 1), dependencies=prev)
end

