#include("setup.jl")

using CUDA
const USE_CPU = !CUDA.has_cuda_gpu()

#using KernelAbstractions.Extras: @unroll
using KernelAbstractions
using KernelGradients
using CUDAKernels
using Statistics
using InteractiveUtils
using Adapt

using Enzyme
using ChainRulesCore

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

diffusivity(i, grid, κᵇ, κᶜ, T) = Base.ifelse(∂zᶠ(i, grid, T) < 0, κᶜ, κᵇ)

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
@kernel function kern_convect!(T, Tprev, grid, κᵇ, κᶜ, surface_flux, Δt)
    i = @index(Global, Linear)
    T[i] = Tprev[i] - Δt * ∂zᶜ(i, grid, diffusive_flux, κᵇ[1], κᶜ[1], surface_flux, Tprev)
    nothing
end

function convect!(Tprev, grid, κᵇ, κᶜ, surface_flux, Δt#=, prev=#)
    T = similar(Tprev)
    if T isa Array
        kern = kern_convect!(CPU(), 256)
    else
        kern = kern_convect!(CUDADevice())
    end
    event = kern(T, Tprev, grid, κᵇ, κᶜ, surface_flux, Δt; ndrange = size(T, 1)#=; dependencies=prev=#)
    wait(event) # either device wait or propagate event
    return T
end

Base.size(x::Enzyme.Duplicated, i) = size(x.val, i)

function gradconvect!(Tprev, dT, grid, κᵇ, κᶜ, surface_flux, Δt #=, prev=#)
    if Tprev isa Array
        kern = kern_convect!(CPU(), 256)
    else
        kern = kern_convect!(CUDADevice())
    end
    kern′ = autodiff(kern)
    T′ = Duplicated(similar(Tprev), dT)
    Tprev′ = Duplicated(Tprev, zero(Tprev))
    κᵇ′ = Duplicated(κᵇ, zero(κᵇ))
    κᶜ′ = Duplicated(κᶜ, zero(κᶜ))
    event = kern′(T′, Tprev′,grid, κᵇ′, κᶜ′, surface_flux, Δt; ndrange = size(Tprev, 1)#=; dependencies=prev=#)
    wait(event) # either device wait or propagate event
    Tprev′.dval, κᵇ′.dval, κᶜ′.dval #=, event=#
end

import ChainRulesCore: rrule
function rrule(::typeof(convect!), Tprev, grid, κᵇ, κᶜ, surface_flux, Δt)
    function pullback(dT)
        dTprev, dκᵇ, dκᶜ = gradconvect!(Tprev, dT, grid, κᵇ, κᶜ, surface_flux, Δt#=, prevEvent=#)
        return (NO_FIELDS, dTprev, NoTangent(), dκᵇ, dκᶜ, NoTangent(), NoTangent())
    end
    return (convect!(Tprev, grid, κᵇ, κᶜ, surface_flux, Δt), pullback)
end
