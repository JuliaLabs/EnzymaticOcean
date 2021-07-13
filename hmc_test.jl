# Dynamic HMC with Enzyme

# HMC Imports
using TransformVariables, LogDensityProblems, DynamicHMC, DynamicHMC.Diagnostics
using MCMCDiagnostics
using Parameters, Statistics, Random

# EnzymaticOcean Imports
include("convective_adjustment_utils.jl")


"""
Setup of a toy problem
"""
struct ToyProblem
    """Total number of draws?"""
    n::Int
end

"""
Function to construct the logdensity we work with
"""
function (problem::ToyProblem)(θ)
    # unpack the parameters to construct the bimodal normal distributions

    # bimodal normal distribution over the convection_diffusivity & background_diffusivity
end

"""
Convective adjustment forward-model to go here
"""




# Initialization of problem
grid = RegularGrid(Nz=32, Lz=128)
const stop_time = 1800
if USE_CPU
    T0 = zeros(Float64, grid.Nz)
else
    T0 = CUDA.zeros(Float64, grid.Nz)  # Pre-alloc
end
z = zᶜ(grid)
temperature_gradient = 1e-4
surface_flux = 1e-4
T0 .= 20 .+ temperature_gradient .* z


# 1. Transform the problem domain into the appropriate space with TransformLogDensity
# 2. Apply the gradients for this transformed mapping






