using KernelAbstractions
using CUDAKernels
using CUDA

@kernel function time_step!(x, k, Δt, N)
    I = @index(Global)
    for _ in 1:N
        @inbounds x[I] = x[I] + Δt * k * x[I]
    end
end

"""
    simulate_exponential_growth(; x₀, k, Δt, N, device)

Simulates exponential growth according to the ODE

    dx/dt = k * x,  x(t = 0) = x₀

using forward Euler time stepping with time step `Δt` for `N` iterations.
"""
function simulate_exponential_growth(; x₀, k, Δt, N, device)

    if device isa CPU
        x = [x₀]
    elseif device isa CUDADevice
        x = CuArray([x₀])
    end

    time_step_kernel! = time_step!(device, 1)
    event = time_step_kernel!(x, k, Δt, N, ndrange=1)
    wait(event)

    return CUDA.@allowscalar x[1]
end

# 1 kernel launch
@show simulate_exponential_growth(x₀=1.0, k=2.5, Δt=0.01, N=1, device=CPU())
@show simulate_exponential_growth(x₀=1.0, k=2.5, Δt=0.01, N=1, device=CUDADevice())

# iterative kernel launches
@show simulate_exponential_growth(x₀=1.0, k=2.5, Δt=0.01, N=100, device=CPU())
@show simulate_exponential_growth(x₀=1.0, k=2.5, Δt=0.01, N=100, device=CUDADevice())

# Optimization problem:
# If x₀=1, Δt = 0.01, N = 100, and x(t = 1) = π then what is k?
