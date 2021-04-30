include("convective_adjustment_utils.jl")

# Problem parameters
grid = RegularGrid(Nz=32, Lz=128)

dTdz = 1e-4
κᶜ = 1.0
κᵇ = 1e-2
surface_flux = 1e-6
dt = 0.01
Nt = 10

# Problem data
T_out = CUDA.zeros(grid.Nz, Nt)
T_truth = CUDA.zeros(grid.Nz, Nt)
scratch = CUDA.zeros(grid.Nz, Nt)
error = Float64[]

T = 20 .+ dTdz * CuArray(-grid.Nz+1/2:-1/2) .* grid.Δz
T .= 20 + grid.

let kernel = uncertain_convect!(CUDADevice())
    ev = kernel(error, κᵇ, κᶜ, surface_flux, scratch, T_truth, T_out, T, grid, dt, Nt, ndrange=size(T))
    wait(ev)
    
    @show all(Array(out) .== 1.0)
end
