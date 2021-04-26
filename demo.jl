include("setup.jl")

@kernel function square(out, @Const(A))
    i = @index(Global)
    @inbounds out[i] = A[i]^2
end

using CUDA

A = CUDA.ones(128)
out = CUDA.zeros(128)

let kernel = square(CUDADevice())
    ev = kernel(out, A, ndrange=size(A))
    wait(ev)

    @show all(Array(out) .== 1.0)
end

dA = CUDA.zeros(128)
dout = CUDA.ones(128)

let kernel = autodiff(square(CUDADevice()))

    ev = kernel(Duplicated(out, dout), Duplicated(A, dA), ndrange=size(A))
    wait(ev)

    @show all(Array(out) .== 1.0)
    @show all(Array(dA) .== 2.0)
end
