using KernelAbstractions
using CUDAKernels
using Enzyme

@inline function CUDAKernels.Cassette.overdub(::CUDAKernels.CUDACtx, ::typeof(Enzyme.autodiff), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; CUDAKernels.Cassette.overdub(CUDAKernels.CUDACTX, f, args...))
    Enzyme.autodiff(f′, args...)
end
