using KernelAbstractions
using CUDAKernels
using Enzyme

# TODO: The need for this currently prevents us from turning this into a neat package
#       Can we do this without requiring the specific kernel?
@inline function CUDAKernels.Cassette.overdub(::CUDAKernels.CUDACtx, ::typeof(Enzyme.autodiff), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; CUDAKernels.Cassette.overdub(CUDAKernels.CUDACTX, f, args...))
    Enzyme.autodiff(f′, args...)
end

function Enzyme.autodiff(kernel::KernelAbstractions.Kernel)
    function df(ctx, args...)
        Enzyme.autodiff(kernel.fun, ctx, args...)
    end
    similar(kernel, df)
end
