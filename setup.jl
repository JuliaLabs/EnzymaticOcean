using KernelAbstractions
using CUDAKernels
using Enzyme

# TODO: The need for this currently prevents us from turning this into a neat package
#       Can we do this without requiring the specific kernel?
@inline function CUDAKernels.Cassette.overdub(::CUDAKernels.CUDACtx, ::typeof(Enzyme.autodiff_no_cassette), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; CUDAKernels.Cassette.overdub(CUDAKernels.CUDACTX, f, args...))
    Enzyme.autodiff_no_cassette(f′, args...)
end

@inline function KernelAbstractions.Cassette.overdub(::KernelAbstractions.CPUCtx, ::typeof(Enzyme.autodiff_no_cassette), f, args...)
    f′ = (args...) -> (Base.@_inline_meta; KernelAbstractions.Cassette.overdub(KernelAbstractions.CPUCTX, f, args...))
    Enzyme.autodiff_no_cassette(f′, args...)
end

function Enzyme.autodiff(kernel::KernelAbstractions.Kernel{<:Any, <:Any, <:Any, Fun}) where Fun
    function df(ctx, args...)
        Enzyme.autodiff_no_cassette(kernel.f, ctx, args...)
    end
    similar(kernel, df)
end
