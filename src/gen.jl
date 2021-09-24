using Gen
using Zygote

# Custom gradient definition to use Enzyme
struct ConvectGF <: CustomDetermGF{Vector{Float64}, Function} end
const convectgf = ConvectGF()

accepts_output_grad(::ConvectGF) = true

function Gen.apply_with_state(::ConvectGF, args)
    return ChainRulesCore.rrule_via_ad(Zygote.ZygoteRuleConfig(), convective_model, args...)
end

function Gen.gradient_with_state(::ConvectGF, pullback, args, retgrad)
    dT = retgrad
    _, _, _, dTgrad, dκᶜ, dκᵇ = pullback(dT)
    return (nothing, nothing, dTgrad, dκᶜ, dκᵇ)
end
Gen.has_argument_grads(::ConvectGF) = (false, false, true, true, true)