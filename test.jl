include("gen_tests/vi_test.jl")

init_param!(approx, :convective_diffusivity_mu, 0.)
init_param!(approx, :convective_diffusivity_log_std, 0.)
init_param!(approx, :background_diffusivity_mu, 0.)
init_param!(approx, :background_diffusivity_log_std, 0.)

approx_trace = simulate(approx, ())