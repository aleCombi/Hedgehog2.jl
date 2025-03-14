"""The Cox-Ross-Rubinstein binomial tree pricing method.

This struct represents the Cox-Ross-Rubinstein binomial pricing model for option pricing.
"""
struct CoxRossRubinsteinMethod <: AbstractPricingMethod 
  steps
end

function binomial_tree_value(step, discounted_continuation, underlying_at_i, payoff, ::European)
  return discounted_continuation
end

function binomial_tree_value(step, discounted_continuation, underlying_at_i, payoff, ::American)
  return max.(discounted_continuation, payoff(underlying_at_i(step)))
end

function binomial_tree_underlying(time_step, forward, rate, delta_time, ::Spot)
  return exp(-rate*time_step*delta_time) * forward
end

function binomial_tree_underlying(_, forward, _, _, ::Forward)
  return forward
end

function compute_price(payoff::P, market_inputs::BlackScholesInputs, method::CoxRossRubinsteinMethod) where P <: AbstractPayoff
  ΔT = Dates.value.(payoff.expiry .- market_inputs.referenceDate) ./ 365 / method.steps # we might want to specify daycount conventions to ensure consistency
  up_step = exp(market_inputs.sigma * sqrt(ΔT))
  forward_at_i(i) = market_inputs.forward * up_step .^ (-i:2:i)
  underlying_at_i(i) = binomial_tree_underlying(i, forward_at_i(i), market_inputs.rate, ΔT, payoff.underlying)
  up_probability = 1 / (1 + up_step)
  value = payoff(forward_at_i(method.steps))

  for step in (method.steps-1):-1:0
    continuation = up_probability * value[2:end] + (1 - up_probability) * value[1:end-1]
    df = exp(-market_inputs.rate * ΔT)
    value = binomial_tree_value(step, df * continuation, underlying_at_i, payoff, payoff.exercise_style)
  end

  return value[1]
end