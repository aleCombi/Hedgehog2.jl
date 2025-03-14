# Cox-Ross-Rubinstein Binomial Tree Pricing Implementation

"""
The Cox-Ross-Rubinstein binomial tree pricing method.

This struct represents the Cox-Ross-Rubinstein (CRR) binomial pricing model for option pricing.

# Fields
- `steps`: The number of time steps in the binomial tree.

# Notes
- This implementation supports options written on either the **forward price** or the **spot price**.
- When pricing an option on the forward price, discounting is already embedded in the forward.
- When pricing an option on the spot price, discounting must be explicitly applied at each step.
- The up probability is defined as:

```
p = 1 / (1 + u)
```
where:
- `u = exp(σ√ΔT)` is the up-move factor,
- `ΔT` is the time step duration in years.
"""
struct CoxRossRubinsteinMethod <: AbstractPricingMethod 
    steps::Int
end

"""
Returns the value at a given node in the binomial tree for a European option.

# Arguments
- `discounted_continuation`: Discounted expected future values from the next step.
- `::European`: Marker type for European exercise style.

# Returns
- The continuation value, as European options do not allow early exercise.
"""
function binomial_tree_value(_, discounted_continuation, _, _, ::European)
    return discounted_continuation
end

"""
Returns the value at a given node in the binomial tree for an American option.

# Arguments
- `step`: Current time step in the binomial tree.
- `discounted_continuation`: Discounted expected future values from the next step.
- `underlying_at_i`: Function to compute the underlying price at a given step.
- `payoff`: Payoff function of the option.
- `::American`: Marker type for American exercise style.

# Returns
- The maximum between the continuation value and the intrinsic value of the option (early exercise is considered).
"""
function binomial_tree_value(step, discounted_continuation, underlying_at_i, payoff, ::American)
    return max.(discounted_continuation, payoff(underlying_at_i(step)))
end

"""
Computes the underlying asset price at a given step when pricing an option on the **spot price**.

# Arguments
- `time_step`: Current time step in the binomial tree.
- `forward`: Forward price of the underlying asset.
- `rate`: Risk-free rate.
- `delta_time`: Time step size in years.
- `::Spot`: Marker type indicating the option is written on the spot price.

# Returns
- The estimated spot price, derived by discounting the forward price.
"""
function binomial_tree_underlying(time_step, forward, rate, delta_time, ::Spot)
    return exp(-rate * time_step * delta_time) * forward
end

"""
Computes the underlying asset price at a given step when pricing an option on the **forward price**.

# Arguments
- `forward`: Forward price of the underlying asset.
- `::Forward`: Marker type indicating the option is written on the forward price.

# Returns
- The forward price (unchanged, as forward prices already embed discounting).
"""
function binomial_tree_underlying(_, forward, _, _, ::Forward)
    return forward
end

"""
Computes the price of a vanilla option using the Cox-Ross-Rubinstein binomial tree model.

# Arguments
- `payoff`: The option payoff function, which determines intrinsic value.
- `market_inputs`: Market data including forward price, risk-free rate, and volatility.
- `method`: The CRR pricing method with a defined number of steps.

# Returns
- The computed option price using backward induction.

# Notes
- This implementation supports **options on either forward prices or spot prices**.
  - If the option is written on a forward price, discounting is embedded.
  - If the option is written on a spot price, discounting is applied explicitly.
- The up-move factor is computed as `u = exp(σ√ΔT)`, and the up probability is given by `p = 1 / (1 + u)`.
"""
function compute_price(payoff, market_inputs, method::CoxRossRubinsteinMethod)
    steps = method.steps
    ΔT = Dates.value.(payoff.expiry - market_inputs.referenceDate) / 365 / steps  # Assuming ACT/365 day count
    u = exp(market_inputs.sigma * sqrt(ΔT))  # Up-move factor
    forward_at_i(i) = market_inputs.forward * u .^ (-i:2:i)
    underlying_at_i(i) = binomial_tree_underlying(i, forward_at_i(i), market_inputs.rate, ΔT, payoff.underlying)

    # Up probability in forward measure
    p = 1 / (1 + u)

    # Initialize terminal payoffs
    value = payoff.(forward_at_i(steps))

    # Backward induction
    for step in (steps-1):-1:0
        continuation = p * value[2:end] + (1 - p) * value[1:end-1]
        df = exp(-market_inputs.rate * ΔT)  # Discounting
        value = binomial_tree_value(step, df * continuation, underlying_at_i, payoff, payoff.exercise_style)
    end

    return value[1]
end
