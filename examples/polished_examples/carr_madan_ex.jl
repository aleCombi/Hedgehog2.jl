using DelimitedFiles
using DataFrames

function run_model_comparison_table(
    prob::PricingProblem,
    models::Vector{Hedgehog2.AbstractPricingMethod},
    lenses::Tuple;
    ad_method::Hedgehog2.GreekMethod = ForwardAD(),
    fd_method::Hedgehog2.GreekMethod = FiniteDifference(1e-3),
)

    results = Dict{String, Any}()
    rows = Vector{NamedTuple}()

    for model in models
        model_name = string(typeof(model).name.name)

        # Price
        price_time = @belapsed Hedgehog2.solve($prob, $model)
        sol = Hedgehog2.solve(prob, model)
        price = sol.price

        # AD Greeks
        batch_prob = BatchGreekProblem(prob, lenses)
        ad_time = @belapsed solve($batch_prob, $ad_method, $model)
        greeks_ad = solve(batch_prob, ad_method, model)

        # FD Greeks
        fd_batch_prob = BatchGreekProblem(prob, lenses)
        fd_time = @belapsed solve($fd_batch_prob, $fd_method, $model)
        greeks_fd = solve(fd_batch_prob, fd_method, model)

        results[model_name] = (
            price=price,
            price_time=price_time,
            greeks_ad=greeks_ad,
            ad_time=ad_time,
            greeks_fd=greeks_fd,
            fd_time=fd_time,
        )
    end

    baseline = first(models)
    baseline_name = string(typeof(baseline).name.name)

    for lens in lenses
        for (name, data) in results
            ad_val  = data.greeks_ad[lens]
            fd_val  = data.greeks_fd[lens]
            price   = data.price
            ad_time = data.ad_time / length(lenses)
            fd_time = data.fd_time / length(lenses)
            price_time = data.price_time

            push!(rows, (
                greek = string(lens),
                model = name,
                metric = "value",
                ad_value = ad_val,
                fd_value = fd_val,
                price = price,
                ad_us = ad_time * 1e6,
                fd_us = fd_time * 1e6,
                price_us = price_time * 1e6,
            ))
        end

        baseline_data = results[baseline_name]

        for (name, data) in results
            if name == baseline_name
                continue
            end

            ad_diff  = data.greeks_ad[lens] - baseline_data.greeks_ad[lens]
            fd_diff  = data.greeks_fd[lens] - baseline_data.greeks_fd[lens]
            price_diff = data.price - baseline_data.price
            ad_time_diff = (data.ad_time / length(lenses) - baseline_data.ad_time / length(lenses)) * 1e6
            fd_time_diff = (data.fd_time / length(lenses) - baseline_data.fd_time / length(lenses)) * 1e6
            price_time_diff = (data.price_time - baseline_data.price_time) * 1e6

            push!(rows, (
                greek = string(lens),
                model = "Δ " * name,
                metric = "diff",
                ad_value = ad_diff,
                fd_value = fd_diff,
                price = price_diff,
                ad_us = ad_time_diff,
                fd_us = fd_time_diff,
                price_us = price_time_diff,
            ))
        end
    end

    return DataFrame(rows)
end


# Setup
reference_date = Date(2020, 1, 1)
market_inputs = BlackScholesInputs(reference_date, 0.2, 100.0, 0.4)
expiry = reference_date + Year(1)
payoff = VanillaOption(100.0, expiry, European(), Call(), Spot())
prob = PricingProblem(payoff, market_inputs)

# Lenses
spot_lens = @optic _.market_inputs.spot
sigma_lens = Hedgehog2.VolLens(1, 1)
lenses = (spot_lens, sigma_lens)

# Models
models = [
    BlackScholesAnalytic(),
    CarrMadan(0.1, 64.0, LognormalDynamics()),
]

# Run
df = run_model_comparison_table(prob, models, lenses)

show(df, allrows=true, allcols=true)
