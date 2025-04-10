using DelimitedFiles
using DataFrames
using Hedgehog2

function run_model_comparison_table(
    prob::PricingProblem,
    models::Vector{Hedgehog2.AbstractPricingMethod},
    lenses::Tuple;
    ad_method::Hedgehog2.GreekMethod = ForwardAD(),
    fd_method::Hedgehog2.GreekMethod = FiniteDifference(1e-3),
    analytic_method::Union{Nothing, Hedgehog2.GreekMethod} = nothing,
)

    results = Dict{String, Any}()
    rows = Vector{NamedTuple}()

    for model in models
        model_name = string(typeof(model).name.name)

        # Price
        price_time = @belapsed Hedgehog2.solve($prob, $model)
        sol = Hedgehog2.solve(prob, model)
        price = sol.price

        # Batch Greeks (AD & FD)
        batch_prob = BatchGreekProblem(prob, lenses)
        ad_time = @belapsed solve($batch_prob, $ad_method, $model)
        fd_time = @belapsed solve($batch_prob, $fd_method, $model)
        greeks_ad = solve(batch_prob, ad_method, model)
        greeks_fd = solve(batch_prob, fd_method, model)

        # Attempt full AnalyticGreek batch solve
        greeks_analytic = Dict{Any, Union{Float64, Missing}}()
        if analytic_method !== nothing
            try
                greeks_full = solve(batch_prob, analytic_method, model)
                for lens in lenses
                    greeks_analytic[lens] = greeks_full[lens]
                end
            catch
                # Fallback to individual lens-based attempts
                for lens in lenses
                    try
                        single_prob = BatchGreekProblem(prob, (lens,))
                        val = solve(single_prob, analytic_method, model)[lens]
                        greeks_analytic[lens] = val
                    catch
                        greeks_analytic[lens] = missing
                    end
                end
            end
        else
            for lens in lenses
                greeks_analytic[lens] = missing
            end
        end

        results[model_name] = (
            price=price,
            price_time=price_time,
            greeks_ad=greeks_ad,
            ad_time=ad_time,
            greeks_fd=greeks_fd,
            fd_time=fd_time,
            greeks_analytic=greeks_analytic,
        )
    end

    baseline = first(models)
    baseline_name = string(typeof(baseline).name.name)

    for lens in lenses
        for (name, data) in results
            ad_val = data.greeks_ad[lens]
            fd_val = data.greeks_fd[lens]
            analytic_val = data.greeks_analytic[lens]
            price = data.price
            ad_time = data.ad_time / length(lenses)
            fd_time = data.fd_time / length(lenses)
            price_time = data.price_time

            push!(rows, (
                greek = string(lens),
                model = name,
                metric = "value",
                ad_value = ad_val,
                fd_value = fd_val,
                analytic_value = analytic_val,
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

            a1 = data.greeks_analytic[lens]
            a0 = baseline_data.greeks_analytic[lens]
            analytic_diff = (!ismissing(a1) && !ismissing(a0)) ? a1 - a0 : missing

            push!(rows, (
                greek = string(lens),
                model = "Δ " * name,
                metric = "diff",
                ad_value = ad_diff,
                fd_value = fd_diff,
                analytic_value = analytic_diff,
                price = price_diff,
                ad_us = ad_time_diff,
                fd_us = fd_time_diff,
                price_us = price_time_diff,
            ))

            add_separator!(rows)
        end
    end

    return DataFrame(rows)
end

function add_separator!(rows)
    # Add a separator row to visually split groups when displayed
    push!(rows, (
        greek = "────────────────────────",
        model = "",
        metric = "",
        ad_value = missing,
        fd_value = missing,
        analytic_value = missing,
        price = missing,
        ad_us = missing,
        fd_us = missing,
        price_us = missing,
    ))
end