using Hedgehog2, Accessors, Dates

begin
    # define payoff
    strike = 5
    expiry = Date(2017, 7, 26)
    call_put = Hedgehog2.Put()
    underlying = Hedgehog2.Forward()
    american_payoff =
        VanillaOption(strike, expiry, Hedgehog2.American(), call_put, underlying)

    # define market inputs
    reference_date = Date(2017, 6, 29)
    rate = 0.013047467783283154
    T = yearfrac(reference_date, expiry)
    spot = 5.6525 * exp(-rate * T)
    sigma = 0.1689900715
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # create Cox Ross Rubinstein pricer
    steps = 80
    crr = CoxRossRubinsteinMethod(steps)
    crr_american_pricer = Pricer(american_payoff, market_inputs, crr)

    function price_fun(pricer, strike, volatility)
        new_pricer = @set pricer.marketInputs.sigma = volatility
        new_pricer = @set new_pricer.payoff.strike = strike
        return new_pricer()
    end

    function ad_vega_fun(pricer, strike, volatility)
        return ForwardDiff.derivative(v -> price_fun(pricer, strike, v), volatility)
    end

    function ad_volga_fun(pricer, strike, volatility)
        return ForwardDiff.derivative(v -> ad_vega_fun(pricer, strike, v), volatility)
    end

    function fd_vega_fun(pricer, strike, volatility, bump_size)
        price = price_fun(pricer, strike, volatility)
        price_bumped = price_fun(pricer, strike, volatility + bump_size)
        return (price_bumped - price) / bump_size
    end

    function fd_volga_fun(pricer, strike, volatility, bump_size)
        price = price_fun(pricer, strike, volatility)
        price_bumped_up = price_fun(pricer, strike, volatility + bump_size)
        price_bumped_down = price_fun(pricer, strike, volatility - bump_size)
        return (price_bumped_up + price_bumped_down - 2 * price) / bump_size / bump_size
    end

    function plot_against_strike(plot_func, fun, pricer, strike_range)
        label = string(nameof(fun))
        markercolor = :blue
        if occursin("fd", label)
            markercolor = :red
        end
        return plot_func(
            strike_range,
            s -> fun(pricer, s, pricer.marketInputs.sigma),
            markercolor = markercolor,
            markersize = 1,
            markerstrokewidth = 0,
            label = string(nameof(fun)),
            xlabel = "Strike",
            ylabel = "Price",
        )
    end

    function plot_against_vol(plot_func, fun, pricer, vol_range)
        label = string(nameof(fun))
        markercolor = :blue
        if occursin("fd", label)
            markercolor = :red
        end

        return plot_func(
            vol_range,
            v -> fun(pricer, pricer.payoff.strike, v),
            markercolor = markercolor,
            markersize = 1,
            markerstrokewidth = 0,
            label = string(nameof(fun)),
            xlabel = "Vol",
            ylabel = "Price",
        )
    end

    strike_range = 5.5:0.01:5.75
    vol_range = 0.1:0.01:0.3
    bump_size = 1E-2
    volatility = crr_american_pricer.marketInputs.sigma
    pricer_vol = crr_american_pricer
    fd_volga_funshort(p, s, v) = fd_volga_fun(p, s, v, 1E-1)
    fd_vega_funshort(p, s, v) = fd_vega_fun(p, s, v, bump_size)

    # plot_against_vol(scatter, ad_vega_fun, pricer_vol, vol_range)
    plot_against_vol(scatter, fd_vega_funshort, pricer_vol, vol_range)
    # savefig("examples/figures/binomial_tree_vega.png")
    plot_against_vol(scatter, fd_volga_funshort, pricer_vol, vol_range)
    # savefig("examples/figures/binomial_tree_volga.png")
    # plot_against_vol(scatter, price_fun, pricer_vol, vol_range)
end
