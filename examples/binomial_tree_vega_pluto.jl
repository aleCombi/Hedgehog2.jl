### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils
using Accessors
# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 3517e1c0-00ba-11f0-08bf-d37b642b75dc
begin
    using Pkg
    Pkg.develop(path = "C:\\repos\\Hedgehog.jl")
    Pkg.instantiate()
    Pkg.add("Accessors")
    Pkg.add("Plots")
    Pkg.add("ForwardDiff")
    Pkg.add("PlutoUI")
    Pkg.add("Interpolations")
    using Hedgehog, Dates, Accessors, Plots, ForwardDiff, PlutoUI, Interpolations
end

# ╔═╡ d275bd54-fa00-4412-b105-2907d62c9da0
begin
    # define payoff
    strike = 5
    expiry = Date(2017, 7, 26)
    call_put = Hedgehog.Put()
    underlying = Hedgehog.Forward()
    american_payoff =
        VanillaOption(strike, expiry, Hedgehog.American(), call_put, underlying)

    # define market inputs
    reference_date = Date(2017, 6, 29)
    T = Dates.values(expiry - reference_date) / 365
    rate = 0.013047467783283154
    spot = 5.6525 * exp(r * T)
    sigma = 0.1689900715
    market_inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

    # create Cox Ross Rubinstein pricer
    steps = 800
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

        y = [fun(pricer, pricer.payoff.strike, vol) for vol in vol_range]
        return plot_func(
            collect(vol_range),
            y,
            markercolor = markercolor,
            markersize = 1,
            markerstrokewidth = 0,
            label = string(nameof(fun)),
            xlabel = "Vol",
            ylabel = "Price",
        )
    end

end

# ╔═╡ 5e769b05-ffeb-483d-a379-ec9f9e45ffd2
sliding_vol = @bind fixed_vol Slider(0.01:0.01:0.4, show_value = true)

# ╔═╡ 2376a0aa-01d9-4416-b22f-136bb216b277
begin
    strike_range = 5.5:0.01:5.75
    vol_range = 0.1:0.001:0.3
    bump_size = 1E-3
    volatility = crr_american_pricer.marketInputs.sigma
    pricer_vol = @set crr_american_pricer.marketInputs.sigma = fixed_vol
    fd_volga_funshort(p, s, v) = fd_volga_fun(p, s, v, bump_size)
    fd_vega_funshort(p, s, v) = fd_vega_fun(p, s, v, bump_size)

    plot_against_vol(scatter, ad_vega_fun, pricer_vol, vol_range)
    # plot_against_vol(scatter, fd_vega_funshort, pricer_vol, vol_range)	

end

# ╔═╡ b8304ed6-a913-433f-82b8-e14bf53d0150
plot_against_vol(scatter, ad_volga_fun, pricer_vol, vol_range)

# ╔═╡ 6f2f37fd-e366-4932-adfb-ee12fe709332
plot_against_vol(scatter, price_fun, pricer_vol, vol_range)

# ╔═╡ Cell order:
# ╠═3517e1c0-00ba-11f0-08bf-d37b642b75dc
# ╠═d275bd54-fa00-4412-b105-2907d62c9da0
# ╠═5e769b05-ffeb-483d-a379-ec9f9e45ffd2
# ╠═2376a0aa-01d9-4416-b22f-136bb216b277
# ╠═b8304ed6-a913-433f-82b8-e14bf53d0150
# ╠═6f2f37fd-e366-4932-adfb-ee12fe709332
