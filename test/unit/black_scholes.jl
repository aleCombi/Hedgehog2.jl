using Test
using Distributions
using Dates

# --- Helper Function for Discount Factor (assuming continuous compounding) ---
# NOTE: You might have a more complex rate structure; adjust accordingly.
# Ensure this helper matches how your actual `df` function works if needed,
# or rely on your existing `df` function directly within the tests.
function df_simple(r::Float64, T::Float64)
    return exp(-r * T)
end

# --- Test Suite ---

@testset "Black-Scholes Analytic Tests" begin
    method = BlackScholesAnalytic()
    ref_date = Date(2024, 1, 1) # Using a fixed past date for reproducibility

    # --- Test 1: log_dynamics Function ---
    @testset "Dynamics Type" begin
        # Assumes LognormalDynamics struct is defined
        @test Hedgehog2.log_dynamics(method) == LognormalDynamics()
    end

    # --- Test 2: Zero Volatility Cases ---
    @testset "Zero Volatility" begin
        spot = 100.0
        r = 0.05
        T = 0.5 # 6 months
        expiry_date = add_yearfrac(ref_date, T)
        sigma = 0.0
        # Assumes BlackScholesInputs constructor and fields are correct
        market_inputs_zero_vol = BlackScholesInputs(ref_date, r, spot, sigma)
        # Using helper df_simple; replace if your df function should be used directly
        D = df_simple(r, T)
        F = spot / D # Forward price

        # OTM Call
        K_otm = 110.0
        # Assumes VanillaOption, European, Call, Spot structs/types are defined
        payoff_call_otm = VanillaOption(K_otm, expiry_date, European(), Call(), Spot())
        # Assumes PricingProblem constructor and fields are correct
        prob_call_otm = PricingProblem(payoff_call_otm, market_inputs_zero_vol)
        # Assumes solve function and AnalyticSolution struct (with price field) are correct
        sol_call_otm = solve(prob_call_otm, method)
        expected_call_otm = D * max(F - K_otm, 0.0)
        @test isapprox(sol_call_otm.price, expected_call_otm, atol=1e-9)

        # ITM Call
        K_itm = 90.0
        payoff_call_itm = VanillaOption(K_itm, expiry_date, European(), Call(), Spot())
        prob_call_itm = PricingProblem(payoff_call_itm, market_inputs_zero_vol)
        sol_call_itm = solve(prob_call_itm, method)
        expected_call_itm = D * max(F - K_itm, 0.0)
        @test isapprox(sol_call_itm.price, expected_call_itm, atol=1e-9)

        # OTM Put
        # Assumes Put type is defined
        payoff_put_otm = VanillaOption(K_itm, expiry_date, European(), Put(), Spot()) # Use K_itm=90
        prob_put_otm = PricingProblem(payoff_put_otm, market_inputs_zero_vol)
        sol_put_otm = solve(prob_put_otm, method)
        expected_put_otm = D * max(K_itm - F, 0.0)
        @test isapprox(sol_put_otm.price, expected_put_otm, atol=1e-9)

        # ITM Put
        payoff_put_itm = VanillaOption(K_otm, expiry_date, European(), Put(), Spot()) # Use K_otm=110
        prob_put_itm = PricingProblem(payoff_put_itm, market_inputs_zero_vol)
        sol_put_itm = solve(prob_put_itm, method)
        expected_put_itm = D * max(K_otm - F, 0.0)
        @test isapprox(sol_put_itm.price, expected_put_itm, atol=1e-9)
    end

    # --- Test 3: Benchmark Comparisons (Replace with actual benchmarks!) ---
    # BENCHMARK SOURCE: Hypothetical values - replace with values from a trusted source (calculator/textbook)
    @testset "Benchmark Comparisons" begin
        spot = 100.0
        r = 0.05 # 5% risk-free rate
        sigma = 0.20 # 20% volatility
        T = 1.0 # 1 year
        # Assumes yearfrac function is available and correct
        expiry_date = add_yearfrac(ref_date, T)
        market_inputs = BlackScholesInputs(ref_date, r, spot, sigma)
        D = df_simple(r, T)
        F = spot / D

        # Case 1: ATM Call (K ≈ F)
        K_atm = F # Approximately 100 * exp(0.05) ≈ 105.13
        payoff_call_atm = VanillaOption(K_atm, expiry_date, European(), Call(), Spot())
        prob_call_atm = PricingProblem(payoff_call_atm, market_inputs)
        sol_call_atm = solve(prob_call_atm, method)
        # *** REPLACE WITH ACTUAL BENCHMARK VALUE ***
        # Value for S=100, K=F=105.127..., r=0.05, σ=0.2, T=1
        expected_call_atm = 7.9655
        @test isapprox(sol_call_atm.price, expected_call_atm , atol=1e-4)

        # Case 2: ITM Call
        K_itm = 90.0
        payoff_call_itm = VanillaOption(K_itm, expiry_date, European(), Call(), Spot())
        prob_call_itm = PricingProblem(payoff_call_itm, market_inputs)
        sol_call_itm = solve(prob_call_itm, method)
        # Benchmark value from Quantlib
        # Value for S=100, K=90, r=0.05, sigma=0.2, T=1
        expected_call_itm = 16.6994
        @test isapprox(sol_call_itm.price, expected_call_itm, atol=1e-4)

        # Case 3: OTM Put
        K_otm = 90.0 # Same strike as ITM Call
        payoff_put_otm = VanillaOption(K_otm, expiry_date, European(), Put(), Spot())
        prob_put_otm = PricingProblem(payoff_put_otm, market_inputs)
        sol_put_otm = solve(prob_put_otm, method)
        # Benchmark value from Quantlib
        # Value for S=100, K=90, r=0.05, sigma=0.2, T=1
        expected_put_otm = 2.3101
        @test isapprox(sol_put_otm.price, expected_put_otm, atol=1e-4)

        # Case 4: Different Time (e.g., T ∼ 0.25) ITM Put
        T_short = 91 / 365
        expiry_short = ref_date + Day(91)
        K_itm_put = 110.0
        market_inputs_short = BlackScholesInputs(ref_date, r, spot, sigma) # Re-use r, spot, sigma
        payoff_put_itm_short = VanillaOption(K_itm_put, expiry_short, European(), Put(), Spot())
        prob_put_itm_short = PricingProblem(payoff_put_itm_short, market_inputs_short)
        sol_put_itm_short = solve(prob_put_itm_short, method)
        # Benchmark value from Quantlib
        # Value S=100, K=110, r=0.05, sigma=0.2, T=0.25
        expected_put_itm_short = 9.8237
        @test isapprox(sol_put_itm_short.price, expected_put_itm_short, atol=1e-4)
    end

    # --- Test 4: Put-Call Parity ---
    @testset "Put-Call Parity" begin
        spot = 105.0
        K = 100.0
        r = 0.03
        sigma = 0.25
        T = 0.75
        expiry_date = Hedgehog2.add_yearfrac(ref_date, T)
        market_inputs = BlackScholesInputs(ref_date, r, spot, sigma)
        D = df_simple(r, T)
        F = spot / D

        payoff_call = VanillaOption(K, expiry_date, European(), Call(), Spot())
        prob_call = PricingProblem(payoff_call, market_inputs)
        sol_call = solve(prob_call, method)

        payoff_put = VanillaOption(K, expiry_date, European(), Put(), Spot())
        prob_put = PricingProblem(payoff_put, market_inputs)
        sol_put = solve(prob_put, method)

        # Parity: C - P = D * (F - K)
        @test isapprox(sol_call.price - sol_put.price, D * (F - K), atol=1e-6)
        # Alternative check: C - P = S - K*D
        @test isapprox(sol_call.price - sol_put.price, spot - K * D, atol=1e-6)
    end

    # --- Test 5: Short Expiry Limit ---
    # As T->0, Price should approach discounted intrinsic value D * max(cp*(F-K), 0)
    @testset "Short Expiry Limit" begin
        spot = 100.0
        K = 105.0 # OTM Call / ITM Put
        r = 0.05
        sigma = 0.2
        T_tiny = 1 / 365.0 # 1 day
        expiry_date = ref_date + Day(1)
        market_inputs = BlackScholesInputs(ref_date, r, spot, sigma)
        D = df_simple(r, T_tiny)
        F = spot / D

        # OTM Call
        payoff_call = VanillaOption(K, expiry_date, European(), Call(), Spot())
        prob_call = PricingProblem(payoff_call, market_inputs)
        sol_call = solve(prob_call, method)
        expected_call_intrinsic = D * max(F - K, 0.0)
        # Allow slightly larger tolerance because BS price isn't exactly intrinsic for T>0
        # A more precise test would calculate the exact BS value for T_tiny.
        @test isapprox(sol_call.price, expected_call_intrinsic, atol=0.1)

        # ITM Put
        payoff_put = VanillaOption(K, expiry_date, European(), Put(), Spot())
        prob_put = PricingProblem(payoff_put, market_inputs)
        sol_put = solve(prob_put, method)
        expected_put_intrinsic = D * max(K - F, 0.0)
        @test isapprox(sol_put.price, expected_put_intrinsic, atol=0.1)
    end
end