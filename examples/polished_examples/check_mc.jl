using Revise, Hedgehog2, BenchmarkTools, Dates
using Accessors
import Accessors: @optic
using DifferentialEquations
# ------------------------------
# Define payoff and pricing problem
# ------------------------------
strike = 1.0
expiry = Date(2020, 1, 2)

payoff = VanillaOption(strike, expiry, European(), Put(), Spot())

reference_date = Date(2020, 1, 1)
rate = 0.03
spot = 1.0
sigma = 1.0

inputs = BlackScholesInputs(reference_date, rate, spot, sigma)

# -- 2. Define payoff and pricing problem --
prob = PricingProblem(payoff, inputs)

# -- 3. Define method and strategy --
trajectories = 10000
strategy = BlackScholesExact(trajectories)
method = MonteCarlo(LognormalDynamics(), strategy)

T = yearfrac(prob.market_inputs.referenceDate, prob.payoff.expiry)
strategy = method.strategy
dynamics = method.dynamics
strategy = method.strategy
tspan = (0.0, T)
dt = T / strategy.steps

antithetic = get(strategy.kwargs, :antithetic, false)
meth=Hedgehog2.EM()
# Step 1: simulate original paths
normal_prob = Hedgehog2.sde_problem(dynamics, strategy, inputs, tspan)
@btime normal_prob = Hedgehog2.sde_problem($dynamics, $strategy, $inputs, $tspan)

ensemble_prob = Hedgehog2.get_ensemble_problem(normal_prob, strategy)
@btime ensemble_prob = Hedgehog2.get_ensemble_problem($normal_prob, $strategy)

@btime normal_sol = DifferentialEquations.solve($ensemble_prob, $meth; dt = $dt, trajectories=$strategy.trajectories)
@code_warntype normal_sol = DifferentialEquations.solve(ensemble_prob, meth; dt = dt, trajectories=strategy.trajectories)

# -- 4. Simulate paths and solve --
@info "Solving..."
@code_warntype Hedgehog2.solve(prob, method)
@btime solve($prob, $method)

solution = solve(prob, method)
@show typeof(solution)
@show typeof(solution.ens)
@show typeof(solution.ens.u)
@show typeof(solution.ens.u[1])
@show typeof(solution.ens.u[1].u)
@show solution.price

