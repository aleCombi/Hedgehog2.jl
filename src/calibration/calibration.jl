struct CalibrationProblem{P, L}
    pricing_problem::P
    wrt::L  # accessor (from Accessors.jl)
end
