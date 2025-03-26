using Revise, Hedgehog2, Interpolations

# Grid definitions
tenors  = [0.25, 0.5, 1.0, 2.0]               # maturities in years
strikes = [80.0, 90.0, 100.0, 110.0]          # strikes

# 2D volatility grid: vols[i, j] corresponds to (tenor[i], strike[j])
vols = [
    0.22  0.21  0.20  0.19;    # T = 0.25
    0.23  0.22  0.21  0.20;    # T = 0.50
    0.25  0.24  0.23  0.22;    # T = 1.00
    0.28  0.27  0.26  0.25     # T = 2.00
]

reference_date = Date(2020,1,1)
vol_surface = build_rect_vol_surface(reference_date, tenors, strikes, vols)

# Query interpolated vol
T = 0.75
K = 95.0
σ = getvol(vol_surface, T, K)

println("Implied vol at T = $T, K = $K is σ = $σ")

