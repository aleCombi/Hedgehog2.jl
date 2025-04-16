using Test
using Hedgehog

function test_model_price_agreement_over_problems(
    problems::Vector{PricingProblem},
    model1::Hedgehog.AbstractPricingMethod,
    model2::Hedgehog.AbstractPricingMethod;
    rtol::Real = 1e-8,
    atol::Real = 1e-10,
)
    name1 = string(typeof(model1).name.name)
    name2 = string(typeof(model2).name.name)

    @testset "Price Agreement for $name1 vs $name2 over $(length(problems)) problems" begin
        for (i, prob) in enumerate(problems)
            @testset "Problem $i: $(prob.payoff)" begin
                price1 = Hedgehog.solve(prob, model1).price
                price2 = Hedgehog.solve(prob, model2).price

                @test isapprox(price1, price2; rtol=rtol, atol=atol)
            end
        end
    end
end
