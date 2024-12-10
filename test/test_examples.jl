@eval module $(gensym())
using ITensorEntropyTools: ITensorEntropyTools
using Test: @test, @testset

@testset "Test examples" begin
  example_files = ["basics.jl", "advanced.jl", "rainbow_ghz_w.jl", "Jensens_inequality.jl"]
  @testset "Test $example_file" for example_file in example_files
    @test include(joinpath(pkgdir(ITensorEntropyTools), "examples", example_file)) ==
      nothing
  end
end
end
