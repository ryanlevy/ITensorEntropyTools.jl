@eval module $(gensym())
using ITensorEntropyTools: ITensorEntropyTools
using Test: @testset

@testset "Test examples" begin
  example_files = ["basiscs.jl"]
  @testset "Test $example_file" for example_file in example_files
    include(joinpath(pkgdir(ITensorEntropyTools), "examples", example_file))
  end
end
end
