using Test
using ITensorEntropyTools
using ITensorEntropyTools: get_density_matrix_sites, get_density_matrix_bond

using ITensors: tr, order
using ITensorMPS: siteinds, random_mps

using Random: seed!
seed!(42)

@testset "Entropy Checks" begin
  d = 2
  N = 5

  s = siteinds(d, N)
  p = random_mps(s; linkdims=4)

  @testset "Site and Bond equivalence" begin
    ee_singles_s = [get_ee_region(p, [i]; mode="sites") for i in 1:N]
    ee_singles_b = [get_ee_region(p, [i]; mode="bond") for i in 1:N]
    @test ee_singles_s ≈ ee_singles_b
    ee_doubles_s = [get_ee_region(p, [i, i + 1]; mode="sites") for i in 1:(N - 1)]
    ee_doubles_b = [get_ee_region(p, [i, i + 1]; mode="bond") for i in 1:(N - 1)]
    @test ee_doubles_s ≈ ee_doubles_b
  end
end
