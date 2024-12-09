using Test
using ITensorTreeEntropyTools
using ITensorTreeEntropyTools: density_matrix_sites, density_matrix_bond

using ITensors: tr, order
using ITensorMPS: siteinds, random_mps

using Random: seed!
seed!(42)

@testset "Entropy Checks" begin
  N = 5

  for d in [2, 3, 4]
    @testset "d=$d" begin
      s = siteinds(d, N)
      p = random_mps(s; linkdims=4)

      @testset "Site and Bond equivalence" begin
        ee_singles_s = [ee_region(p, [i]; mode="sites") for i in 1:N]
        ee_singles_b = [ee_region(p, [i]; mode="bond") for i in 1:N]
        @test ee_singles_s ≈ ee_singles_b
        ee_doubles_s = [ee_region(p, [i, i + 1]; mode="sites") for i in 1:(N - 1)]
        ee_doubles_b = [ee_region(p, [i, i + 1]; mode="bond") for i in 1:(N - 1)]
        @test ee_doubles_s ≈ ee_doubles_b
      end
    end
  end
  for stype in ["Qubit", "S=1", "Electron"]
    @testset "QN sites=$stype" begin
      s = siteinds(stype, N; conserve_qns=true)
      state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
      p = random_mps(s, state; linkdims=4)

      @testset "Site and Bond equivalence" begin
        ee_singles_s = [ee_region(p, [i]; mode="sites") for i in 1:N]
        ee_singles_b = [ee_region(p, [i]; mode="bond") for i in 1:N]
        @test ee_singles_s ≈ ee_singles_b
        ee_doubles_s = [ee_region(p, [i, i + 1]; mode="sites") for i in 1:(N - 1)]
        ee_doubles_b = [ee_region(p, [i, i + 1]; mode="bond") for i in 1:(N - 1)]
        @test ee_doubles_s ≈ ee_doubles_b
      end
    end
  end
end
