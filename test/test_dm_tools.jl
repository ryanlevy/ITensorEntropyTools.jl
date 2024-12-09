using Test
using ITensorTreeEntropyTools
using ITensorTreeEntropyTools: density_matrix_sites, density_matrix_bond

using ITensors: tr, order
using ITensorMPS: siteinds, random_mps
using ITensorNetworks: ttn

using Random: seed!
seed!(42)

@testset "Density Matrix Checks" begin
  d = 2
  N = 5

  s = siteinds(d, N)
  p = random_mps(s; linkdims=4)
  p = ttn(p[:])

  @testset "Test Legal DM" begin
    @testset "Sites" begin
      for L in 1:N
        for start in 1:(N - L + 1)
          region = start:(start + L - 1)
          ρ = density_matrix_sites(p, region)
          @test tr(ρ) ≈ 1
          @test order(ρ) == length(region) * 2
        end
      end
    end

    @testset "Sites - noncontiguous" begin
      # odd
      region = 1:2:N
      ρ = density_matrix_sites(p, region)
      @test tr(ρ) ≈ 1
      @test order(ρ) == length(region) * 2
      # even
      region = 2:2:N
      ρ = density_matrix_sites(p, region)
      @test tr(ρ) ≈ 1
      @test order(ρ) == length(region) * 2
    end

    @testset "Bond - interior" begin
      for L in 2:(N - 1)
        for start in 2:(N - L) # ignore some end points for now
          region = start:(start + L - 1)
          ρ = density_matrix_bond(p, region[1], region[end])
          @test tr(ρ) ≈ 1
          @test order(ρ) == 4
        end
      end
    end
  end
end

@testset "QN Density Matrix Checks" begin
  N = 5

  s = siteinds("Qubit", N; conserve_qns=true)
  state = [isodd(n) ? "Up" : "Dn" for n in 1:N]
  p = random_mps(s, state; linkdims=4)
  p = ttn(p[:])

  @testset "Test Legal DM" begin
    @testset "Sites" begin
      for L in 1:N
        for start in 1:(N - L + 1)
          region = start:(start + L - 1)
          ρ = density_matrix_sites(p, region)
          @test tr(ρ) ≈ 1
          @test order(ρ) == length(region) * 2
        end
      end
    end

    @testset "Sites - noncontiguous" begin
      # odd
      region = 1:2:N
      ρ = density_matrix_sites(p, region)
      @test tr(ρ) ≈ 1
      @test order(ρ) == length(region) * 2
      # even
      region = 2:2:N
      ρ = density_matrix_sites(p, region)
      @test tr(ρ) ≈ 1
      @test order(ρ) == length(region) * 2
    end

    @testset "Bond - interior" begin
      for L in 2:(N - 1)
        for start in 2:(N - L) # ignore some end points for now
          region = start:(start + L - 1)
          ρ = density_matrix_bond(p, region[1], region[end])
          @test tr(ρ) ≈ 1
          @test order(ρ) == 4
        end
      end
    end
  end
end
