using ITensorMPS: AbstractMPS, orthogonalize, linkinds, siteinds, linkind
using ITensors: svd, diag, prime, dag, eigen, swapinds

using ITensors: norm, dim
include("singular_value_funcs.jl")
include("density_matrix_tools.jl")

function ee_bipartite(
  ψ::AbstractMPS, cut::Int; ee_type=EEType("vN"), verbose=false, kwargs...
)::Real
  """
  Obtain the bipartite entangelment entropy of a MPS left of the location cut
  """

  ψ = orthogonalize(ψ, cut)
  U, S, V = svd(ψ[cut], (linkinds(ψ, cut - 1)..., siteinds(ψ, cut)...))
  Sd = Array(diag(S))
  Sd = Sd .^ 2 # for entropy calc, we need sᵢ^2
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  return compute_ee(ee_type, Sd; kwargs...)
end

function ee_bipartite!(ψ::AbstractMPS, cut::Int; kwargs...)::Real
  orthogonalize!(ψ, cut)
  return ee_bipartite(ψ, cut; kwargs...)
end

function ee_region(
  ψ::AbstractMPS, region; ee_type=EEType("vN"), mode="auto", verbose=false, kwargs...
)::Real
  """
    Get the entanglement entropy of a region of sites, using either the "site" basis
    or the "link" basis. mode="auto" should select the smallest density matrix to compute
  """

  (length(region) == length(ψ)) && return 0.0

  ρ = density_matrix_region(ψ, region; mode, verbose, kwargs...)

  D, U = eigen(ρ; ishermitian=true)
  Sd = Array(diag(D))
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  return compute_ee(ee_type, Sd; kwargs...)
end

function negativity_region(
  ψ::AbstractMPS, region, subregion; verbose=false, kwargs...
)::Real
  """
    Compute the negativity of subregion within region
    N(rho_region) = (||rho^{T_subregion}_region ||_1 -1 )/2

    Warning: this is experimental
  """

  ρ = density_matrix_region(ψ, region; mode="sites", verbose, kwargs...)
  # perform partial transpose
  trans_inds = siteinds(ψ)[subregion]
  ρT = swapinds(ρ, trans_inds, prime(trans_inds))

  D, U = eigen(ρT; ishermitian=true)
  Sd = Array(diag(D))
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  N = sum(λ -> (abs(λ) - λ) / 2, Sd)
  return N
end
