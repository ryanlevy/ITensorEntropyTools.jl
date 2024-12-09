using ITensorNetworks: TreeTensorNetwork, orthogonalize, linkinds, siteinds, linkind
using ITensors: svd, diag, prime, dag, eigen

using ITensors: norm, dim
include("singular_value_funcs.jl")
include("density_matrix_tools.jl")

function ee_bipartite(
  ψ::TreeTensorNetwork, cut_edge::Pair; ee_type=EEType("vN"), verbose=false, kwargs...
)::Real
  """
  Obtain the bipartite entangelment entropy of a tree from edge to edge_next
  """

  cut = cut_edge[1]
  ψ = orthogonalize(ψ, cut)
  U, S, V = svd(ψ[cut], (linkinds(ψ, cut_edge)..., siteinds(ψ, cut)...))
  Sd = Array(diag(S))
  Sd = Sd .^ 2 # for entropy calc, we need sᵢ^2
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  return compute_ee(ee_type, Sd; kwargs...)
end

function ee_region(
  ψ::TreeTensorNetwork, region; ee_type=EEType("vN"), mode="auto", verbose=false, kwargs...
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
