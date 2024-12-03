using ITensorMPS: AbstractMPS, orthogonalize, linkinds, siteinds, linkind
using ITensors: svd, diag, prime, dag, eigen

include("singular_value_funcs.jl")

function get_ee_bipartite(ψ::AbstractMPS, cut::Int; ee_type=EEType("vN"), kwargs...)::Real
  """
  Obtain the bipartite entangelment entropy of a MPS left of the location cut
  """

  ψ = orthogonalize(ψ, cut)
  U, S, V = svd(ψ[cut], (linkinds(ψ, cut - 1)..., siteinds(ψ, cut)...))
  Sd = Array(diag(S))
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  Sd = Sd .^ 2 # for entropy calc, we need pᵢ^2
  return compute_ee(ee_type, Sd, kwargs...)
end

function get_density_matrix_sites(ψ_::AbstractMPS, region;)
  """
    Obtain a density matrix, with external site indices
    leaves sites uncontracted
  """
  # TODO: This should generalize to a network easily
  # but currently assuming the sites are ordered 
  region = sort(region)
  start, stop = region[1], region[end]
  # add check that these are legal values?
  ψ = orthogonalize(ψ_, start)
  if length(region) == 1
    return ψ[start] * prime(dag(ψ[start]), "Site")
  end
  ψH = prime(dag(ψ), "Link")
  lᵢ₁ = linkind(ψ, start - 1) #this is n,n+1 link
  ρ = (lᵢ₁ == nothing) ? ψ[start] : prime(ψ[start], lᵢ₁)
  ρ *= prime(ψH[start], "Site")
  si = 2
  for k in (start + 1):(stop - 1)
    ρ *= ψ[k]
    if region[si] == k
      ρ *= prime(ψH[k], "Site")
      si += 1
    else
      ρ *= ψH[k]
    end
  end

  lⱼ = linkind(ψ, stop)
  ρ *= (lⱼ == nothing) ? ψ[stop] : prime(ψ[stop], lⱼ)
  ρ *= prime(ψH[stop], "Site")
  return ρ
end

function get_density_matrix_bond(ψ_::AbstractMPS, start, stop;)
  """
    Obtain a density matrix, with external link indices
    leaves edge links uncontracted
    Requires the region to be contiguous
  """
  # TODO: This should generalize to a network easily
  # but currently assuming the sites are ordered 
  ψ = orthogonalize(ψ_, start)
  ψH = prime(dag(ψ), "Link")
  ρ = ψ[start] * ψH[start]
  for k in (start + 1):stop
    ρ *= ψ[k]
    ρ *= ψH[k]
  end

  return ρ
end

function get_ee_region(
  ψ::AbstractMPS, region; ee_type=EEType("vN"), mode="auto", kwargs...
)::Real
  """
    Get the entanglement entropy of a region of sites, using either the "site" basis
    or the "link" basis. mode="auto" should select the smallest density matrix to compute
  """

  # TODO: add in calculations to redo sites and mode 
  if mode == "auto"
    is_contiguous = maximum(region) - minimum(region) + 1 == length(region)

    @warn "auto mode currently defaults to sites"
    mode = "sites"
  end

  if mode == "sites"
    ρ = get_density_matrix_sites(ψ, region)
  elseif mode == "bond"
    # TODO: make network friendly
    @assert maximum(region) - minimum(region) + 1 == length(region)
    ρ = get_density_matrix_bond(ψ, region[1], region[end])
  end

  D, U = eigen(ρ; ishermitian=true)
  Sd = Array(diag(D))
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  return compute_ee(ee_type, Sd, kwargs...)
end
