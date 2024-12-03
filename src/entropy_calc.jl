using ITensorMPS: AbstractMPS, orthogonalize, linkinds, siteinds, linkind
using ITensors: svd, diag, prime, dag, eigen

using ITensors: norm, dim
include("singular_value_funcs.jl")

function ee_bipartite(ψ::AbstractMPS, cut::Int; ee_type=EEType("vN"), kwargs...)::Real
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

function density_matrix_sites(ψ_::AbstractMPS, region;)
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

function density_matrix_bond(ψ_::AbstractMPS, start, stop;)
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

function get_best_mode(ψ::AbstractMPS, region; verbose=false)
  """
    Determine the best mode given a region and MPS
    if the region is contiguous then determine site vs bond
    otherwise use the site basis

    TODO: use swap method to force form contiguous
  """
  # is this hacky?
  start, stop = minimum(region), maximum(region)
  is_contiguous = stop - start + 1 == length(region)
  (is_contiguous && verbose) &&
    println("Contiguous region found, considering sites and bond versions")
  (!is_contiguous && verbose) &&
    println("No contiguous region found, future version will consider swap")
  (!is_contiguous) && return "sites"

  # get size of site version
  # we can always do dim(s[region]) but I'm worried about overflow
  s = siteinds(ψ)
  log_sitedim = sum([log2(dim(s[i])) for i in region])
  (verbose) && println("Site density matrix would be size (log2) $log_sitedim")

  # check that one should really give the inverse region, TODO: modify region based on this
  log_inverse = sum([log2(dim(si)) for si in s if si ∉ s[region]])

  log_bonddim = log2(dim(linkinds(ψ, start - 1))) + log2(dim(linkinds(ψ, stop)))
  (verbose) && println("Bond-based density matrix would be size (log2) $log_bonddim")
  if (log_inverse < log_sitedim) && (log_inverse < log_bonddim)
    @warn "The compliment of the requested region ($log_inverse) is smaller than the region and the bond version, you should use that instead"
  end
  (log_bonddim < log_sitedim) && return "bond"

  return "sites"
end

function ee_region(
  ψ::AbstractMPS, region; ee_type=EEType("vN"), mode="auto", verbose=false, kwargs...
)::Real
  """
    Get the entanglement entropy of a region of sites, using either the "site" basis
    or the "link" basis. mode="auto" should select the smallest density matrix to compute
  """

  (length(region) == length(ψ)) && return 0.0

  # TODO: add in calculations to redo sites and mode 
  if mode == "auto"
    mode = get_best_mode(ψ, region; verbose)
  end

  if mode == "sites"
    (verbose) && println("Using site mode")
    ρ = density_matrix_sites(ψ, region)
  elseif mode == "bond"
    # TODO: make network friendly
    (verbose) && println("Using bond mode")
    @assert maximum(region) - minimum(region) + 1 == length(region)
    ρ = density_matrix_bond(ψ, region[1], region[end])
  end

  D, U = eigen(ρ; ishermitian=true)
  Sd = Array(diag(D))
  S_norm = sum(Sd)
  if !(S_norm ≈ 1.0)
    @warn "Normalization of the density matrix isn't 1 (actual=$S_norm)! Be careful!"
  end
  return compute_ee(ee_type, Sd; kwargs...)
end
