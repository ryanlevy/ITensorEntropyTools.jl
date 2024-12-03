using ITensorMPS: AbstractMPS, orthogonalize, linkinds, siteinds, linkind
using ITensors: prime, dag

function density_matrix_sites(ψ_::AbstractMPS, region;)
  """
    Obtain a density matrix, leaving some indices uncontracted
    Assumes very little about tag structure, only site/linkinds
    Currently assume ordered and orthogonality center is within region
  """
  region = sort(region)
  start, stop = region[1], region[end]
  s = siteinds(ψ_)
  ψ = orthogonalize(ψ_, start)
  if length(region) == 1
    return ψ[start] * prime(dag(ψ[start]), s[start])
  end

  ψH = dag(prime(ψ, linkinds(ψ)..., s[region]...))
  lᵢ₁ = linkinds(ψ, start - 1) #this is n,n+1 link
  ρ = prime(ψ[start], lᵢ₁)
  ρ *= ψH[start]

  for k in (start + 1):(stop - 1) # replace this with ITensorNetworks iterator
    ρ *= ψ[k]
    ρ *= ψH[k]
  end

  lⱼ = linkinds(ψ, stop)
  ρ *= prime(ψ[stop], lⱼ)
  ρ *= ψH[stop]

  return ρ
end

function density_matrix_bond(ψ_::AbstractMPS, start, stop;)
  """
    Obtain a density matrix, with external link indices
    leaves bond links uncontracted
    Requires the region to be contiguous
  """
  # TODO: This should generalize to a network easily
  # but currently assuming the sites are ordered 
  ψ = orthogonalize(ψ_, start)
  ψH = prime(dag(ψ), linkinds(ψ)...)
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

  # check that one should really give the inverse region
  # since Tr_B[rho_A] = Tr_A[rho_B]
  log_inverse = sum([log2(dim(si)) for si in s if si ∉ s[region]])

  log_bonddim = log2(dim(linkinds(ψ, start - 1))) + log2(dim(linkinds(ψ, stop)))

  if verbose
    println("Site density matrix would be size (log2) $log_sitedim")
    println("Complement sites would be size (log2) $log_inverse")
    println("Bond-based density matrix would be size (log2) $log_bonddim")
  end

  # selection logic
  ((log_inverse < log_sitedim) && (log_inverse < log_bonddim)) && return "sites_i"
  (log_bonddim < log_sitedim) && return "bond"

  return "sites"
end

function density_matrix_region(
  ψ::AbstractMPS, region; mode="auto", verbose=false, kwargs...
)
  """ 
    Obtain a density matrix from a region of a MPS
    Uses either the "site" basis or the "link" basis. 
    mode="auto" should select the smallest density matrix to compute
  """
  if mode == "auto"
    mode = get_best_mode(ψ, region; verbose)
  end

  if mode == "sites" || mode == "site"
    (verbose) && println("Using site mode")
    ρ = density_matrix_sites(ψ, region)
  elseif mode == "sites_i"
    (verbose) && println("Using site mode, with inverted region")
    region_i = [i for i in 1:length(ψ) if i ∉ region]
    ρ = density_matrix_sites(ψ, region_i)
  elseif mode == "bond"
    # TODO: make network friendly
    (verbose) && println("Using bond mode")
    @assert maximum(region) - minimum(region) + 1 == length(region)
    ρ = density_matrix_bond(ψ, region[1], region[end])
  end

  return ρ
end
