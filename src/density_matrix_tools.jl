using ITensorNetworks: TreeTensorNetwork, orthogonalize, linkinds, siteinds, linkind
using ITensorNetworks:
  a_star,
  BeliefPropagationCache,
  environment,
  update,
  factor,
  QuadraticFormNetwork,
  tensornetwork,
  partitioned_tensornetwork,
  operator_vertex,
  messages,
  default_message,
  default_message_update,
  contract,
  flatten_linkinds,
  underlying_graph,
  src,
  dst,
  vertices,
  noncommoninds
using ITensors: prime, dag

using ITensorNetworks.NamedGraphs:
  NamedGraph, NamedEdge, NamedGraphs, rename_vertices, src, dst
using ITensorNetworks.NamedGraphs.GraphsExtensions: rem_vertex
using ITensorNetworks.NamedGraphs.PartitionedGraphs:
  PartitionEdge, partitionvertices, partitioned_graph, PartitionVertex

function is_contiguous_region(ψ::TreeTensorNetwork, region)::Bool
  """
    Check if the region has the same number of entries as the a* path
  """
  path = a_star(underlying_graph(ψ), region[1], region[end])
  is_contiguous = length(path) + 1 == length(region)
  return is_contiguous
end

function density_matrix_sites(
  ψ_::TreeTensorNetwork, region; (cache!)=nothing, cache_update_kwargs=(;)
)
  """
    Obtain a density matrix, leaving some indices uncontracted
    Assumes very little about tag structure, only site/linkinds
    Currently assume ordered and orthogonality center is within region
  """
  # region = sort(region)
  if length(region) == 1
    start = region[1]
    ψ = orthogonalize(ψ_, start)
    return ψ[start] * prime(dag(ψ[start]), siteinds(ψ)[start])
  end

  #ψH = dag(prime(ψ, linkinds(ψ)..., s[region]...))
  #lᵢ₁ = linkinds(ψ, start - 1) #this is n,n+1 link
  #ρ = prime(ψ[start], lᵢ₁)
  #ρ *= ψH[start]

  #for k in (start + 1):(stop - 1) # replace this with ITensorNetworks iterator
  #  ρ *= ψ[k]
  #  ρ *= ψH[k]
  #end

  #lⱼ = linkinds(ψ, stop)
  #ρ *= prime(ψ[stop], lⱼ)
  #ρ *= ψH[stop]

  ψIψ_bpc = if isnothing(cache!)
    update(BeliefPropagationCache(QuadraticFormNetwork(ψ_)); cache_update_kwargs...)
  else
    cache![]
  end
  ψIψ = tensornetwork(ψIψ_bpc)
  pg = partitioned_tensornetwork(ψIψ_bpc)

  path = PartitionEdge.(a_star(partitioned_graph(ψIψ_bpc), region[1], region[end]))
  for v in region
    pg = rem_vertex(pg, operator_vertex(ψIψ, v))
  end
  ψIψ_bpc_mod = BeliefPropagationCache(pg, messages(ψIψ_bpc), default_message)
  ψIψ_bpc_mod = update(
    ψIψ_bpc_mod, path; message_update=ms -> default_message_update(ms; normalize=false)
  )
  incoming_mts = environment(ψIψ_bpc_mod, [PartitionVertex(region[end])])
  local_state = factor(ψIψ_bpc_mod, PartitionVertex(region[end]))
  rdm = contract(vcat(incoming_mts, local_state); sequence="automatic")
  rdm /= tr(rdm)
  return rdm
end

function density_matrix_bond(ψ_::TreeTensorNetwork, start, stop;)
  """
    Obtain a density matrix, with external link indices
    leaves bond links uncontracted
    Requires the region to be contiguous
  """
  ψ = orthogonalize(ψ_, start)
  ψH = prime(dag(ψ), flatten_linkinds(ψ))
  ρ = ψ[start] * ψH[start]
  path = a_star(underlying_graph(ψ), start, stop)
  for e in path
    # hacking a bit, along the path select
    # toward next edge to not include start
    k = dst(e)
    ρ *= ψ[k]
    ρ *= ψH[k]
  end

  return ρ
end

function get_best_mode(ψ::TreeTensorNetwork, region; verbose=false)
  """
    Determine the best mode given a region and MPS
    if the region is contiguous then determine site vs bond
    otherwise use the site basis

    TODO: use swap method to force form contiguous
  """
  is_contiguous = is_contiguous_region(ψ, region)
  (is_contiguous && verbose) &&
    println("Contiguous region found, considering sites and bond versions")
  (!is_contiguous && verbose) &&
    println("No contiguous region found, future version will consider swap")
  (!is_contiguous) && return "sites"

  # get size of site version
  # we can always do dim(s[region]) but I'm worried about overflow
  s = siteinds(ψ)
  vs = vertices(ψ)
  log_sitedim = sum(i -> log2(dim(s[i])), region)

  # check that one should really give the inverse region
  # since Tr_B[rho_A] = Tr_A[rho_B]
  log_inverse = sum(v -> v ∈ region ? 0.0 : log2(dim(s[v])), vs; init=0.0)

  possible_inds = noncommoninds([inds(ψ[i]) for i in region]..., [s[i] for i in region]...)
  log_bonddim = sum(log2.(dim.(possible_inds)))

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
  ψ::TreeTensorNetwork, region; mode="auto", verbose=false, kwargs...
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
    region_i = [i for i in vertices(ψ) if i ∉ region]
    ρ = density_matrix_sites(ψ, region_i)
  elseif mode == "bond"
    (verbose) && println("Using bond mode")
    @assert is_contiguous_region(ψ, region)
    ρ = density_matrix_bond(ψ, region[1], region[end])
  end

  return ρ
end
