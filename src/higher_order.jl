using ITensors: tr, inds, mapprime

function mutual_info_region(ψ::AbstractMPS, region_A, region_B; kwargs...)::Real
  """
    Get the mutual information of a region of A and B sites
    I(A:B) = Sₙ(A) + Sₙ(B) - Sₙ(A,b)

    See get_ee_region for more options
    """
  S_A = ee_region(ψ, region_A; kwargs...)
  S_B = ee_region(ψ, region_B; kwargs...)
  region_AB = unique(vcat(region_A, region_B))
  S_AB = ee_region(ψ, region_AB; kwargs...)
  return S_A + S_B - S_AB
end

function tripartite_ee_region(ψ::AbstractMPS, region_A, region_B, region_C; kwargs...)::Real
  """
    Get the tripartite mutual information of a region of A,B,C sites
    I3(A:B:C) = I(A:B) + I(A:C) - I(A:BC)

    See get_ee_region for more options
    """
  I_AB = mutual_info_region(ψ, region_A, region_B; kwargs...)
  I_AC = mutual_info_region(ψ, region_A, region_C; kwargs...)
  region_BC = unique(vcat(region_B, region_C))
  I_ABC = mutual_info_region(ψ, region_A, region_BC; kwargs...)
  return I_AB + I_AC - I_ABC
end
