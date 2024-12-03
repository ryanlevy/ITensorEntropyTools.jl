function get_mutual_info_region(ψ::AbstractMPS, region_A, region_B; kwargs...)::Real
  """
    Get the mutual information of a region of A and B sites
    I(A:B) = Sₙ(A)+Sₙ(B) - Sₙ(A,b)

    See get_ee_region for more options
    """
  S_A = get_ee_region(ψ, region_A; kwargs...)
  S_B = get_ee_region(ψ, region_B; kwargs...)
  S_AB = get_ee_region(ψ, vcat(region_A, region_B); kwargs...)
  return S_A + S_B - S_AB
end

function get_tripartite_ee_region(
  ψ::AbstractMPS, region_A, region_B, region_C; kwargs...
)::Real
  """
    Get the tripartite mutual information of a region of A,B,C sites
    I3(A:B:C) = I(A:B) + I(B:C) - I(A:BC)

    See get_ee_region for more options
    """
  I_AB = get_mutual_info_region(ψ, region_A, region_B; kwargs...)
  I_AC = get_mutual_info_region(ψ, region_B, region_C; kwargs...)
  I_ABC = get_mutual_info_region(ψ, region_A, vcat(region_B, region_C); kwargs...)
  return I_AB + I_AC - I_ABC
end
