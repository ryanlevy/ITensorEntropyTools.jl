module ITensorEntropyTools
using Infinities

include("entropy_calc.jl")
include("higher_order.jl")

export EEType,
  @EEType_str,
  compute_ee,
  ee_bipartite,
  ee_bipartite!,
  ee_region,
  mutual_info_region,
  tripartite_ee_region

end # module ITensorEntropyTools
