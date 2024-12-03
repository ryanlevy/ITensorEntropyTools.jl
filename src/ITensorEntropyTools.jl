module ITensorEntropyTools
include("entropy_calc.jl")
include("higher_order.jl")

export EEType, @EEType_str, compute_ee
export get_ee_bipartite, get_ee_region
export get_mutual_info_region, get_tripartite_ee_region

end # module ITensorEntropyTools
