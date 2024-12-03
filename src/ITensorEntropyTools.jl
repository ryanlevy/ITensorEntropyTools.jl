module ITensorEntropyTools
include("entropy_calc.jl")
include("higher_order.jl")

export EEType, @EEType_str, compute_ee
export ee_bipartite, ee_region
export mutual_info_region, tripartite_ee_region

end # module ITensorEntropyTools
