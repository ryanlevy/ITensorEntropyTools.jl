using ITensorMPS: siteinds, random_mps
using ITensorEntropyTools

using Random
Random.seed!(0x5eed)

d = 2
N = 10

s = siteinds(d, N)
p = random_mps(s; linkdims=4)

mutual_info = mutual_info_region(p, [2, 3], [6, 7]; verbose=false)
@show mutual_info

# this is Renyi n=2 entropy instead of von Neumann
mutual_info = mutual_info_region(p, [2, 3], [6, 7]; ee_type=EEType("Renyi"), verbose=false)
@show mutual_info

# this is Renyi n=0.5 entropy instead of von Neumann
mutual_info = mutual_info_region(
  p, [2, 3], [6, 7]; ee_type=EEType("Renyi"), verbose=false, n=0.5
)
@show mutual_info

trimutual_info = tripartite_ee_region(p, [2, 3], [4, 5], [6, 7]; verbose=true)
@show trimutual_info

nothing
