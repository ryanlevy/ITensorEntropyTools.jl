using ITensorMPS: siteinds, random_mps
using ITensorEntropyTools

using Random
Random.seed!(55)

d = 2
N = 20

s = siteinds(d, N)

# powers of 2 make this look nice
# by setting to 8, we ensure the biggest region below is at most low rank
p = random_mps(s; linkdims=16)

start = N ÷ 2
for shifts in
    [[0], collect(-1:0), collect(-1:1), collect(-2:1), collect(-2:2), collect(-3:2)]
  region = start .+ shifts
  H0 = ee_region(p, region; ee_type=EEType("Renyi"), n=0) / log(2)
  H1 = ee_region(p, region; ee_type=EEType("Renyi"), n=1) / log(2)
  H2 = ee_region(p, region; ee_type=EEType("Renyi"), n=2) / log(2)
  H∞ = ee_region(p, region; ee_type=EEType("Renyi"), n=Inf) / log(2)
  @show H0, H1, H2, H∞
  # Jensen's Inequality
  @assert H0 >= H1 >= H2 >= H∞
  @assert H2 <= 2 * H∞
end
nothing
