using ITensorMPS: siteinds, random_mps
using ITensorEntropyTools

d = 2
N = 5

s = siteinds(d, N)
p = random_mps(s; linkdims=4)

# noice ee_bipartite[end] ≈ 0, as there is nothing right of that cut
ee_bipartite = [get_ee_bipartite(p, i) for i in 1:N]
@show ee_bipartite

ee_singles = [get_ee_region(p, [i]; mode="sites") for i in 1:N]
@show ee_singles
ee_singles = [get_ee_region(p, [i]; mode="bond") for i in 1:N]
@show ee_singles
ee_doubles = [get_ee_region(p, [i, i + 1]; mode="sites") for i in 1:(N - 1)]
@show ee_doubles
ee_doubles = [get_ee_region(p, [i, i + 1]; mode="bond") for i in 1:(N - 1)]
@show ee_doubles

nothing
