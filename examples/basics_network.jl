using ITensorTreeEntropyTools
using ITensorNetworks: siteinds, ttn, maxlinkdim, random_tensornetwork, norm
using NamedGraphs.NamedGraphGenerators: named_comb_tree

using Random
Random.seed!(42)

d = 2
N = 10

g = named_comb_tree((3, N))
s = siteinds(d, g)

p = random_tensornetwork(s; link_space=4)
p = ttn(p)
p /= norm(p)

ee_cuts = [ee_bipartite(p, (1, i) => (1, i + 1)) for i in 1:(N - 1)]
@show ee_cuts

ee_singles = [ee_region(p, [(1, i)]; mode="sites") for i in 1:N]
@show ee_singles
ee_doubles = [ee_region(p, [(1, i), (1, i + 1)]; mode="sites") for i in 1:(N - 1)]
@show ee_doubles

# one in each leg
ee_trip = [ee_region(p, [(1, i), (2, i + 1), (3, i + 2)]; mode="sites") for i in 1:(N - 2)]
@show ee_trip
