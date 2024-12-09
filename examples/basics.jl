using ITensorMPS: siteinds, random_mps
using ITensorTreeEntropyTools
using ITensorNetworks: siteinds, ttn, maxlinkdim, random_tensornetwork, norm

using Random
Random.seed!(42)

d = 2
N = 10

# Make an MPS and convert it into a tree network
s = siteinds(d, N)
p = random_mps(s; linkdims=4)

p = ttn(p[:])
p /= norm(p)

#ee_cuts = [ee_bipartite(p, i) for i in 1:N]
#@show ee_cuts
#
#ee_singles = [ee_region(p, [i]; mode="sites") for i in 1:N]
#@show ee_singles
#ee_singles = [ee_region(p, [i]; mode="bond") for i in 1:N]
#@show ee_singles
#ee_doubles = [ee_region(p, [i, i + 1]; mode="sites") for i in 1:(N - 1)]
#@show ee_doubles
#ee_doubles = [ee_region(p, [i, i + 1]; mode="bond") for i in 1:(N - 1)]
#@show ee_doubles

ee_cuts = [ee_bipartite(p, (i) => (i + 1)) for i in 1:(N - 1)]
@show ee_cuts

ee_singles = [ee_region(p, [(i)]; mode="sites") for i in 1:N]
@show ee_singles
ee_doubles = [ee_region(p, [(i), (i + 1)]; mode="sites") for i in 1:(N - 1)]
@show ee_doubles

nothing
