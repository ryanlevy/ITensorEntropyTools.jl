using ITensors, ITensorMPS
using ITensorEntropyTools

using Random
Random.seed!(0x5eed)

L = 8
@assert L % 2 == 0

println("\nRainbow Version\n")
s = siteinds("Qubit", L)
psi = MPS(s, "Up") # |0...0>

gates = []

for i in 1:(L ÷ 2)
  s1 = i
  s2 = L - i + 1
  push!(gates, ("H", s1))
  push!(gates, ("CNOT", s1, s2))
end
psi = apply(ops(s, gates), psi)

@show maxlinkdim(psi)
# Rainbow "detection"
@show [round(ee_bipartite(psi, i) / log(2); digits=2) for i in 1:(L - 1)]
@show [round(mutual_info_region(psi, [1], [i]) / log(2); digits=2) for i in 1:L]

@show round(
  tripartite_ee_region(
    psi, [L ÷ 2, L ÷ 2 + 1], [L ÷ 2 - 2, L ÷ 2 + 2], [L ÷ 2 - 3, L ÷ 2 + 3]
  ) / log(2),
  digits=2,
)
@show round(tripartite_ee_region(psi, [1, 2], [2, 3], [4, 5]) / log(2), digits=2)
@show round(tripartite_ee_region(psi, [1, 2, 3, 4], [5, 6], [7, 8]) / log(2), digits=2)

println("\nGHZ Version\n")

ghz = MPS(s, "Up") # |0...0>
gates = []
push!(gates, ("H", 1))
# probably could do 1807.05572 linear time
for i in 1:(L - 1)
  push!(gates, ("CNOT", i, i + 1))
end
ghz = apply(ops(s, gates), ghz)

@show maxlinkdim(ghz)
# Rainbow "detection"
@show [round(ee_bipartite(ghz, i) / log(2); digits=2) for i in 1:(L - 1)]
@show [round(mutual_info_region(ghz, [1], [i]) / log(2); digits=2) for i in 1:L]

@show round(
  tripartite_ee_region(
    ghz, [L ÷ 2, L ÷ 2 + 1], [L ÷ 2 - 2, L ÷ 2 + 2], [L ÷ 2 - 3, L ÷ 2 + 3]
  ) / log(2),
  digits=2,
)
@show round(tripartite_ee_region(ghz, [1, 2], [2, 3], [4, 5]) / log(2), digits=2)
@show round(tripartite_ee_region(ghz, [1, 2, 3, 4], [5, 6], [7, 8]) / log(2), digits=2)

println("\nW Version\n")

# hardcode 1807.05572 log time
if L != 8
  L = 8
  s = siteinds("Qubit", L)
end

function ITensors.op(::OpName"Bp", ::SiteType"Qubit")
  return [
    # fixed to p=1/2
    1 0 0 0
    0 1 0 0
    0 0 √(1 / 2) -√(1 - 1 / 2)
    0 0 √(1 - 1 / 2) √(1 / 2)
  ]
end

Ws = MPS(s, "Up") # |0...0>
gates = []
push!(gates, ("X", 1))
for pairs in [(1, 2), (1, 3), (2, 4), (1, 5), (3, 6), (2, 7), (4, 8)]
  push!(gates, ("Bp", pairs...))
  push!(gates, ("CNOT", pairs[2], pairs[1]))
end
Ws = apply(ops(s, gates), Ws)

@show maxlinkdim(Ws)
# Rainbow "detection"
@show [round(ee_bipartite(Ws, i) / log(2); digits=4) for i in 1:(L - 1)]
@show [round(mutual_info_region(Ws, [1], [i]) / log(2); digits=4) for i in 1:L]
@show [round(mutual_info_region(Ws, [1, 2], [i]) / log(2); digits=4) for i in 3:L]

@show round(
  tripartite_ee_region(
    Ws, [L ÷ 2, L ÷ 2 + 1], [L ÷ 2 - 2, L ÷ 2 + 2], [L ÷ 2 - 3, L ÷ 2 + 3]
  ) / log(2),
  digits=4,
)
@show round(tripartite_ee_region(Ws, [1, 2], [2, 3], [4, 5]) / log(2), digits=4)
@show round(tripartite_ee_region(Ws, [1, 2, 3, 4], [5, 6], [7, 8]) / log(2), digits=2)
nothing
