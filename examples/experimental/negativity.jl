using ITensors, ITensorMPS
using ITensorMPS: MPS # in case Metal.jl is used
using ITensorEntropyTools
using ITensorEntropyTools: negativity_region

using Random
Random.seed!(0x5eed)

L = 4
@assert L % 2 == 0

println("\nGHZ Version")

s = siteinds("S=1/2", L)
ghz = MPS(s, "Up") # |0...0>
gates = []
push!(gates, ("H", 1))
# probably could do 1807.05572 linear time
for i in 1:(L - 1)
  push!(gates, ("CNOT", i, i + 1))
end
ghz = apply(ops(s, gates), ghz)

# GHZ state subregion is known to have 0
Ñ = negativity_region(ghz, [1, 2, 3, 4], [3, 4])
@show log2(2 * Ñ + 1)

println("\nGHZ Mixed State Version")
Ñ = negativity_region(ghz, [2, 3], [3])
@show log2(2 * Ñ + 1)

println("\nRandom MPS Version")
# Random states generally have something
p_rand = random_mps(s; linkdims=4)
Ñ = negativity_region(p_rand, [1, 2, 3, 4], [3, 4])
@show log2(2 * Ñ + 1)

println("\nBell Pair Version")
s = siteinds("S=1/2", 2)
bell = MPS(s, "Up") # |0...0>
gates = []
push!(gates, ("H", 1))
push!(gates, ("CNOT", 1, 1 + 1))
bell = apply(ops(s, gates), bell)

Ñ = negativity_region(bell, [1, 2], [2])
@show log2(2 * Ñ + 1)
nothing
