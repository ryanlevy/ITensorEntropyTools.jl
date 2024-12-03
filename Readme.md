# ITensorEntropyTools.jl

A set of tools designed to compute entanglement entropy of a MPS.

## Examples
Consider some MPS `p`.

The standard way to calculate the entanglement entropy would be a bipartition cut between two regions, which you get from `get_ee_bipartite(p, cut)`.

However, we generally want to obtain a density matrix of some `region` of sites. We can do this by tracing out all the sites or if the region is contiguous tracing out the sites leaving the bond or link dimensions. 
The function `get_ee_region` will somewhat automatically determine the best way to do this for you

```julia
julia> get_ee_region(p, [2,3,4]; verbose=true)
Contiguous region found, considering sites and bond versions
Site density matrix would be size (log2) 3.0
Bond-based density matrix would be size (log2) 4.0
Using site mode
1.3381099313375202

julia> get_ee_region(p, [2,3,4,5,6]; verbose=true)
Contiguous region found, considering sites and bond versions
Site density matrix would be size (log2) 5.0
Bond-based density matrix would be size (log2) 4.0
Using bond mode
1.37377464991627
```

<img src="images/density_matrices.png"  width="500px" />

There is also support for generalized Renyi entropy, in case you don't want von Neumann all the time
```julia
julia> get_ee_region(p, [2,3,4,5,6]; ee_type=EEType("Renyi"),n=0.1)
1.982722528519394
```

### Entropy Functions

There is support for mutual information between two regions
```math
I(A:B) = S_n(A) + S_n(B) - S_n(A\cup B)
```
```julia
julia> get_mutual_info_region(p,[2,3],[5,6])
0.5235354764420115
```
And tripartite mutual information
```math
I3(A:B:C) = I(A:B) + I(B:C) - I(A:BC)
```
```julia
julia> get_tripartite_ee_region(p,[2,3],[5,7],[8,9])
0.05629360845248765
```

## Credits
This library was written by Ryan Levy, with heavy inspiration from prior work with Abid Khan ([@abid1214](https://github.com/abid1214)) with helpful conversations with [Bryan K Clark](https://clark.physics.illinois.edu/) and [Edgar Solomonik](https://solomonik.cs.illinois.edu/).


## TODO

- [ ] Matrix Free density matrix tools
- [ ] Remove dependence on tags
- [ ] combine site and link codes? 
- [ ] Generalized to ITensorNetworks
