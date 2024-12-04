using ITensors.SmallStrings: SmallString
using ITensors: array

# borrowing design from ITensors SiteType
struct EEType{T} end

macro EEType_str(s)
  return EEType{SmallString(s)}
end

EEType(s::AbstractString) = EEType{SmallString(s)}()
EEType(t::SmallString) = EEType{t}()

# These functions all take a vector/diagonal and return
# a float based on the transform

function compute_ee(t::EEType, D; kwargs...)
  # Attempt to convert any stray GPU arrays to CPU
  # This is annoying but unclear
  return compute_ee(t, Array(array(D)); kwargs...)
end

function compute_ee(::EEType"vonNeumann", D::Vector{<:Real}; cutoff=1e-12)
  total = sum(d -> (d > cutoff) ? -d * log(d) : 0.0, D)
  return total
end

function compute_ee(::EEType"vN", args...; kwargs...)
  return compute_ee(EEType("vonNeumann"), args...; kwargs...)
end

function compute_ee(::EEType"Renyi", D::Vector{<:Real}; cutoff=1e-12, n=2)
  (n == 1) && return compute_ee(EEType("VonNeumann"), D, cutoff)

  total = sum(d -> (d > cutoff) ? d^n : 0.0, D)
  return log(total) / (1 - n)
end
