# Algorithms for enumerating key candidates based on probability
# tables for subkeys, and for estimating the rank (or guessing
# entropy) of a known correct key

using DataStructures
import Base.isless, Base.show, Base.iterate, Base.length

export KeyEnumerator, depth, nth, value, compare, estimate_rank

# V type of subkey value, e.g. UInt8
# P type of (log-)probability, e.g. Float64

abstract type Candidate{P,V}; end

"a single subkey"
mutable struct Candidate1{P,V} <: Candidate{P,V};
    p :: P   # probability
    v :: V   # value
end

"a concatenation of two subkeys"
mutable struct Candidate2{P,V,
                          C1<:Candidate{P,V},
                          C2<:Candidate{P,V}} <: Candidate{P,V};
    p  :: P
    v1 :: C1
    v2 :: C2
end

isless(c1,c2::Candidate{P,V}) where {P,V} = isless(c1.p,c2.p)
len(::Type{Candidate1{P,V}}) where {P,V} = 1
len(::Type{Candidate2{P,V,C1,C2}}) where {P,V,
                                          C1<:Candidate{P,V},
                                          C2<:Candidate{P,V}} =
                                              len(C1) + len(C2)

"recursively extract value"
function value(c::Candidate2{P,V,C1,C2}) where {P,V,
                                                C1<:Candidate{P,V},
                                                C2<:Candidate{P,V}}
    v = Vector{V}(undef, len(typeof(c)))
    value!(v, c)
    return v
end
function value!(v::AbstractVector{V},
                c::Candidate2{P,V,C1,C2}) where {P,V,
                                                 C1<:Candidate{P,V},
                                                 C2<:Candidate{P,V}}
    l1 = len(typeof(c.v1))
    l2 = len(typeof(c.v2))
    value!(view(v, 1:l1), c.v1)
    value!(view(v, l1+1:l1+l2), c.v2)
end
function value!(v::AbstractVector{V},
                c::Candidate1{P,V}) where {P,V}
    v[1] = c.v
    return
end
function compare(v::AbstractVector{V},
                 c::Candidate2{P,V,C1,C2}) where {P,V,
                                                  C1<:Candidate{P,V},
                                                  C2<:Candidate{P,V}}
    l1 = len(typeof(c.v1))
    l2 = len(typeof(c.v2))
    return compare(view(v, 1:l1), c.v1) &&
        compare(view(v, l1+1:l1+l2), c.v2)
end
function compare(v::AbstractVector{V},
                 c::Candidate1{P,V}) where {P,V}
    return v[1] == c.v
end

function show(c::Candidate)
    show((c.p, value(c)))
end

struct Key{P}
    p  :: P    # needed by BinaryMaxHeap for sorting
    i1 :: Int  # index in v1 and o1
    i2 :: Int  # index in v2 and o2
end
isless(x::Key{P},y::Key{P}) where {P} = isless(x.p, y.p)

"""
Return an iterator that will iterates over all keys in decreasing
order of probability. Each key is formed from `d` independent subkeys,
each of which can have one of `n` different values. The `j`-th value
of subkey `i` is provided in v[i,j] and the logarithm of its
probability in p[i,j], where `1≤i≤d` and `1≤j≤n`.
"""

mutable struct KeyEnumerator{P,V}
    # iterators for 1st and 2nd subkey
    i1 :: Union{KeyEnumerator{P,V}, Vector{<:Candidate{P,V}}}
    i2 :: Union{KeyEnumerator{P,V}, Vector{<:Candidate{P,V}}}
    v1 :: Vector{Candidate{P,V}}   # vector of 1st subkeys iterated so far
    v2 :: Vector{Candidate{P,V}}   # vector of 2nd subkeys iterated so far
    o1 :: BitVector         # occupancy flags for s1
    o2 :: BitVector         # occupancy flags for s2
    F  :: BinaryMaxHeap{Key{P}}  # frontier
    l  :: Float64           # expected length

    function KeyEnumerator{P,V}(i1, i2) where {P,V}
        if isa(i1, AbstractVector{<:Candidate})
            i1 = collect(i1)
            v1 = i1
            c1 = v1[1]
            o1 = falses(length(v1))
            l1 = Float64(length(v1))
        else
            (c1, s1) = iterate(i1)  # we ignore KeyEnumerator state here
            v1 = [c1]
            o1 = falses(1)
            l1 = i1.l
        end
        if isa(i2, AbstractVector{<:Candidate})
            i2 = collect(i2)
            v2 = i2
            c2 = v2[1]
            o2 = falses(length(v2))
            l2 = Float64(length(v2))
        else
            (c2, s2) = iterate(i2)  # deal with iterator state
            v2 = [c2]
            o2 = falses(1)
            l2 = i2.l
        end
        p::P = c1.p + c2.p;
        F = BinaryMaxHeap([Key{P}(p, 1, 1)])
        o1[1] = true
        o2[1] = true
        return new{P,V}(i1, i2, v1, v2, o1, o2, F, l1 * l2)
    end
end

function KeyEnumerator(c::Matrix{<:Candidate{P,V}}) where {P,V}
    # build tree over columns
    e = map((x) -> KeyEnumerator{P,V}(x...),
            Iterators.partition(eachcol(c), 2))
    while length(e) > 1
        e = map((x) -> KeyEnumerator{P,V}(x...),
                Iterators.partition(e, 2))
    end
    return only(e)
end

"""
    function KeyEnumerator(p::AbstractMatrix{P}, v::AbstractMatrix{V})

Create an iterator that enumerates candidates for cryptographic keys
in decreasing order of probability. Each key is a sequence of `d`
independent subkeys and each subkey can take on `n` different values.
Argument `p` is an `n`-by-`d` matrix of logarithms of probabilities
for each value of each subkey, and `v` is an `n`-by-`d` matrix of the
corresponding subkey values.

The iterator `e` returned is mutable and therefore can only be used
once, therefore make a `deepcopy` of it before trying to use it
multiple times. The iterator can be invoked as usual with

    c, s = iterator(e, s)

Here `c` is a handle for a candidate key where `c.p::P` returns the
logarithm of the candidate's probability, and `value(c)::Vector{V}`
returns the value of the key. (The state `s` required by the iteration
protocol is currently just a dummy counter that can be ignored.)
"""
function KeyEnumerator(p::AbstractMatrix{P}, v::AbstractMatrix{V}) where {P,V}
    @assert(size(p) == size(v))
    i = sortperm(p, dims=1, rev=true)
    p = p[i]
    v = v[i]
    c = [Candidate1(p[i,j],v[i,j]) for i=axes(p,1), j=axes(p,2)]
    e = KeyEnumerator(c)
end

len(i::KeyEnumerator) = i.l
len(i::AbstractVector) = Float64(length(i))

# while length is known, length(i) is likely too long for Int
Base.IteratorSize(::Type{KeyEnumerator{P,V}}) where {P,V} = Base.SizeUnknown()

# this iterator mutates and the state just counts
function iterate(e::KeyEnumerator{P,V}, state = 0) where {P,V}
    if isempty(e.F); return end
    k = pop!(e.F)
    e.o1[k.i1] = false
    e.o2[k.i2] = false
    # check field below
    if k.i1 < len(e.i1)
        # extend vectors?
        if length(e.v1) == k.i1
            r = iterate(e.i1)
            if r !== nothing
                (next, _) = r
                push!(e.v1, next)
                push!(e.o1, false)
            end
        end
        # extend frontier?
        if length(e.v1) > k.i1 && !e.o1[k.i1+1] && !e.o2[k.i2]
            push!(e.F, Key{P}(e.v1[k.i1+1].p + e.v2[k.i2].p, k.i1+1, k.i2))
            e.o1[k.i1+1] = true
            e.o2[k.i2]   = true
        end
    end
    # check field to the right
    if k.i2 < len(e.i2)
        # extend vectors?
        if length(e.v2) == k.i2
            r = iterate(e.i2)
            if r !== nothing
                (next, _) = r
                push!(e.v2, next)
                push!(e.o2, false)
            end
        end
        # extend frontier?
        if length(e.v2) > k.i2 && !e.o1[k.i1] && !e.o2[k.i2+1]
            push!(e.F, Key{P}(e.v1[k.i1].p + e.v2[k.i2+1].p, k.i1, k.i2+1))
            e.o1[k.i1]   = true
            e.o2[k.i2+1] = true
        end
    end
    return (Candidate2(k.p, e.v1[k.i1], e.v2[k.i2]), state + 1)
end

"""
    i, p = depth(e::KeyEnumerator, k::AbstractVector{V};
                 maxdepth = nothing) where {V}

Iterate over key enumerator `e` until a key matching `k` is
encountered. Returns the integer position `i` (“depth”, “rank”, or
“guessing entropy”) on the enumerated list of keys, as well as the
logarithm of the probability, `p`, of that key. A return value of `i
== 1` means that `k` matched the most-probable key at the top of the
enumerated list.

Set `maxdepth` to limit after now many keys the search is aborted.

Returns `nothing` if all possible (or `maxdepth` if set) keys have
been enumerated without a match.
"""
function depth(e::KeyEnumerator, v::AbstractVector{V},
               maxdepth = nothing) where {V}
    e = deepcopy(e)
    state = 0
    while (r = iterate(e, state)) !== nothing
        c, state = r
        if compare(v, c)
            return state, c.p
        end
        if maxdepth !== nothing && state >= maxdepth
            return nothing
        end
    end
    return nothing
end

"""
   nth(e::KeyEnumerator, n)

Return the value of the `n`-th key enumerated by `e`
"""
nth(e::KeyEnumerator, n) =
    value(first(Iterators.drop(deepcopy(e),n-1)))


# https://github.com/JuliaDSP/DSP.jl/issues/292
function _conv_kern_direct!(out, u, v)
    fill!(out, 0)
    u_region = CartesianIndices(u)
    v_region = CartesianIndices(v)
    one_index = oneunit(first(u_region))
    for vindex in v_region
        @simd for uindex in u_region
            @inbounds out[uindex + vindex - one_index] += u[uindex] * v[vindex]
        end
    end
    out
end
function _conv_kern_direct(
    u::AbstractArray{T, N}, v::AbstractArray{S, N}, su, sv) where {T, S, N}
    sout = su .+ sv .- 1
    out = similar(u, promote_type(T, S), sout)
    _conv_kern_direct!(out, u, v)
end
function conv_direct(u,v)
    _conv_kern_direct(u, v, size(u), size(v))
end

using StatsBase
using DSP

# key rank estimation using histograms
# https://link.springer.com/chapter/10.1007/978-3-662-48116-5_6
function estimate_rank(p::AbstractMatrix{P}, v::AbstractMatrix{V},
                       k::AbstractVector{V}) where {P,V}
    @assert size(p) == size(v)
    nv = size(p,1)  # number of values per subkey
    ns = size(p,2)  # number of subkeys

    # calculate probability of k
    prob = zero(P)
    for d = axes(v,2)
        i = findfirst(isequal(k[d]), @view v[:,d])
        prob -= p[i,d]
    end

    # build histogram of probabilities
    #edges = StatsBase.histrange(-p[:], 100)
    lo,hi = extrema(-p[:])
    @assert lo == 0.0
    step = round(hi / 60000, sigdigits=3, base=2)
    edges = range(lo, hi+step; step)

    # find bin of probability
    #l = floor(Int, (prob - lo) / step) - 1
    l = findlast((x)->x<=prob, edges)

    H = Int128      # datatype for histogram counters, also return value
    # prepare histogram of probabilities for each subkey
    h = [
        let h = fit(Histogram{H}, -p[:,i], edges).weights
        # truncate each histogram to remove trailing zeros,
        # as well as any probabilities lower than that of the whole key
        resize!(h, min(findlast(!iszero, h), l+ns))
        #@assert sum(h) == nv
        h
        end
        for i = 1:ns
    ]
    for i = 2:ns
        h[1] = conv_direct(h[1], h[i])
        if !isa(H, Integer); h[1] .= round.(h[1]); end # round float counts
    end
    #@assert sum(UInt128.(hc)) == UInt128(size(p,2))^size(p,1)

    # lookup probability in histogram
    hc = h[1]
    low  = sum(view(hc, 1:l-ns))
    est  = sum(view(hc, 1:l))
    high = sum(view(hc, 1:l+ns))
    return est, low, high
end
