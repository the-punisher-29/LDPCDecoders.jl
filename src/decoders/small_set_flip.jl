using SparseArrays

# Generate all k-element combinations of `items` without external dependencies.
# Returns a Vector{Vector{T}}.
function _combinations(items::Vector{T}, k::Int) where {T}
    result = Vector{T}[]
    n = length(items)
    k > n && return result
    k == 0 && return [T[]]
    indices = collect(1:k)
    while true
        push!(result, items[indices])
        # Advance to next combination
        i = k
        while i > 0 && indices[i] == n - k + i
            i -= 1
        end
        i == 0 && break
        indices[i] += 1
        for j in i+1:k
            indices[j] = indices[j-1] + 1
        end
    end
    return result
end
# SCRATCH SPACE FOR SSF COMPUTATIONS
struct SmallSetFlipScratchSpace
    "Current error estimate"
    err::Vector{Int}

    "Current syndrome"
    syndrome::Vector{Bool}

    "Scratch buffer for syndrome delta when evaluating a candidate flip"
    syn_delta::Vector{Bool}
end

function SmallSetFlipScratchSpace(s::Int, n::Int)
    return SmallSetFlipScratchSpace(
        zeros(Int, n),
        zeros(Bool, s),
        zeros(Bool, s)
    )
end

"""
    SmallSetFlipDecoder(H, max_iters::Int; t::Int=0)

Small-Set Flip decoder for quantum LDPC codes (especially hypergraph product
and expander codes).

Generalizes the classical bit-flip decoder by flipping *sets* of qubits within
the support of a stabilizer, rather than individual bits. At each iteration the
algorithm greedily selects the flip set that maximally reduces the syndrome
weight.

Based on Algorithm 2 of Leverrier, Tillich & Zémor (2015) [arXiv:1504.00822],
with practical parameters from Grospellier & Krishna (2019) [arXiv:1810.03681].

# Arguments
- `H`: Parity check matrix (`BitMatrix` or `SparseMatrixCSC{Bool,Int}`).
- `max_iters::Int`: Maximum number of greedy flip iterations.
- `t::Int`: Maximum subset size for flip sets (default `0` = use full
  stabilizer support, i.e. all subsets up to the stabilizer weight).
"""
struct SmallSetFlipDecoder <: AbstractDecoder
    "Maximum number of greedy flip iterations"
    max_iters::Int

    "Number of stabilizers (rows of H)"
    s::Int

    "Number of qubits (columns of H)"
    n::Int

    "Maximum subset size parameter"
    t::Int

    "Sparse parity check matrix"
    sparse_H::SparseMatrixCSC{Bool,Int}

    "Pre-computed small sets: small_sets[i] is a vector of subsets of supp(row i)"
    small_sets::Vector{Vector{Vector{Int}}}

    "Pre-computed syndrome columns: syn_columns[i][k] = H[:, small_sets[i][k]] XORed"
    syn_columns::Vector{Vector{Vector{Bool}}}

    "Scratch space for computations"
    scratch::SmallSetFlipScratchSpace
end

function SmallSetFlipDecoder(H::Union{SparseMatrixCSC{Bool,Int}, BitMatrix}, max_iters::Int; t::Int=0)
    s, n = size(H)
    sparse_H = sparse(H)

    # Build the support (list of nonzero column indices) for each row of H
    sparse_HT = sparse(H')
    row_supports = Vector{Vector{Int}}(undef, s)
    for i in 1:s
        cols = Int[]
        for idx in nzrange(sparse_HT, i)
            j = rowvals(sparse_HT)[idx]
            if nonzeros(sparse_HT)[idx]
                push!(cols, j)
            end
        end
        row_supports[i] = cols
    end

    # Determine effective max subset size per stabilizer
    # t=0 means "use full support weight"
    small_sets = Vector{Vector{Vector{Int}}}(undef, s)
    syn_columns = Vector{Vector{Vector{Bool}}}(undef, s)

    for i in 1:s
        supp = row_supports[i]
        w = length(supp)
        max_size = t == 0 ? w : min(t, w)

        # Enumerate all subsets of supp with size 1..max_size
        sets = Vector{Int}[]
        for k in 1:max_size
            for combo in _combinations(supp, k)
                push!(sets, combo)
            end
        end
        small_sets[i] = sets

        # Pre-compute the syndrome contribution (XOR of H columns) for each set
        set_syns = Vector{Bool}[]
        for F in sets
            syn_F = zeros(Bool, s)
            for j in F
                for idx in nzrange(sparse_H, j)
                    r = rowvals(sparse_H)[idx]
                    if nonzeros(sparse_H)[idx]
                        syn_F[r] = syn_F[r] ⊻ true
                    end
                end
            end
            push!(set_syns, syn_F)
        end
        syn_columns[i] = set_syns
    end

    scratch = SmallSetFlipScratchSpace(s, n)

    return SmallSetFlipDecoder(max_iters, s, n, t == 0 ? -1 : t, sparse_H, small_sets, syn_columns, scratch)
end

"""
    reset!(decoder::SmallSetFlipDecoder)

Reset the scratch space between decodings.

# Examples
```jldoctest
julia> decoder = SmallSetFlipDecoder(LDPCDecoders.parity_check_matrix(100, 5, 4), 50);

julia> LDPCDecoders.reset!(decoder);
```
"""
function reset!(decoder::SmallSetFlipDecoder)
    scratch = decoder.scratch
    fill!(scratch.err, 0)
    fill!(scratch.syndrome, false)
    fill!(scratch.syn_delta, false)
    return decoder
end

"""
    decode!(decoder::SmallSetFlipDecoder, syndrome::AbstractVector)

Decode `syndrome` using the Small-Set Flip algorithm.

At each iteration, the algorithm searches over all pre-computed small sets
(subsets of each stabilizer's support) and greedily applies the flip that
maximally reduces the syndrome weight.

Returns `(error_estimate, converged)` where `converged` is `true` if the
syndrome was fully resolved.

# Examples
```jldoctest
julia> using StableRNGs; rng = StableRNG(42);

julia> H = LDPCDecoders.parity_check_matrix(100, 5, 4);

julia> decoder = SmallSetFlipDecoder(H, 50);

julia> error = rand(rng, 100) .< 0.02;

julia> syndrome = Bool.((H * error) .% 2);

julia> guess, success = decode!(decoder, syndrome);
```
"""
function decode!(decoder::SmallSetFlipDecoder, syndrome::AbstractVector)
    reset!(decoder)
    state = decoder.scratch

    # Copy syndrome into working buffer
    for i in 1:decoder.s
        state.syndrome[i] = !iszero(syndrome[i])
    end

    # Check if syndrome is already zero
    syn_weight = count(state.syndrome)
    if syn_weight == 0
        return state.err, true
    end

    for _iter in 1:decoder.max_iters
        best_reduction = 0
        best_stab = 0
        best_set_idx = 0

        # Search over all small sets for the best flip
        for i in 1:decoder.s
            sets = decoder.small_sets[i]
            set_syns = decoder.syn_columns[i]

            for k in eachindex(sets)
                # Compute syndrome weight after flipping set k of stabilizer i
                # new_syndrome = syndrome ⊻ syn_columns[i][k]
                # new_weight = count(new_syndrome)
                # reduction = syn_weight - new_weight
                #
                # Equivalently: reduction = 2 * (number of bits in syn_columns[i][k]
                # that are currently 1 in syndrome) - |syn_columns[i][k]|
                # This avoids allocating/computing the full new syndrome.
                syn_F = set_syns[k]
                reduction = 0
                @inbounds for r in 1:decoder.s
                    if syn_F[r]
                        # This bit of syndrome will be flipped.
                        # If currently 1 → becomes 0 → net -1 (good)
                        # If currently 0 → becomes 1 → net +1 (bad)
                        reduction += state.syndrome[r] ? 1 : -1
                    end
                end

                if reduction > best_reduction
                    best_reduction = reduction
                    best_stab = i
                    best_set_idx = k
                end
            end
        end

        # No improvement found — terminate
        if best_reduction <= 0
            break
        end

        # Apply the best flip
        best_F = decoder.small_sets[best_stab][best_set_idx]
        best_syn_F = decoder.syn_columns[best_stab][best_set_idx]

        # Flip error bits
        @inbounds for j in best_F
            state.err[j] = state.err[j] ⊻ 1
        end

        # Update syndrome
        @inbounds for r in 1:decoder.s
            if best_syn_F[r]
                state.syndrome[r] = !state.syndrome[r]
            end
        end

        syn_weight -= best_reduction

        if syn_weight == 0
            return state.err, true
        end
    end

    return state.err, false
end
