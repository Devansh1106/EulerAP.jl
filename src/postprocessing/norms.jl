# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    ErrorAccumulator(T)

Construct a zero-initialized accumulator.
"""
@inline function ErrorAccumulator(::Type{T}) where {T}
    return ErrorAccumulator(
        zero(T),
        zero(T),
        zero(T)
    )
end


"""
    reset!(acc)

Reset an accumulator.
"""
@inline function reset!(acc::ErrorAccumulator)

    acc.L1 = zero(acc.L1)
    acc.L2 = zero(acc.L2)
    acc.Linf = zero(acc.Linf)

    return nothing
end


"""
    accumulate!(acc, err)

Accumulate the contribution of one cell.
"""
@inline function accumulate!(acc::ErrorAccumulator,
                             err)

    abs_err = abs(err)

    acc.L1 += abs_err
    acc.L2 += abs_err * abs_err
    acc.Linf = max(acc.Linf, abs_err)

    return nothing
end


"""
    finish(acc, cell_volume)

Convert accumulated values into discrete norms.
"""
@inline function finish(acc::ErrorAccumulator,
                        cell_volume)

    return ErrorNorms(
        acc.L1 * cell_volume,
        sqrt(acc.L2 * cell_volume),
        acc.Linf
    )
end

end # @muladd