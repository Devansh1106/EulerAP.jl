# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
@muladd begin
#! format: noindent

@inline function minmod(a, b)
    if a * b <= 0
        return zero(a)
    else
        return sign(a) * min(abs(a), abs(b))
    end
end

end # @muladd