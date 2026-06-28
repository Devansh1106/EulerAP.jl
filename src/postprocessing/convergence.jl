# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin
#! format: noindent

"""
    experimental_order(error_old, error_new)

Compute the experimental order of convergence.
"""
@inline function experimental_order(error_old,
                                    error_new)

    return log2(error_old / error_new)
end


"""
    convergence_table(cells,
                      results;
                      variable = 1)

Print a convergence table.
"""
function convergence_table(cells,
                           results;
                           variable = 1)

    println()
    println("---------------------------------------------------------------")
    println(" Cells        L1 Error        L2 Error       Linf Error     EOC")
    println("---------------------------------------------------------------")

    previous = nothing

    for (i, N) in enumerate(cells)

        norms = results[i].norms[variable]

        if previous === nothing

            println(
                rpad(N,8),
                lpad(@sprintf("%.6e", norms.L1),18),
                lpad(@sprintf("%.6e", norms.L2),18),
                lpad(@sprintf("%.6e", norms.Linf),18),
                lpad("-",10)
            )

        else

            eoc =
                experimental_order(previous.L2,
                                   norms.L2)

            println(
                rpad(N,8),
                lpad(@sprintf("%.6e", norms.L1),18),
                lpad(@sprintf("%.6e", norms.L2),18),
                lpad(@sprintf("%.6e", norms.Linf),18),
                lpad(@sprintf("%.3f", eoc),10)
            )

        end

        previous = norms

    end

    println("---------------------------------------------------------------")

    return nothing
end

end # @muladd