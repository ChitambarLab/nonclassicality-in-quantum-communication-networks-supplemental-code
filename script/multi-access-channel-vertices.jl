# multi-access channel vertices
using BellScenario
using LinearAlgebra
using QBase
using Combinatorics

using Test

"""
    multi_access_vertices()

Computes the bipartite multi-access channel vertices using brute force and taking
the unique combinations.
"""
function multi_access_vertices(X :: Int64, Y :: Int64, Z :: Int64, dA :: Int64, dB :: Int64) :: Vector{Vector{Int64}}

    P_A = BlackBox(dA,X)
    P_B = BlackBox(dB,Y)
    P_C = BlackBox(Z,dA*dB)

    P_A_vertices = deterministic_strategies(P_A)
    P_B_vertices = deterministic_strategies(P_B)
    P_C_vertices = deterministic_strategies(P_C)

    num_verts_raw = dA^X*dB^Y*Z^(dA*dB)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A in P_A_vertices, v_B in P_B_vertices, v_C in P_C_vertices
        V = v_C * kron(v_A,v_B)
        verts[id] = V[1:end-1,:][:]
        id += 1
    end

    unique(verts)
end

"""
    multi_access_num_bit_vertices(X :: Int64, Y :: Int64, Z :: Int64) :: Int64

Counts the number of multi-access channel vertices when no more than a bit is used
per signaling party.
"""
function multi_access_num_bit_vertices(X :: Int64, Y :: Int64, Z :: Int64) :: Int64
    dA = 2
    dB = 2

    # dA = dB = 1 case
    num_vs = Z

    # dA = 2, dB = 1 case or dA = 1, dB = 2
    num_vs += sum(cA -> LocalPolytope.num_vertices(LocalSignaling(X,Z,cA), rank_d_only=true), 2:dA)
    num_vs += sum(cB -> LocalPolytope.num_vertices(LocalSignaling(Y,Z,cB), rank_d_only=true), 2:dB)

    # dA = dB = 2 case
    dC = dA*dB
    for c in 2:min(Z,dC)
        num_non_unique = 0
        if c ≤ dA
            num_non_unique += QMath.stirling2(dA,c)
        end
        if c ≤ dB
            num_non_unique += QMath.stirling2(dB,c)
        end

        num_vs += QMath.stirling2(X,dA)*QMath.stirling2(Y,dB)*(QMath.stirling2(dC,c)-num_non_unique) * binomial(Z,c)*factorial(c)
    end

    num_vs
end

"""
    multi_access_num_vertices(X :: Int64, Y :: Int64, Z :: Int64, dA :: Int64, dB :: Int64) :: Int64

Counts the number of multi-access channel vertices. This method is currently wrong for cases where dA or
dB are not equal to 2.
"""
function multi_access_num_vertices(X :: Int64, Y :: Int64, Z :: Int64, dA :: Int64, dB :: Int64) :: Int64
    println("Warning this method is not working properly for dA >  2 or  dB > 2")

    num_vs = Z

    num_vs += sum(cA -> LocalPolytope.num_vertices(LocalSignaling(X,Z,cA), rank_d_only=true), 2:dA)
    num_vs += sum(cB -> LocalPolytope.num_vertices(LocalSignaling(Y,Z,cB), rank_d_only=true), 2:dB)

    dC = dA*dB
    for c in 2:min(Z,dC)
        num_non_unique = 0
        if c ≤ dA || c ≤ dB
            if c ≤ dA
                for cA in c:dA
                    num_non_unique += QMath.stirling2(dA,cA)
                end
            end
            if c ≤ dB
                for cB in c:dB
                    num_non_unique += QMath.stirling2(dB,cB)
                end
            end

        elseif c ≤ 4
            # for cC in c:4
            #     num_non_unique += QMath.stirling2(dC, cC)
            # end
        end

        num_vs += QMath.stirling2(X,dA)*QMath.stirling2(Y,dB)*(QMath.stirling2(dC,c)-num_non_unique) * binomial(Z,c)*factorial(c)
    end

    num_vs
end

@testset "Testing multi-access channel vertex methods" begin
    # testing that the num_vertices count is correct
    @testset "trivial case" begin
        X = 2    # num inputs Alice
        Y = 2    # num inputs Bob
        Z = 2    # num outputs charlie

        dA = 2   # Alice signaling dimension
        dB = 2   # Bob signaling dimension

        # implemented methods
        vertices = multi_access_vertices(X,Y,Z,dA,dB)  # brute force compute vertices
        num_bit_vertices = multi_access_num_bit_vertices(X,Y,Z)  # compute the number of vertices for dA=dB=2
        num_vertices = multi_access_num_vertices(X,Y,Z,dA,dB)  # TODO: fix the scaling for dA and dB > 2

        @test length(vertices) == num_vertices
        @test length(vertices) == num_bit_vertices
    end

    # testing vertex count over range of quickly computable cases
    @testset "dA=2, dB=2," begin
        @testset "dA=2, dB=2, ($X,$Y,$Z)" for  X  in 2:5, Y in 2:5, Z in 2:5
            dA = 2
            dB = 2

            vertices = multi_access_vertices(X,Y,Z,dA,dB)
            num_vertices = multi_access_num_bit_vertices(X,Y,Z)

            @test length(vertices) == num_vertices
        end
    end
end
