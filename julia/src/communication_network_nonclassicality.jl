using LinearAlgebra
using Combinatorics
using QBase
using BellScenario
using Base.Iterators: flatten
using SparseArrays

using JuMP
using HiGHS

"""
Enumerates the two-sender multiaccess network polytope vertices using brute force and taking
the unique behaviors.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function multi_access_vertices(
    X1 :: Int64,
    X2 :: Int64,
    Y :: Int64,
    d1 :: Int64,
    d2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1_vertices = sparse.(BellScenario.stirling2_matrices(X1, d1))
    P_A2_vertices = sparse.(BellScenario.stirling2_matrices(X2, d2))
    println("enumerated senders")
    
    P_B = BlackBox(Y, d1*d2)
    P_B_vertices = sparse.(deterministic_strategies(P_B))
    println("enumerated receiver")

    num_verts_raw = length(P_A1_vertices) * length(P_A2_vertices) * length(P_B_vertices)
    println(num_verts_raw)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A1 in P_A1_vertices, v_A2 in P_A2_vertices, v_B in P_B_vertices
        if id % 100000 == 0
            println(id)
        end
        
        V = v_B * kron(v_A1, v_A2)

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end


"""
Enumerates the two-receiver broadcast network polytope vertices using brute force and taking
the unique behaviors.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function broadcast_vertices(
    X :: Int64,
    Y1 :: Int64,
    Y2 :: Int64,
    d1 :: Int64,
    d2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}
    
    num_A_vertices_raw = BellScenario.stirling2(X, d1) * BellScenario.stirling2(X, d2)
    P_A_vertices = Vector{Matrix{Int64}}(undef, num_A_vertices_raw)
    id = 1
    for v_A_d1 in BellScenario.stirling2_matrices(X, d1), v_A_d2 in BellScenario.stirling2_matrices(X, d2)
        v_encoder = hcat(map(i -> kron(v_A_d1[:,i], v_A_d2[:,i]), 1:X)...)

        P_A_vertices[id] = sparse(v_encoder)
        id += 1
    end

    P_B1_vertices = deterministic_strategies(BlackBox(Y1, d1))
    P_B2_vertices = deterministic_strategies(BlackBox(Y2, d2))

    num_verts_raw = length(P_A_vertices) * Y1^d1 * Y2^d2 
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A in P_A_vertices, v_B1 in P_B1_vertices, v_B2 in P_B2_vertices
        V = sparse(kron(v_B1, v_B2)) * v_A

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

"""
Enumerates the hourlgass network polytope vertices using brute force and taking
the unique behaviors. Note that all communication channels are assumed to 
have signaling dimension `d=2`.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function hourglass_network_vertices(
    X1 :: Int64,
    X2 :: Int64,
    Y1 :: Int64,
    Y2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    dA11 = 2
    dA12 = 2
    dA21 = 2
    dA22 = 2

    P_A1_dA11_vertices = BellScenario.stirling2_matrices(X1, dA11)
    P_A1_dA12_vertices = BellScenario.stirling2_matrices(X1, dA12)
    P_A2_dA21_vertices = BellScenario.stirling2_matrices(X2, dA21)
    P_A2_dA22_vertices = BellScenario.stirling2_matrices(X2, dA22)

    P_B1_vertices = deterministic_strategies(BlackBox(Y1, dA11*dA21))
    P_B2_vertices = deterministic_strategies(BlackBox(Y2, dA12*dA22))


    num_encoder_verts_raw = length(P_A1_dA11_vertices) * length(P_A1_dA12_vertices) * length(P_A2_dA21_vertices) * length(P_A2_dA22_vertices)
    println(num_encoder_verts_raw)
    encoder_vertices = Vector{Matrix{Int64}}(undef, num_encoder_verts_raw)

    id = 1
    for v_A1_d11 in P_A1_dA11_vertices, v_A1_d12 in P_A1_dA12_vertices, v_A2_d21 in P_A2_dA21_vertices, v_A2_d22 in P_A2_dA22_vertices

        v_A1 = hcat(map(i -> kron(v_A1_d11[:,i], v_A1_d12[:,i]), 1:X1)...)
        v_A2 = hcat(map(i -> kron(v_A2_d21[:,i], v_A2_d22[:,i]), 1:X2)...)

        encoder_vertices[id] = sparse(kron(v_A1, v_A2))
        id += 1
    end

    num_verts_raw = length(encoder_vertices) * Y1^(dA11*dA21) * Y2^(dA12*dA22)

    println(num_verts_raw)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1

    swap = sparse([1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1])  # swap needed to send d12 to B2 and d21 to B1
    int_layer = kron(I(2), swap ,I(2))
    for v_enc in encoder_vertices, v_B1 in P_B1_vertices, v_B2 in P_B2_vertices
        V = sparse(kron(v_B1,v_B2)) * int_layer * v_enc

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

"""
Enumerates the interference network polytope vertices using brute force and taking
the unique behaviors.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function interference_vertices(
    X1 :: Int64,
    X2 :: Int64,
    Z1 :: Int64,
    Z2 :: Int64,
    dA1 :: Int64,
    dA2 :: Int64,
    dB1 :: Int64,
    dB2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1_vertices = deterministic_strategies(BlackBox(dA1, X1))
    P_A2_vertices = deterministic_strategies(BlackBox(dA2, X2))

    P_B_vertices = deterministic_strategies(BlackBox(dB1*dB2, dA1*dA2))

    P_C1_vertices = deterministic_strategies(BlackBox(Z1, dB1))
    P_C2_vertices = deterministic_strategies(BlackBox(Z2, dB2))


    num_verts_raw = length(P_A1_vertices) * length(P_A2_vertices) * length(P_B_vertices) * length(P_C1_vertices) * length(P_C2_vertices)
    println(num_verts_raw)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A1 in P_A1_vertices, v_A2 in P_A2_vertices, v_B in P_B_vertices, v_C1 in P_C1_vertices, v_C2 in P_C2_vertices
        V = kron(v_C1,v_C2) * v_B * kron(v_A1, v_A2)

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

"""
Enumerates the compressed interference network polytope vertices using brute force and taking
the unique behaviors.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function interference2_vertices(
    X1 :: Int64,
    X2 :: Int64,
    Z1 :: Int64,
    Z2 :: Int64,
    dA1 :: Int64,
    dA2 :: Int64,
    dB :: Int64,
    dC1 :: Int64,
    dC2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1_vertices = deterministic_strategies(BlackBox(dA1,X1))
    P_A2_vertices = deterministic_strategies(BlackBox(dA2,X2))

    P_B1_vertices = deterministic_strategies(BlackBox(dB,dA1*dA2))
    P_B2_vertices = deterministic_strategies(BlackBox(dC1*dC2,dB))

    P_C1_vertices = deterministic_strategies(BlackBox(Z1,dC1))
    P_C2_vertices = deterministic_strategies(BlackBox(Z2,dC2))

    num_verts_raw = dA1^X1 * dA2^X2 * dB^(dA1*dA2)*(dC1*dC2)^(dB) * Z1^dC1 * Z2^dC2
    println(num_verts_raw)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A1 in P_A1_vertices, v_A2 in P_A2_vertices, v_B1 in P_B1_vertices, v_B2 in P_B2_vertices, v_C1 in P_C1_vertices, v_C2 in P_C2_vertices
        V = kron(v_C1,v_C2) * v_B2 * v_B1 * kron(v_A1, v_A2)

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

"""
Enumerates the butterfly network polytope vertices using brute force and taking
the unique behaviors.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function butterfly_vertices(
    X1 :: Int64,
    X2 :: Int64,
    Z1 :: Int64,
    Z2 :: Int64,
    dA1 :: Int64,
    dA2 :: Int64,
    dA3 :: Int64,
    dA4 :: Int64,
    dB :: Int64,
    dC1 :: Int64,
    dC2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1_dA1_vertices = BellScenario.stirling2_matrices(X1, dA1)
    P_A1_dA2_vertices = BellScenario.stirling2_matrices(X1, dA2)
    P_A2_dA3_vertices = BellScenario.stirling2_matrices(X2, dA3)
    P_A2_dA4_vertices = BellScenario.stirling2_matrices(X2, dA4)

    P_B1_vertices = deterministic_strategies(BlackBox(dB,dA2*dA3))
    P_B2_vertices = deterministic_strategies(BlackBox(dC1*dC2,dB))
    P_C1_vertices = deterministic_strategies(BlackBox(Z1,dC1*dA1))
    P_C2_vertices = deterministic_strategies(BlackBox(Z2,dC2*dA4))

    num_interference_node_verts_raw = dB^(dA2*dA3)*(dC1*dC2)^(dB)
    interference_node_vertices = Vector{Matrix{Int64}}(undef, num_interference_node_verts_raw)
    id = 1
    for v_B1 in P_B1_vertices, v_B2 in P_B2_vertices
        interference_node_vertices[id] = sparse(kron(I(dA1), v_B2 * v_B1, I(dA4)))
        id += 1
    end
    interference_node_vertices = unique(interference_node_vertices)
    println(length(interference_node_vertices))
    println(size(interference_node_vertices[1]))

    num_encoder_verts_raw = length(P_A1_dA1_vertices) * length(P_A1_dA2_vertices) * length(P_A2_dA3_vertices) * length(P_A2_dA4_vertices)
    println(num_encoder_verts_raw)
    encoder_vertices = Vector{Matrix{Int64}}(undef, num_encoder_verts_raw)

    id = 1
    for v_A1_d1 in P_A1_dA1_vertices, v_A1_d2 in P_A1_dA2_vertices, v_A2_d3 in P_A2_dA3_vertices, v_A2_d4 in P_A2_dA4_vertices

        v_A1 = hcat(map(i -> kron(v_A1_d1[:,i], v_A1_d2[:,i]), 1:X1)...)
        v_A2 = hcat(map(i -> kron(v_A2_d3[:,i], v_A2_d4[:,i]), 1:X2)...)

        encoder_vertices[id] = sparse(kron(v_A1, v_A2))
        id += 1
    end
    println("length of encoder vertices  : ", length(encoder_vertices[1]))
    encoder_vertices = unique(encoder_vertices)
    println(length(encoder_vertices))
    println("size encoder ", size(encoder_vertices[1]) )

    num_decoder_vertices_raw = Z1^(dC1*dA1) * Z2^(dC2*dA4)
    decoder_vertices = Vector{Matrix{Int64}}(undef, num_decoder_vertices_raw)
    id = 1
    for v_C1 in P_C1_vertices, v_C2 in P_C2_vertices
        decoder_vertices[id] = sparse(kron(v_C1,v_C2))

        id += 1
    end
    decoder_vertices = unique(decoder_vertices)
    println("num decoder vertices = ", length(decoder_vertices))

    num_decoder_vertices_raw2 = length(decoder_vertices) * length(interference_node_vertices)
    decoder_vertices2 = Vector{Matrix{Int64}}(undef, num_decoder_vertices_raw2)
    id = 1
    for v_C in decoder_vertices, v_B in interference_node_vertices
        decoder_vertices2[id] = sparse(v_C * v_B)

        id += 1
    end
    decoder_vertices2 = unique(decoder_vertices2)
    println("num decoder vertices2 = ", length(decoder_vertices2))

    num_verts_raw = length(encoder_vertices) * length(decoder_vertices2)
    println(num_verts_raw)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A in encoder_vertices, v_BC in decoder_vertices2
        V = sparse(v_BC * v_A)

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        if id % 100000 == 0
            println(id)
        end


        id += 1
    end

    unique(verts)
end

"""
Enumerates the butterflied interference network polytope vertices using brute force and taking
the unique behaviors.

If `normalize=true`, then the vertices are returned in the normalized representation in which
the last row of the determistic behavior matrix is omitted.
"""
function butterfly_vertices_if(
    X1 :: Int64,
    X2 :: Int64,
    Z1 :: Int64,
    Z2 :: Int64,
    dA1 :: Int64,
    dA2 :: Int64,
    dA3 :: Int64,
    dA4 :: Int64,
    dB1 :: Int64,
    dB2 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1_dA1_vertices = BellScenario.stirling2_matrices(X1, dA1)
    P_A1_dA2_vertices = BellScenario.stirling2_matrices(X1, dA2)
    P_A2_dA3_vertices = BellScenario.stirling2_matrices(X2, dA3)
    P_A2_dA4_vertices = BellScenario.stirling2_matrices(X2, dA4)

    P_B_vertices = deterministic_strategies(BlackBox(dB1*dB2,dA2*dA3))

    P_C1_vertices = deterministic_strategies(BlackBox(Z1,dB1*dA1))
    P_C2_vertices = deterministic_strategies(BlackBox(Z2,dB2*dA4))

    num_interference_node_verts_raw = length(P_B_vertices)
    interference_node_vertices = Vector{Matrix{Int64}}(undef, num_interference_node_verts_raw)
    id = 1
    for v_B in P_B_vertices
        interference_node_vertices[id] = sparse(kron(I(dA1), v_B, I(dA4)))
        id += 1
    end
    interference_node_vertices = unique(interference_node_vertices)
    println(length(interference_node_vertices))

    num_encoder_verts_raw = length(P_A1_dA1_vertices) * length(P_A1_dA2_vertices) * length(P_A2_dA3_vertices) * length(P_A2_dA4_vertices)
    println(num_encoder_verts_raw)
    encoder_vertices = Vector{Matrix{Int64}}(undef, num_encoder_verts_raw)

    id = 1
    for v_A1_d1 in P_A1_dA1_vertices, v_A1_d2 in P_A1_dA2_vertices, v_A2_d3 in P_A2_dA3_vertices, v_A2_d4 in P_A2_dA4_vertices

        v_A1 = hcat(map(i -> kron(v_A1_d1[:,i], v_A1_d2[:,i]), 1:X1)...)
        v_A2 = hcat(map(i -> kron(v_A2_d3[:,i], v_A2_d4[:,i]), 1:X2)...)

        encoder_vertices[id] = sparse(kron(v_A1, v_A2))
        id += 1
    end
    println("length of encoder vertices  : ", length(encoder_vertices[1]))
    encoder_vertices = unique(encoder_vertices)
    println(length(encoder_vertices))

    num_decoder_vertices_raw = Z1^(dB1*dA1) * Z2^(dB2*dA4)
    decoder_vertices = Vector{Matrix{Int64}}(undef, num_decoder_vertices_raw)
    id = 1
    for v_C1 in P_C1_vertices, v_C2 in P_C2_vertices
        decoder_vertices[id] = sparse(kron(v_C1,v_C2))

        id += 1
    end
    decoder_vertices = unique(decoder_vertices)
    println("num decoder vertices = ", length(decoder_vertices))

    num_decoder_vertices_raw2 = length(decoder_vertices) * length(interference_node_vertices)
    decoder_vertices2 = Vector{Matrix{Int64}}(undef, num_decoder_vertices_raw2)
    id = 1
    for v_C in decoder_vertices, v_B in interference_node_vertices
        decoder_vertices2[id] = sparse(v_C * v_B)

        id += 1
    end
    decoder_vertices2 = unique(decoder_vertices2)
    println("num decoder vertices2 = ", length(decoder_vertices2))

    num_verts_raw = length(encoder_vertices) * length(decoder_vertices2)
    println(num_verts_raw)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A in encoder_vertices, v_BC in decoder_vertices2
        V = sparse(v_BC * v_A)
        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        if id % 100000 == 0
            println(id)
        end

        id += 1
    end

    unique(verts)
end

"""
Enumerates the verticess for the Bacon and Toner scenario (https://arxiv.org/abs/quant-ph/0208057)
where one bit of communication is allowed from either A to B or from B to A.
"""
function bacon_and_toner_vertices(X1,X2,Y1,Y2, normalize=true)

    P_A1 = BlackBox(2*Y1, X1)
    P_B1 = BlackBox(Y2, 2*X2)

    P_A2 = BlackBox(Y1, 2*X1)
    P_B2 = BlackBox(2*Y2, X2)


    P_A1_vertices = deterministic_strategies(P_A1)
    P_B1_vertices = deterministic_strategies(P_B1)
    P_A2_vertices = deterministic_strategies(P_A2)
    P_B2_vertices = deterministic_strategies(P_B2)

    num_verts_raw = (2*Y1)^X1 * Y2^(2*X2) + (Y1)^(2*X1) * (2*Y2)^X2

    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A1 in P_A1_vertices, v_B1 in P_B1_vertices

        V1 = kron(I(Y1), v_B1) * kron(v_A1, I(X2))

        verts[id] = normalize ? V1[1:end-1,:][:] : V1[:]

        id += 1
    end

    for v_A2 in P_A2_vertices, v_B2 in P_B2_vertices
        V2 = kron(v_A2, I(Y1)) * kron(I(X1), v_B2)

        verts[id] = normalize ? V2[1:end-1,:][:] : V2[:]

        id += 1
    end

    unique(verts)
end

"""
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

    dC = dA*dB
    for c in 2:min(Z,dC)
        num_non_unique = 0
        if c ≤ dA
            num_non_unique += QBase.stirling2(dA,c)
        end
        if c ≤ dB
            num_non_unique += QBase.stirling2(dB,c)
        end

        num_vs += QBase.stirling2(X,dA)*QBase.stirling2(Y,dB)*(QBase.stirling2(dC,c)-num_non_unique) * binomial(Z,c)*factorial(c)
    end

    num_vs
end

"""
Enumerates the input or output permutation set for two independent parties.
"""
function local_bipartite_permuatations(X1 :: Int64, X2 :: Int64)

    prod_perms = Vector{Vector{Int64}}(undef, factorial(X1)*factorial(X2))
    prod_perm_id = 1

    for x1_perm in permutations(1:X1), x2_perm in permutations(1:X2)
        prod_perm = zeros(Int64, X1*X2)

        for x1 in 1:X1, x2 in 1:X2
            # find id in product
            prod_id = n_product_id([x1,x2],[X1,X2])
            perm_prod_id = n_product_id([x1_perm[x1], x2_perm[x2]], [X1,X2])

            # set as id of contents
            prod_perm[prod_id] = perm_prod_id
        end

        prod_perms[prod_perm_id] = prod_perm
        prod_perm_id += 1
    end

    prod_perms
end

"""
Enumerates the input or output permutation set for three independent parties.
"""
function local_tripartite_permutations(X :: Int64, Y :: Int64, Z :: Int64)

    prod_perms = Vector{Vector{Int64}}(undef, factorial(X)*factorial(Y)*factorial(Z))
    prod_perm_id = 1

    for x_perm in permutations(1:X), y_perm in permutations(1:Y), z_perm in permutations(1:Z)
        prod_perm = zeros(Int64, X*Y*Y)

        for x in 1:X, y in 1:Y, z in 1:Z
            # find id in product
            prod_id = n_product_id([x,y,z],[X,Y,Z])
            perm_prod_id = n_product_id([x_perm[x],y_perm[y],z_perm[z]],[X,Y,Z])

            # set as id of contents
            prod_perm[prod_id] = perm_prod_id
        end

        prod_perms[prod_perm_id] = prod_perm
        prod_perm_id += 1
    end

    prod_perms
end

"""
Enumerates the input or output permutation set for four independent parties.
"""
function local_quadpartite_permutations(W :: Int64, X :: Int64, Y :: Int64, Z :: Int64)

    prod_perms = Vector{Vector{Int64}}(undef, factorial(W)*factorial(X)*factorial(Y)*factorial(Z))
    prod_perm_id = 1

    for w_perm in permutations(1:W), x_perm in permutations(1:X), y_perm in permutations(1:Y), z_perm in permutations(1:Z)
        prod_perm = zeros(Int64, W*X*Y*Y)

        for w in 1:W, x in 1:X, y in 1:Y, z in 1:Z
            # find id in product
            prod_id = n_product_id([w,x,y,z],[W,X,Y,Z])
            perm_prod_id = n_product_id([w_perm[w],x_perm[x],y_perm[y],z_perm[z]],[W,X,Y,Z])

            # set as id of contents
            prod_perm[prod_id] = perm_prod_id
        end

        prod_perms[prod_perm_id] = prod_perm
        prod_perm_id += 1
    end

    prod_perms
end

"""
Given each party's id and input, return the id of the matrix element in the global tensor product.
"""
function n_product_id(party_ids :: Vector{Int64}, party_inputs::Vector{Int64}) :: Int64
    num_ids = length(party_ids)
    if !all(x -> party_inputs[x] ≥ party_ids[x] ≥ 1, 1:num_ids)
        throw(DomainError(party_inputs, "not all num inputs are valid"))
    elseif num_ids != length(party_inputs)
        throw(DomainError(party_ids, "id does not contain the right number of elements"))
    end

    id = 1
    multiplier = 1
    for i in num_ids:-1:1
        multiplier *= (i+1 > num_ids) ? 1 : party_inputs[i+1]
        id += (party_ids[i] - 1) * multiplier
    end

    id
end

"""
Classical network polytopes are invariant under relabeling the inputs and outputs.
Thus, classcal network polytopes have a permutation symmetry, which can be exploited
to describe the polytope using only a small set of generator facet inequalities. Each
generator facet inequality designates facet class where all facet inequalities in the class
can be obtained by permuting the inputs and outputs of the generator facet.

This function takes a set of input permuations, output permuations, and facet ineuqalities represented
as `BellScenario.BellGame` objects and outptus a dictionary grouping each facet into its distinct class. 
"""
function polytope_facet_classes(input_perms, output_perms, bell_games)
    num_perms = length(input_perms)*length(output_perms)

    facet_class_dict = Dict{Int64, Vector{Matrix{Int64}}}()
    facet_class_id = 1

    for bell_game in bell_games
        facet_considered = false

        for facet_class in values(facet_class_dict)
            if bell_game in facet_class
                facet_considered = true
                break
            end
        end

        if !facet_considered
            # construct all unique permutations
            perms = Array{Matrix{Int64}}(undef, num_perms)
            id = 1
            for input_perm in input_perms, output_perm in output_perms
                perms[id] = bell_game[output_perm, input_perm]
                id += 1
            end

            # add facet to facet dictionary
            facet_class_dict[facet_class_id] = unique(perms)
            facet_class_id += 1
        end
    end

    facet_class_dict
end

"""
For a multiaccess network with input sizes `X1` and `X2` and output size `Y` and a list of `bell_games`
representing classcial network polytope facet inequalities, partition the set of nonclassciality witnesses
into canonical facet classes..
"""
function bipartite_multiaccess_facet_classes(X1, X2, Y, bell_games :: Vector{BellScenario.BellGame})
    input_perms = local_bipartite_permuatations(X1,X2)
    output_perms = permutations(1:Y)
    
    return polytope_facet_classes(input_perms, output_perms, bell_games)
end

"""
For a broadcast network with input size `X` and output sizes `Y1` and `Y2` and a list of `bell_games`
representing classcial network polytope facet inequalities, partition the set of nonclassciality witnesses
into canonical facet classes. Each facet class is represented by a generator facet inequality and all other
facet inequalities in the class can be obtained by permuting the input and outputs.
"""
function bipartite_broadcast_facet_classes(X, Y1, Y2, bell_games :: Vector{BellScenario.BellGame})
    output_perms = local_bipartite_permuatations(Y1, Y2)
    input_perms =  permutations(1:X)
    
    return polytope_facet_classes(input_perms, output_perms, bell_games)
end

"""
For a mulipoint network with input sizes `X1` and `X2` and output sizes `Y1` and `Y2` and a list of `bell_games`
representing classcial network polytope facet inequalities, partition the set of nonclassciality witnesses
into canonical facet classes. Each facet class is represented by a generator facet inequality and all other
facet inequalities in the class can be obtained by permuting the input and outputs.
"""
function bipartite_interference_facet_classes(X1,X2,Y1,Y2, bell_games)
    output_perms = local_bipartite_permuatations(Y1, Y2)
    input_perms = local_bipartite_permuatations(X1, X2)

    return polytope_facet_classes(input_perms, output_perms, bell_games)
end
