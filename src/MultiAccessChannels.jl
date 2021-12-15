using LinearAlgebra
using Combinatorics
using QBase
using BellScenario
using Base.Iterators: flatten

using Convex
using SCS

"""
    multi_access_vertices()

Computes the bipartite multi-access channel vertices using brute force and taking
the unique combinations.
"""
function multi_access_vertices(
    X :: Int64,
    Y :: Int64,
    Z :: Int64,
    dA :: Int64,
    dB :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

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

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

function three_sender_multi_access_vertices(
    X1 :: Int64,
    X2 :: Int64,
    X3 :: Int64,
    Z :: Int64,
    d1 :: Int64,
    d2 :: Int64,
    d3 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1 = BlackBox(d1, X1)
    P_A2 = BlackBox(d2, X2)
    P_A3 = BlackBox(d3, X3)

    P_B = BlackBox(Z, d1*d2*d3)

    P_A1_vertices = deterministic_strategies(P_A1)
    P_A2_vertices = deterministic_strategies(P_A2)
    P_A3_vertices = deterministic_strategies(P_A3)

    P_B_vertices = deterministic_strategies(P_B)

    num_verts_raw = d1^X1 * d2^X2 * d3^X3 * Z^(d1*d2*d3)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A1 in P_A1_vertices, v_A2 in P_A2_vertices, v_A3 in P_A3_vertices, v_B in P_B_vertices

        V = v_B * kron(v_A1,v_A2,v_A3)

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

function four_sender_multi_access_vertices(
    X1 :: Int64,
    X2 :: Int64,
    X3 :: Int64,
    X4 :: Int64,
    Z :: Int64,
    d1 :: Int64,
    d2 :: Int64,
    d3 :: Int64,
    d4 :: Int64;
    normalize=true :: Bool
) :: Vector{Vector{Int64}}

    P_A1 = BlackBox(d1, X1)
    P_A2 = BlackBox(d2, X2)
    P_A3 = BlackBox(d3, X3)
    P_A4 = BlackBox(d4, X4)

    P_B = BlackBox(Z, d1*d2*d3*d4)

    P_A1_vertices = deterministic_strategies(P_A1)
    P_A2_vertices = deterministic_strategies(P_A2)
    P_A3_vertices = deterministic_strategies(P_A3)
    P_A4_vertices = deterministic_strategies(P_A4)

    P_B_vertices = deterministic_strategies(P_B)

    num_verts_raw = d1^X1 * d2^X2 * d3^X3 * d4^X4 * Z^(d1*d2*d3*d4)
    verts = Vector{Vector{Int64}}(undef, num_verts_raw)

    id = 1
    for v_A1 in P_A1_vertices, v_A2 in P_A2_vertices, v_A3 in P_A3_vertices, v_A4 in P_A4_vertices, v_B in P_B_vertices

        V = v_B * kron(v_A1, v_A2, v_A3, v_A4)

        verts[id] = normalize ? V[1:end-1,:][:] : V[:]

        id += 1
    end

    unique(verts)
end

"""
    finger_printing_game(num_senders, num_in)

Constructs the finger printing `Game` for the multiple access channel
with `num_senders` each having `num_in` inputs.

The game is won if the multiple access channel correctly guesses whether all
senders are given the same input or not.
"""
function finger_printing_game(num_senders, num_in)
    success_tensor = zeros((ones(Int, num_senders)*num_in)...)
    for i in 1:num_in
        success_tensor[(ones(Int, num_senders)*i)...] = 1
    end

    success_row = Int.(success_tensor[:])
    error_row = map(el -> Int(el + 1)%2, success_row)

    game_matrix = [success_row'; error_row']
    bound = 1 + (num_in^num_senders - num_in)

    return Game(game_matrix, bound)
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

"""
function to perform permutations on a tensor product structure


"""
function multi_access_input_permutations(X :: Int64, Y :: Int64)

    prod_perms = Vector{Vector{Int64}}(undef, factorial(X)*factorial(Y))
    prod_perm_id = 1

    for x_perm in permutations(1:X), y_perm in permutations(1:Y)
        prod_perm = zeros(Int64, X*Y)

        for x in 1:X, y in 1:Y
            # find id in product
            prod_id = n_product_id([x,y],[X,Y])
            perm_prod_id = n_product_id([x_perm[x],y_perm[y]],[X,Y])

            # set as id of contents
            prod_perm[prod_id] = perm_prod_id
        end

        prod_perms[prod_perm_id] = prod_perm
        prod_perm_id += 1
    end

    prod_perms
end

function multi_access_output_permutations(Z :: Int64)
    permutations(1:Z)
end

function tripartite_input_permutations(X :: Int64, Y :: Int64, Z :: Int64)

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

function quadpartite_input_permutations(W :: Int64, X :: Int64, Y :: Int64, Z :: Int64)

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


function n_product_id(party_ids :: Vector{Int64}, party_inputs::Vector{Int64})
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

function facet_classes(X, Y, Z, bell_games :: Vector{BellScenario.BellGame})
    input_perms = multi_access_input_permutations(X,Y)
    output_perms = multi_access_output_permutations(Z)
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

function multi_access_optimize_measurement(
    X,Y,Z,dA,dB,
    game::BellGame,
    ρA_states::Vector{<:State},
    ρB_states::Vector{<:State},
) :: Dict
    # if scenario.X != length(ρ_states)
    #     throw(DomainError(scenario, "expected length of `ρ_states` is $(scenario.X)), but got $(length(ρ_states)) instead"))
    # end
    #
    # if size(ρ_states[1]) != (scenario.d,scenario.d)
    #     throw(DomainError(ρ_states, "dimension of `ρ_states` is not $(scenario.d)"))
    # end

    ρ_states = collect(flatten( map( ρA -> map(ρB -> kron(ρA,ρB), ρB_states), ρA_states)))


    norm_game_vector = convert(Vector{Int64}, game)
    norm_bound = norm_game_vector[end]
    norm_game = reshape(norm_game_vector[1:(end-1)], (Z-1, X*Y))

    # add povm variables and constraints
    Π_vars = map(i -> HermitianSemidefinite(dA*dB), 1:Z)
    constraints = (sum(map(Π_y -> real(Π_y), Π_vars)) == Matrix{Float64}(I, dA*dB, dA*dB))
    constraints += (sum(map(Π_y -> imag(Π_y), Π_vars)) == zeros(Float64, dA*dB, dA*dB))

    # sum up the state contibutions for each row
    H_y = map(row_id -> sum(norm_game[row_id,:] .* ρ_states), 1:Z-1)

    # add the objective
    objective = maximize(real(tr(sum(Π_vars[1:end-1] .* H_y))), constraints)

    # optimize model
    solve!(objective, SCS.Optimizer(verbose=0))

    # parse/return results
    score = objective.optval
    violation = score - norm_bound
    # Π_opt = _opt_vars_to_povm(map(Π_y -> Π_y.value, Π_vars))
    Π_opt = map(Π_y -> Π_y.value, Π_vars)

    Dict(
        "violation" => violation,
        "povm" => Π_opt,
        "game" => game,
        "scenario" => (X,Y,Z,dA,dB),
        "states" => ρ_states
    )
end
