using JuMP
using HiGHS

using Test

include("./MultiAccessChannels.jl")

@testset "4 -> (2,2) -> (4,4) Broadcast Facet"  begin
    p = [0.2,0.5,0.3,-1]
    d1 = [1,0,0,-1]
    d2 = [0.5,0.5,0,-1]
    d3 = [0.5,0,0.5,-1]

    d_set = [d1,d2,d3]

    p = [0.3,0.7,0.5,0.5,0.2,0.3,0.4,0.1,-1]
    v_set = [
        [-1, -1, -1, -1,  1,  1,  1,  1,-1],
        [-1, -1, -1,  1,  1, -1,  1, -1,-1],
        [-1, -1,  1, -1, -1,  1, -1,  1,-1],
        [-1, -1,  1,  1, -1, -1, -1, -1,-1],
        [-1,  1, -1, -1,  1,  1, -1, -1,-1],
        [-1,  1, -1,  1,  1, -1, -1,  1,-1],
        [-1,  1,  1, -1, -1,  1,  1, -1,-1],
        [-1,  1,  1,  1, -1, -1,  1,  1,-1],
        [ 1, -1, -1, -1, -1, -1,  1,  1,-1],
        [ 1, -1, -1,  1, -1,  1,  1, -1,-1],
        [ 1, -1,  1, -1,  1, -1, -1,  1,-1],
        [ 1, -1,  1,  1,  1,  1, -1, -1,-1],
        [ 1,  1, -1, -1, -1, -1, -1, -1,-1],
        [ 1,  1, -1,  1, -1,  1, -1,  1,-1],
        [ 1,  1,  1, -1,  1, -1,  1, -1,-1],
        [ 1,  1,  1,  1,  1,  1,  1,  1,-1],
    ]


    (X, Y, Z, dA, dB) = (4, 4, 4, 2, 2)

    vertices = broadcast_vertices(X,Y,Z,dA,dB)

    # p_mat = [
    #     1 0 0 0;
    #     0 0 0 0;
    #     0 0 0 0;
    #     0 0 0 0;
    #     0 0 0 0;
    #     0 1 0 0;
    #     1 0 0 0;
    #     0 0 0 0;

    #     0 0 0 0;
    #     0 0 0 0;
    #     0 0 1 0;
    #     0 0 0 0;
    #     0 0 0 0;
    #     0 0 0 0;
    #     0 0 0 0;
    # ]
    p_mat = hcat(kron([1;0],[0.5;0;0;0.5],[1;0]), kron([1;0],[0.5;0;0;0.5],[0;1]), kron([0;1],[0.5;0;0;.5],[1;0]), kron([0;1],[0;.5;.5;0],[0;1]))
    pq_mat = hcat(kron([1;0],[1/(2*sqrt(2));(1 - 1/sqrt(2))/2;(1 - 1/sqrt(2))/2;1/(2*sqrt(2))],[1;0]), kron([1;0],[1/(2*sqrt(2));(1 - 1/sqrt(2))/2;(1 - 1/sqrt(2))/2;1/(2*sqrt(2))],[0;1]), kron([0;1],[1/(2*sqrt(2));(1 - 1/sqrt(2))/2;(1 - 1/sqrt(2))/2;1/(2*sqrt(2))],[1;0]), kron([0;1],[(1 - 1/sqrt(2))/2;1/(2*sqrt(2));1/(2*sqrt(2));(1 - 1/sqrt(2))/2],[0;1]))


    p = p_mat[1:end-1,:][:]
    # p = pq_mat[1:end-1,:][:]

    num_v = length(vertices)
    dim_v = length(vertices[1]) + 1 

    model = Model(HiGHS.Optimizer)

    @variable(model, s[1:dim_v])


    for v in vertices
        va = [v..., -1]
        @constraint(model, sum(s.*va) <= 0)
    end

    # @constraint(model, c1, sum(s.*d1) <= 0)
    # @constraint(model, c2, sum(s.*d2) <= 0)
    # @constraint(model, c3, sum(s.*d3) <= 0)




    @constraint(model, c, sum(s.*[p..., -1]) <= 1)

    @objective(model, Max, sum(s.*[p..., -1]))

    optimize!(model)
    println(value.(s))

    bg = convert(BellGame, round.(Int, 2*value.(s)), BlackBox(16,4), rep="normalized")
    bg.β

    objective_value(model)

    verts = Array{Vector{Int}}([])
    for v in vertices
        if sum([v...,-1].*value.(s)) ≈ 0
            push!(verts, convert.(Int, v))
        end
    end
    verts


    BellScenario.dimension(verts)
    BellScenario.dimension(vertices)

    ρ = State([0.5 0.5im;-0.5im 0.5])

    Π = [0.5 0.35im;-0.35im 0.5]

    sqrt(Π) * ρ * sqrt(Π)'

    sqrt(Π) * ρ * sqrt(Π)'

    ρ =State( [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]/2)

    partial_trace(kron([1 0;0 1], sqrt(Π)) * ρ * kron([1 0;0 1], sqrt(Π)')  , [2,2], 1)

    transpose(sqrt(Π)) *[1 0;0 1]/2 * transpose(sqrt(Π)')

    sqrt(Π) *[1 0;0 1]/2 * sqrt(Π)'

    using QBase

    trine_povm = trine_qubit_povm()

    double_trine_povm = POVM(
        [
            kron(trine_povm[1],trine_povm[1]),
            kron(trine_povm[1], trine_povm[2]),
            kron(trine_povm[1], trine_povm[3]),
            kron(trine_povm[2], trine_povm[1]),
            kron(trine_povm[2], trine_povm[2]),
            kron(trine_povm[2], trine_povm[3]),
            kron(trine_povm[3], trine_povm[1]),
            kron(trine_povm[3], trine_povm[2]),
            kron(trine_povm[3], trine_povm[3]),
        ]
    )

    measure(double_trine_povm, State([1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]/2))

    σx * σy * σx

    σy
end

@testset "(3,3) -> (2,2) -> (2,2) -> (3,3) Interference summer game" begin
    (X1, X2, Z1, Z2, dA1, dA2, dB1, dB2) = (3, 3, 3, 2, 2, 2, 2, 2)

    vertices = interference_vertices(X1,X2,Z1,Z2,dA1,dA2,dB1,dB2)

    p1_mat = [
        1 0 0 0 0 0 0 0 0;
        0 1 0 1 0 0 0 0 0;
        0 0 1 0 1 0 1 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
    ]
    p2_mat = [ # multiplication game
        1 1 1 1 0 0 1 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
    ]


    p = p1_mat[1:end-1,:][:]
    p = p2_mat[1:end-1,:][:]

    num_v = length(vertices)
    dim_v = length(vertices[1]) + 1 

    model = Model(HiGHS.Optimizer)

    @variable(model, s[1:dim_v])


    for v in vertices
        va = [v..., -1]
        @constraint(model, sum(s.*va) <= 0)
    end

    # @constraint(model, c1, sum(s.*d1) <= 0)
    # @constraint(model, c2, sum(s.*d2) <= 0)
    # @constraint(model, c3, sum(s.*d3) <= 0)




    @constraint(model, c, sum(s.*[p..., -1]) <= 1)

    @objective(model, Max, sum(s.*[p..., -1]))

    optimize!(model)
    println(value.(s))
    print(round.(Int, 8 * value.(s)))

    bg1 = convert(BellGame, round.(Int, 8*value.(s)), BlackBox(6,9), rep="normalized")
    bg2 = convert(BellGame, round.(Int, 4*value.(s)), BlackBox(6,9), rep="normalized")


    bg.β
    bg1.β

    objective_value(model)

    verts = Array{Vector{Int}}([])
    for v in vertices
        if sum([v...,-1].*value.(s)) ≈ 0
            push!(verts, convert.(Int, v))
        elseif sum([v...,-1].*value.(s)) > 0
            println("not a polytope bound")
        end
    end
    verts


    BellScenario.dimension(verts)
    BellScenario.dimension(vertices)

    println(bg1.game)

    @test all(bg.game == bg1)

    bg.game == bg1
    bg.β


    bg1_game_match = [
        3 0 1 0 0 1 0 1 2;
        0 3 0 2 1 0 0 1 2;
        0 0 4 0 2 0 2 0 0;
        2 1 2 0 1 2 0 2 0;
        2 2 2 1 1 0 0 0 2;
        2 2 3 1 1 1 1 1 1
    ]
end

@testset "(3,3) -> (2,2) -> (2,2) -> (3,3) Interference" begin
    (X1, X2, Z1, Z2, dA1, dA2, dB1, dB2) = (3, 3, 3, 3, 2, 2, 2, 2)

    vertices = interference_vertices(X1,X2,Z1,Z2,dA1,dA2,dB1,dB2)

    p1_mat = I(9)
    p2_mat = [
        6 0 0 0 0 0 0 0 0;
        0 1 1 1 0 1 1 1 0;
        0 1 1 1 0 1 1 1 0;
        0 1 1 1 0 1 1 1 0;
        0 0 0 0 6 0 0 0 0;
        0 1 1 1 0 1 1 1 0;
        0 1 1 1 0 1 1 1 0;
        0 1 1 1 0 1 1 1 0;
        0 0 0 0 0 0 0 0 6;
    ]/6
    p3_mat = [ # multiplication game
        1 0 0 0 0 0 0 0 0;
        0 1 0 1 0 0 0 0 0;
        0 0 1 0 0 0 1 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
    ]
    

    p = p1_mat[1:end-1,:][:]
    # p = p2_mat[1:end-1,:][:]
    # p = p3_mat[1:end-1,:][:]

    num_v = length(vertices)
    dim_v = length(vertices[1]) + 1 

    model = Model(HiGHS.Optimizer)

    @variable(model, s[1:dim_v])


    for v in vertices
        va = [v..., -1]
        @constraint(model, sum(s.*va) <= 0)
    end

    # @constraint(model, c1, sum(s.*d1) <= 0)
    # @constraint(model, c2, sum(s.*d2) <= 0)
    # @constraint(model, c3, sum(s.*d3) <= 0)




    @constraint(model, c, sum(s.*[p..., -1]) <= 1)

    @objective(model, Max, sum(s.*[p..., -1]))

    optimize!(model)
    println(value.(s)*7)
    print(round.(Int, 13/3 * value.(s)))

    # bg1 = convert(BellGame, round.(Int, 7*value.(s)), BlackBox(9,9), rep="normalized")
    bg2 = convert(BellGame, round.(Int, 13/3*value.(s)), BlackBox(9,9), rep="normalized")
    bg3 = convert(BellGame, round.(Int, 7*value.(s)), BlackBox(9,9), rep="normalized")


    bg3.β

    objective_value(model)

    verts = Array{Vector{Int}}([])
    for v in vertices
        if sum([v...,-1].*value.(s)) ≈ 0
            push!(verts, convert.(Int, v))
        elseif sum([v...,-1].*value.(s)) > 0
            println("not a polytope bound")
        end
    end
    verts


    BellScenario.dimension(verts)
    BellScenario.dimension(vertices)

    @test all(bg.game == bg1)

    bg.game == bg1


    bg1_game_match = [
        3  0  0  0  2  0  1  0  1;
        0  3  0  2  0  0  0  1  1;
        0  0  4  1  1  0  1  1  0;
        1  1  2  2  0  0  1  0  1;
        1  1  2  0  2  0  0  1  1;
        1  2  2  0  0  2  1  1  0;
        0  1  3  0  1  1  2  0  0;
        1  0  3  1  0  1  0  2  0;
        2  2  3  1  1  1  1  1  0;
    ]
    bg3_game_match = [
        3  0  0  0  2  0  1  0  1;
        0  3  0  2  0  0  0  1  1;
        0  0  4  1  1  0  1  1  0;
        1  1  2  2  0  0  1  0  1;
        1  1  2  0  2  0  0  1  1;
        1  2  2  0  0  2  1  1  0;
        0  1  3  0  1  1  2  0  0;
        1  0  3  1  0  1  0  2  0;
        2  2  3  1  1  1  1  1  0;
    ]
end