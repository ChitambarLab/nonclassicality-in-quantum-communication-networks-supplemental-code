using Test
using BellScenario

include("../src/MultiAccessChannels.jl")

@testset "(3,3)->(2,2)->2 qmac fingerprinting is local " begin
    
    vertices = multi_access_vertices(3,3,2,2,2)

    diff_test = [
        3 1 1 1 3 1 1 1 3;
        1 3 3 3 1 3 3 3 1;
    ] / 4

    raw_game = optimize_linear_witness(vertices, diff_test[1,:][:])

    @test raw_game ≈ [0,0,0,0,0,0,0,0,0,0] 
end


@testset "(4,4)->(2,2)->2 MAC" begin
    
    vertices = multi_access_vertices(4,4,2,2,2)

    fp_test = [
        1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1;
        0 1 1 1 1 0 1 1 1 1 0 1 1 1 1 0;
    ]

    raw_game = optimize_linear_witness(vertices, fp_test[1,:][:])

    # bell_game = convert(BellGame, round.(Int, 2.6*raw_game), BlackBox(2,16), rep="normalized")
    bell_game = convert(BellGame, round.(Int, 3*raw_game), BlackBox(2,16), rep="normalized")

    bell_game.β

    verts = Array{Vector{Int}}([])
    for v in vertices
        if isapprox(sum([v...,-1].*raw_game), 0, atol=1e-6)
            push!(verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game) > 0
            println("not a polytope bound")
        end
    end
    verts

    BellScenario.dimension(verts)
    BellScenario.dimension(vertices)
end

@testset "(3,3)->(2,2)->3" begin
    
    vertices = multi_access_vertices(3,3,3,2,2)

    diff_test = [
        1 0 0 0 1 0 0 0 1;
        0 1 0 1 0 1 0 1 0;
        0 0 1 0 0 0 1 0 0;
    ]

    raw_game = optimize_linear_witness(vertices, diff_test[1:2,:][:])
    bell_game = convert(BellGame, round.(Int, 2*raw_game), BlackBox(3,9), rep="normalized")


    verts = Array{Vector{Int}}([])
    for v in vertices
        if isapprox(sum([v...,-1].*raw_game), 0, atol=1e-6)
            push!(verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game) > 0
            println("not a polytope bound")
        end
    end
    verts

    BellScenario.dimension(verts)
    BellScenario.dimension(vertices)

    println(raw_game)
end