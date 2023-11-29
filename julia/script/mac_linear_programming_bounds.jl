using Test
using BellScenario

include("../src/classical_network_vertices.jl")

"""
This script investigates the classical bounds of various simulation games in multiaccess networks.
In each case, we enumerae the vertices for the network. Then, 
"""

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

    @test bell_game.β == 10

    @test bell_game == [
        1  0  0  0  0  1  0  0  0  0  1  0  0  0  0  1;
        0  1  2  2  1  0  1  1  0  0  0  1  0  0  0  0;
    ]
end

@testset "(3,3)->(2,2)->3" begin
    
    vertices = multi_access_vertices(3,3,3,2,2)

    @test length(vertices) == 633

    diff_test = [
        1 0 0 0 1 0 0 0 1;
        0 1 0 1 0 1 0 1 0;
        0 0 1 0 0 0 1 0 0;
    ]

    raw_game = optimize_linear_witness(vertices, diff_test[1:2,:][:])
    bell_game = convert(BellGame, round.(Int, 2*raw_game), BlackBox(3,9), rep="normalized")

    @testt bell_game.β == 8
    @test bell_game == [
        2  0  0  0  1  0  0  0  1;
        0  0  1  2  0  1  0  1  0;
        1  1  1  1  0  2  1  0  0;
    ]
end

@testset "(3,3) -> (2,2) -> 3" begin
    vertices33_22_3 = multi_access_vertices(3,3,3,2,2, normalize=false)

    
    compare_game = [
        1 0 0 0 1 0 0 0 1;
        0 1 1 0 0 1 0 0 0;
        0 0 0 1 0 0 1 1 0;
    ]

    @test 7 == max(map(v -> sum(v[:].*compare_game[:]), vertices33_22_3)...)
end

@testset "(3,3) -> (2,2) -> 2" begin
    vertices33_22_2 = multi_access_vertices(3,3,2,2,2, normalize=false)

    @test 104 == length(vertices33_22_2)
    
    compare_game = [
        1 0 0 1 1 0 1 1 1;
        0 1 1 0 0 1 0 0 0;
    ]

    @test 8 == max(map(v -> sum(v[:].*compare_game[:]), vertices33_22_2)...)
end

@testset "(4,4) -> (2,2) -> 2" begin
    vertices44_22_2 = multi_access_vertices(4,4,2,2,2, normalize=false)

    @test 520 == length(vertices44_22_2)
    
    compare_game = [
        1 0 0 0 1 1 0 0 1 1 1 0 1 1 1 1;
        0 1 1 1 0 0 1 1 0 0 0 1 0 0 0 0;
    ]

    @test 14 == max(map(v -> sum(v[:].*compare_game[:]), vertices44_22_2)...)
end

@testset "(4,4) -> (2,2) -> 3" begin
    vertices44_22_3 = multi_access_vertices(4,4,3,2,2, normalize=false)

    @test 3321 == length(vertices44_22_3)
    
    compare_game = [
        1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1;
        0 1 1 1 0 0 1 1 0 0 0 1 0 0 0 0;
        0 0 0 0 1 0 0 0 1 1 0 0 1 1 1 0;
    ]

    @test 12 == max(map(v -> sum(v[:].*compare_game[:]), vertices44_22_3)...)
end

@testset "(4,4) -> (2,2) -> 4" begin
    vertices44_22_4 = multi_access_vertices(4,4,4,2,2, normalize=false)

    @test length(vertices44_22_4) == 11344

    qkd_game = [
        1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1;
        0 1 0 0 1 0 0 0 0 0 0 1 0 0 1 0;
        0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0;
        0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0;
    ]

    @test 8 == max(map(v -> sum(v[:].*qkd_game[:]), vertices44_22_4)...)
end

 @testset "44->22->4 MAC Game" begin
        
    vertices_44_22_4 = multi_access_vertices(4,4,4,2,2, normalize=false)

    vertices_44_34_4 = multi_access_vertices(4,4,4,3,4, normalize=false)
    vertices_44_43_4 = multi_access_vertices(4,4,4,4,3, normalize=false)

    vertices_44_3443_4 = unique(cat(
        vertices_44_34_4,
        vertices_44_43_4,
        dims=1
    ))

    game1 = [
        1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1;
        0 1 0 0 1 0 0 0 0 0 0 1 0 0 1 0;
        0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0;
        0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0;
    ]
    game2 = [
        1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1;
        0 1 0 0 1 0 1 0 0 1 0 1 0 0 1 0;
        0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0;
        0 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0;
    ]
    game1_scores = map(v -> (v, sum(game1[:].*v[:])), vertices_44_3443_4)
    game2_scores = map(v -> (v, sum(game2[:].*v[:])), vertices_44_3443_4)

    @test 8 == max(map(tuple -> tuple[2], game1_scores)...)
    @test 10 == max(map(tuple -> tuple[2], game2_scores)...)
end

@testset "(6,3)->(2,3)->2 qmac  " begin
    
    vertices = multi_access_vertices(6,3,2,2,3)

    diff_test = [
        1 0 0 0 1 1 0 1 0 1 0 1 0 0 1 1 1 0;
        0 1 1 1 0 0 1 0 1 0 1 0 1 1 0 0 0 1;
    ]

    raw_game = optimize_linear_witness(vertices, diff_test[1,:][:])
    bell_game = convert(BellGame, round.(Int, 2*raw_game), BlackBox(2,18), rep="normalized")

    @test raw_game ≈ [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
end

@testset "(8,3)->(2,3)->2 qmac  " begin
    
    vertices = multi_access_vertices(8,3,2,2,3)
    gen_vertices = multi_access_vertices(8,3,2,2,3, normalize=false)

    diff_test = [
        1 1 1  1 1 0  1 0 1  1 0 0  0 1 1  0 1 0  0 0 1  0 0 0;
        0 0 0  0 0 1  0 1 0  0 1 1  1 0 0  1 0 1  1 1 0  1 1 1;
    ]

    max_score = 0
    for v in gen_vertices
        score = sum(diff_test[:].*v[:])
        if score > max_score
            max_score = score
        end
    end
    max_score

    raw_game = optimize_linear_witness(vertices, diff_test[1,:][:])
    bell_game = convert(BellGame, round.(Int, 2*raw_game), BlackBox(2,24), rep="normalized")
    bell_game.β


    println(raw_game)

    @test raw_game ≈ [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
end
