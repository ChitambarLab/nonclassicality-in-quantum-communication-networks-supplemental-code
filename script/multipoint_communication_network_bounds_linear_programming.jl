using Test
using BellScenario

include("../src/MultiAccessChannels.jl")


@testset "(3,3)->(2,2)->(2,2) -> (2,2) " begin
    
    vertices = butterfly_vertices(3,3,2,2,2,2,2,2,2,2,2)

    diff_test = [
        1 0 0 0 1 0 0 0 1;
        0 1 1 1 0 1 1 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]

    length(vertices[1])
    raw_game = optimize_linear_witness(vertices, diff_test[1:3,:][:])

    @test raw_game ≈ [0,0,0,0,0,0,0,0,0,0] 
end

@testset "(4,4)->(2,2)->(2,2) -> (2,2) " begin
    
    butterfly_vertices_4422 = butterfly_vertices(4,4,2,2,2,2,2,2,2,2,2)
    butterfly_vertices_4422_unnormalized = butterfly_vertices(4,4,2,2,2,2,2,2,2,2,2, normalize=false)

    @test length(butterfly_vertices_4422) == 506176

    rac_test = [
        1 1 0 0 1 1 0 0 0 0 1 0 0 0 0 0;
        0 0 1 1 0 0 0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 0 1 0 0 0 1 0 1 0;
        0 0 0 0 0 0 1 1 0 1 0 0 0 1 0 1;
    ]
    rac_max_violation = max(map(v -> sum(rac_test[:] .* v), butterfly_vertices_4422_unnormalized)...)
    @test rac_max_violation == 13


    raw_game_rac = optimize_linear_witness(butterfly_vertices_4422, rac_test[1:end-1,:][:])

    println(raw_game_rac)
    bell_game_rac = convert(BellGame, round.(Int, 6*raw_game_rac), BlackBox(4,16), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_4422
        if isapprox(sum([v...,-1].*raw_game_rac), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_rac) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1


    bell_game_rac_match = [
        3  2  0  0  4  4   7   3  0  2  4  0  0  0  0  6;
        2  0  6  8  0  2   0   0  0  1  0  8  0  2  2  0;
        0  0  0  6  2  0   8   7  6  0  2  2  4  0  4  0;
        6  1  2  8  5  1  10  10  4  2  4  4  4  2  2  0;
    ]
    @test bell_game_rac_match == bell_game_rac
    @test bell_game_rac.β == 71


end

@testset "(3,3)->(3,3) Butterfly" begin
    
    butterfly_vertices_3333 = butterfly_vertices(3,3,3,3,2,2,2,2,2,2,2)
    butterfly_vertices_3333_unnormalized = butterfly_vertices(3,3,3,3,2,2,2,2,2,2,2, normalize=false)
    polytope_dim = BellScenario.dimension(butterfly_vertices_3333)

    @test length(butterfly_vertices_3333) == 690813

    # stirling encodings gives 690813 vertices

    mult1_test = [ # multiplication game [1,2,3] no zero 
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
    mult0_test = [ # multiplication with zero 
        1 1 1 1 0 0 1 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]
    swap_test = [ # swap game
        1 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0;
        0 1 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 1 0;
        0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 1;
    ]
    adder_test = [ # adder game
        1 0 0 0 0 0 0 0 0;
        0 1 0 1 0 0 0 0 0;
        0 0 1 0 1 0 1 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]
    compare_test = [
        1 0 0 0 1 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 1 1 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 1 1 0;
        0 0 0 0 0 0 0 0 0;
    ]
    perm_test = [ # on receiver permutes output based on other receiver
        1 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 0 0;
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 1 0;
    ]
    diff_test = [
        1 0 0 0 1 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 1 0 1 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 1 0 0;
    ]
    cv_test = I(9)


    mult1_max_violation = max(map(v -> sum(mult1_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test mult1_max_violation == 6
    mult0_max_violation = max(map(v -> sum(mult0_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test mult0_max_violation == 7
    swap_max_violation = max(map(v -> sum(swap_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test swap_max_violation ==4
    perm_max_violation = max(map(v -> sum(perm_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test perm_max_violation == 6
    adder_max_violation = max(map(v -> sum(adder_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test adder_max_violation == 6
    compare_max_violation = max(map(v -> sum(compare_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test compare_max_violation == 7
    diff_max_violation = max(map(v -> sum(diff_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test diff_max_violation == 6
    cv_max_violation = max(map(v -> sum(cv_test[:] .* v), butterfly_vertices_3333_unnormalized)...)
    @test cv_max_violation == 7


    raw_game_mult0 = optimize_linear_witness(butterfly_vertices_3333, mult0_test[1:end-1,:][:])
    raw_game_mult1 = optimize_linear_witness(butterfly_vertices_3333, mult1_test[1:end-1,:][:])
    raw_game_swap = optimize_linear_witness(butterfly_vertices_3333, swap_test[1:end-1,:][:])
    raw_game_adder = optimize_linear_witness(butterfly_vertices_3333, adder_test[1:end-1,:][:])
    raw_game_compare = optimize_linear_witness(butterfly_vertices_3333, compare_test[1:end-1,:][:])
    raw_game_perm = optimize_linear_witness(butterfly_vertices_3333, perm_test[1:end-1,:][:])
    raw_game_diff = optimize_linear_witness(butterfly_vertices_3333, diff_test[1:end-1,:][:])
    raw_game_cv = optimize_linear_witness(butterfly_vertices_3333, cv_test[1:end-1,:][:])




    println(raw_game_mult0)
    bell_game_mult0 = convert(BellGame, round.(Int, 3*raw_game_mult0), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_mult0), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_mult0) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1


    bell_game_mult0_match = [
        1  3  1  1  0  0  1  0  0;
        0  0  0  1  2  0  1  0  1;
        0  1  0  0  0  1  0  2  0;
        0  2  1  0  1  1  0  1  1;
        0  1  0  1  2  0  0  0  2;
        0  2  1  1  1  1  0  1  1;
        0  2  1  0  1  1  0  1  1;
        0  0  0  1  1  0  1  0  1;
        1  2  1  1  1  1  1  1  1;
    ]
    @test bell_game_mult0_match == bell_game_mult0
    @test bell_game_mult0.β == 11

    println(raw_game_mult1)
    bell_game_mult1 = convert(BellGame, round.(Int, 4*raw_game_mult1), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_mult1), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_mult1) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1
    bell_game_mult1_match = [
        2  0  1  1  0  0  0  0  0;
        0  1  1  3  0  0  0  0  0;
        0  0  1  1  0  0  2  0  0;
        1  0  1  0  1  1  0  1  0;
        0  1  1  1  0  2  0  1  0;
        0  0  1  1  0  2  0  2  0;
        1  1  1  1  1  1  0  0  1;
        0  1  0  2  1  1  1  0  1;
        1  1  1  2  1  1  1  1  1;
    ]
    @test bell_game_mult1_match == bell_game_mult1
    @test bell_game_mult1.β == 11

    println(raw_game_swap)
    bell_game_swap = convert(BellGame, round.(Int, 5*raw_game_swap), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_swap), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_swap) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1

    bell_game_swap_match = [
        2  0  1  0  0  1  0  1  0;
        0  0  1  2  0  1  0  1  0;
        1  0  1  1  1  1  1  1  1;
        0  2  0  0  0  1  1  0  0;
        0  0  1  0  2  0  1  0  0;
        0  1  1  1  1  1  0  1  1;
        0  0  3  0  0  2  0  1  0;
        0  0  2  0  0  3  0  1  0;
        1  1  2  1  1  2  1  1  1;
    ]
    @test bell_game_swap == bell_game_swap_match
    @test bell_game_swap.β == 12

    println(raw_game_adder)
    bell_game_adder = convert(BellGame, round.(Int, 3*raw_game_adder), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_adder), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_adder) > 0
            println("not a polytope bound")
        end
    end

    # game_verts = Array{Vector{Int}}([])
    # for v in butterfly_vertices_3333_unnormalized
    #     if sum(bell_game_adder_match .* v)
    #         push!(game_verts, convert.(Int, v))
    #     elseif sum([v...,-1].*raw_game_adder) > 0
    #         println("not a polytope bound")
    #     end
    # end
    
    

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1

    bell_game_adder_match = [
        1  0  1  0  1  1  1  1  0;
        0  2  0  1  0  0  0  0  2;
        1  1  2  0  1  1  1  1  1;
        0  1  1  0  0  2  1  1  0;
        0  1  0  0  0  0  0  0  2;
        0  1  2  0  0  1  1  1  1;
        1  0  1  0  1  1  1  1  1;
        0  1  0  1  0  0  0  0  2;
        1  1  2  0  1  1  1  1  1;
    ]

    @test bell_game_adder == bell_game_adder_match
    @test bell_game_adder.β == 10 

    println(raw_game_compare)
    bell_game_compare = convert(BellGame, round.(Int, 3*raw_game_compare), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_compare), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_compare) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1

    bell_game_compare_match = [
        3  0  0  0  2  0  1  0  1;
        2  1  0  0  1  0  1  1  1;
        2  1  1  0  0  1  0  1  0;
        1  1  2  0  0  2  0  1  1;
        0  2  2  0  0  2  0  1  1;
        0  3  2  0  0  3  0  0  0;
        1  1  2  0  1  1  1  1  0;
        1  2  2  1  1  1  1  1  0;
        2  2  3  1  1  2  1  1  0;
    ]

    @test bell_game_compare == bell_game_compare_match
    @test bell_game_compare.β == 14 

    println(raw_game_perm)
    bell_game_perm = convert(BellGame, round.(Int, 4*raw_game_perm), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_perm), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_perm) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1

    bell_game_perm_match = [
        2  1  0  1  3  0  1  1  2;
        1  5  2  0  0  2  1  1  0;
        0  3  3  1  3  3  0  2  0;
        1  2  0  1  4  0  1  0  2;
        0  3  1  1  1  3  1  1  0;
        0  3  3  2  1  2  0  1  0;
        1  0  0  1  3  0  1  1  1;
        0  3  1  0  0  2  2  1  1;
        2  3  3  1  3  3  1  2  0;
    ]

    @test bell_game_perm == bell_game_perm_match
    @test bell_game_perm.β == 20 


    println(raw_game_diff)
    bell_game_diff = convert(BellGame, round.(Int, 2*raw_game_diff), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_diff), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_diff) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1

    bell_game_diff_match = [
        3  0  0  0  2  0  0  1  1;
        1  0  0  1  1  0  0  1  1;
        2  0  0  0  1  0  0  0  1;
        2  0  1  0  0  2  0  1  0;
        1  0  1  2  0  1  0  2  0;
        2  1  1  1  1  2  0  1  0;
        1  0  0  0  0  1  0  1  1;
        0  0  1  1  0  2  0  1  1;
        2  1  2  1  1  2  1  1  0;
    ]

    @test bell_game_diff == bell_game_diff_match
    @test bell_game_diff.β == 12


    println(raw_game_cv)
    bell_game_cv = convert(BellGame, round.(Int, 2*raw_game_cv), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in butterfly_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_cv), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_cv) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)

    @test polytope_dim == facet_dim + 1

    bell_game_cv_match = [
        3  0  1  0  3  1  0  0  0;
        1  3  0  0  3  1  0  0  0;
        0  0  3  0  4  1  0  0  1;
        1  0  1  3  2  1  0  0  0;
        3  1  1  1  3  0  0  0  0;
        3  1  2  0  0  2  0  0  1;
        0  0  1  0  3  1  2  1  1;
        2  0  1  0  0  1  2  1  1;
        4  1  2  1  4  1  1  0  0;
    ]

    @test bell_game_cv == bell_game_cv_match
    @test bell_game_cv.β == 18
end

@testset "(3,3) -> (3,3) Interference" begin
    interference_vertices_3333 = interference_vertices(3,3,3,3,2,2,2,2)
    interference_vertices_3333_unnormalized = interference_vertices(3,3,3,3,2,2,2,2, normalize=false)

    polytope_dim = BellScenario.dimension(interference_vertices_3333)

    @test length(interference_vertices_3333) == 17289
    @test polytope_dim == 72

    mult1_test = [ # multiplication game [1,2,3] no zero 
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
    mult0_test = [ # multiplication with zero 
        1 1 1 1 0 0 1 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]
    swap_test = [ # swap game
        1 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0;
        0 1 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 1 0;
        0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 1;
    ]
    adder_test = [ # adder game
        1 0 0 0 0 0 0 0 0;
        0 1 0 1 0 0 0 0 0;
        0 0 1 0 1 0 1 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]
    compare_test = [
        1 0 0 0 1 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 1 1 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 1 1 0;
        0 0 0 0 0 0 0 0 0;
    ]
    perm_test = [ # on receiver permutes output based on other receiver
        1 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 0 0;
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 1 0;
    ]
    diff_test = [
        1 0 0 0 1 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 1 0 1 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 1 0 0;
    ]
    cv_test = I(9)


    mult1_max_violation = max(map(v -> sum(mult1_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test mult1_max_violation == 5
    mult0_max_violation = max(map(v -> sum(mult0_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test mult0_max_violation == 7
    swap_max_violation = max(map(v -> sum(swap_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test swap_max_violation == 4
    perm_max_violation = max(map(v -> sum(perm_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test perm_max_violation == 4
    adder_max_violation = max(map(v -> sum(adder_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test adder_max_violation == 5
    compare_max_violation = max(map(v -> sum(compare_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test compare_max_violation == 6
    diff_max_violation = max(map(v -> sum(diff_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test diff_max_violation == 7
    cv_max_violation = max(map(v -> sum(cv_test[:] .* v), interference_vertices_3333_unnormalized)...)
    @test cv_max_violation == 4

    raw_game_mult0 = optimize_linear_witness(interference_vertices_3333, mult0_test[1:end-1,:][:])
    raw_game_mult1 = optimize_linear_witness(interference_vertices_3333, mult1_test[1:end-1,:][:])
    raw_game_swap = optimize_linear_witness(interference_vertices_3333, swap_test[1:end-1,:][:])
    raw_game_adder = optimize_linear_witness(interference_vertices_3333, adder_test[1:end-1,:][:])
    raw_game_compare = optimize_linear_witness(interference_vertices_3333, compare_test[1:end-1,:][:])
    raw_game_perm = optimize_linear_witness(interference_vertices_3333, perm_test[1:end-1,:][:])
    raw_game_diff = optimize_linear_witness(interference_vertices_3333, diff_test[1:end-1,:][:])
    raw_game_cv = optimize_linear_witness(interference_vertices_3333, cv_test[1:end-1,:][:])

    println(raw_game_mult0)
    bell_game_mult0 = convert(BellGame, round.(Int, 4*raw_game_mult0), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_mult0), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_mult0) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_mult0_match = [
        1  2  1  3  0  1  0  0  1;
        1  0  0  1  3  0  0  1  2;
        1  0  0  1  1  3  0  2  0;
        0  1  1  0  1  2  1  1  2;
        0  1  1  0  1  2  1  1  2;
        0  1  1  1  1  2  1  1  2;
        1  0  0  1  3  1  0  0  2;
        1  1  0  1  1  2  0  1  2;
        1  1  1  2  2  2  1  1  1;
    ]
    @test bell_game_mult0_match == bell_game_mult0
    @test bell_game_mult0.β == 13

    println(raw_game_mult1*7)
    bell_game_mult1 = convert(BellGame, round.(Int, 7*raw_game_mult1), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_mult1), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_mult1) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_mult1_match = [
        3  0  1  0  2  1  1  0  1;
        0  3  0  2  0  0  1  0  1;
        0  0  4  0  2  1  2  0  0;
        2  1  2  0  2  1  0  1  1;
        2  1  2  1  1  2  0  1  1;
        2  1  2  0  0  3  0  2  0;
        2  1  2  0  2  1  0  0  1;
        2  1  2  0  2  1  0  0  1;
        2  2  3  1  1  2  1  1  0;
    ]
    @test bell_game_mult1_match == bell_game_mult1
    @test bell_game_mult1.β == 14

    println(raw_game_swap*7)
    bell_game_swap = convert(BellGame, round.(Int, 7*raw_game_swap), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_swap), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_swap) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_swap_match = [
        3  0  0  0  2  0  1  0  1;
        1  1  2  2  0  0  1  0  1;
        0  1  3  0  1  1  2  0  0;
        0  3  0  2  0  0  0  1  1;
        1  1  2  0  2  0  0  1  1;
        1  0  3  1  0  1  0  2  0;
        0  0  4  1  1  0  1  1  0;
        1  2  2  0  0  2  1  1  0;
        2  2  3  1  1  1  1  1  0;
    ]
    @test bell_game_swap_match == bell_game_swap
    @test bell_game_swap.β == 13

    
    println(raw_game_adder*8)
    bell_game_adder = convert(BellGame, round.(Int, 8*raw_game_adder), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_adder), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_adder) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_adder_match = [
        3  0  1  0  2  1  1  0  1;
        0  3  0  2  0  0  1  0  1;
        0  0  4  0  2  1  2  0  0;
        2  1  2  0  0  3  0  2  0;
        1  0  3  1  1  2  1  1  1;
        1  2  2  1  1  2  0  1  1;
        2  1  2  0  2  1  0  0  1;
        1  1  3  0  2  1  1  0  1;
        2  2  3  1  1  2  1  1  0;
    ]
    @test bell_game_adder_match == bell_game_adder
    @test bell_game_adder.β == 14

    println(raw_game_compare)
    bell_game_compare = convert(BellGame, round.(Int, 5*raw_game_compare), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_compare), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_compare) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_compare_match = [
        2  0  0  0  3  0  0  0  1;
        0  1  1  1  0  3  1  1  0;
        1  1  1  0  0  3  0  1  0;
        0  2  1  0  2  2  1  0  1;
        1  1  1  0  0  3  1  1  0;
        0  3  1  0  1  3  1  0  0;
        1  2  1  1  1  2  0  0  1;
        0  1  2  2  0  1  1  1  0;
        1  2  2  1  2  2  1  0  0;
    ]
    @test bell_game_compare_match == bell_game_compare
    @test bell_game_compare.β == 12

    println(raw_game_perm)
    bell_game_perm = convert(BellGame, round.(Int, 7*raw_game_perm), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_perm), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_perm) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_perm_match = [
        3  0  0  0  1  1  0  1  1;
        0  3  2  1  0  1  0  0  1;
        0  0  4  1  1  1  0  1  0;
        1  1  2  1  2  0  0  1  1;
        1  2  2  1  0  3  0  1  0;
        1  2  2  2  0  0  0  0  1;
        1  2  2  1  1  1  0  1  1;
        0  1  3  0  1  2  1  0  0;
        2  2  3  1  1  2  0  1  0;
    ]
    @test bell_game_perm_match == bell_game_perm
    @test bell_game_perm.β == 13

    println(raw_game_diff*3)
    bell_game_diff = convert(BellGame, round.(Int, 3*raw_game_diff), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_diff), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_diff) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_diff_match = [
        2  0  0  0  2  0  1  0  1;
        0  0  2  1  1  2  1  0  1;
        1  0  2  0  0  2  1  0  1;
        0  0  2  1  1  2  1  0  1;
        0  0  2  2  0  2  0  1  0;
        0  0  2  2  1  2  0  0  1;
        1  0  2  0  0  2  1  0  1;
        0  0  2  2  1  2  0  0  1;
        1  1  3  1  1  2  1  0  0;
    ]
    @test bell_game_diff_match == bell_game_diff
    @test bell_game_diff.β == 11

    println(raw_game_cv*7)
    bell_game_cv = convert(BellGame, round.(Int, 7*raw_game_cv), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_cv), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_cv) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_cv_match = [
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
    @test bell_game_cv_match == bell_game_cv
    @test bell_game_cv.β == 13


end


@testsett "(3,3)->(3,3) interference 2" begin
    interference2_vertices_3333 = interference2_vertices(3,3,3,3,2,2,2,2,2)
    interference2_vertices_3333_unnormalized = interference2_vertices(3,3,3,3,2,2,2,2,2, normalize=false)

    polytope_dim = BellScenario.dimension(interference2_vertices_3333)

    @test length(interference2_vertices_3333) == 3681
    @test polytope_dim == 72

    mult1_test = [ # multiplication game [1,2,3] no zero 
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
    mult0_test = [ # multiplication with zero 
        1 1 1 1 0 0 1 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]
    swap_test = [ # swap game
        1 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 1 0 0;
        0 1 0 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 0 1 0;
        0 0 1 0 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 1;
    ]
    adder_test = [ # adder game
        1 0 0 0 0 0 0 0 0;
        0 1 0 1 0 0 0 0 0;
        0 0 1 0 1 0 1 0 0;
        0 0 0 0 0 1 0 1 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
    ]
    compare_test = [
        1 0 0 0 1 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 1 1 0 0 1 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 1 0 0 1 1 0;
        0 0 0 0 0 0 0 0 0;
    ]
    perm_test = [ # on receiver permutes output based on other receiver
        1 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 1 0 0 0;
        0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 0 0 0 1;
        0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 1 0;
    ]
    diff_test = [
        1 0 0 0 1 0 0 0 1;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 1 0 1 0 1 0 1 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 1 0 0;
    ]
    cv_test = I(9)


    mult1_max_violation = max(map(v -> sum(mult1_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test mult1_max_violation == 4
    mult0_max_violation = max(map(v -> sum(mult0_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test mult0_max_violation == 7
    swap_max_violation = max(map(v -> sum(swap_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test swap_max_violation == 2
    perm_max_violation = max(map(v -> sum(perm_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test perm_max_violation == 2
    adder_max_violation = max(map(v -> sum(adder_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test adder_max_violation == 5
    compare_max_violation = max(map(v -> sum(compare_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test compare_max_violation == 6
    diff_max_violation = max(map(v -> sum(diff_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test diff_max_violation == 7
    cv_max_violation = max(map(v -> sum(cv_test[:] .* v), interference2_vertices_3333_unnormalized)...)
    @test cv_max_violation == 2

    raw_game_mult0 = optimize_linear_witness(interference2_vertices_3333, mult0_test[1:end-1,:][:])
    raw_game_mult1 = optimize_linear_witness(interference2_vertices_3333, mult1_test[1:end-1,:][:])
    raw_game_swap = optimize_linear_witness(interference2_vertices_3333, swap_test[1:end-1,:][:])
    raw_game_adder = optimize_linear_witness(interference2_vertices_3333, adder_test[1:end-1,:][:])
    raw_game_compare = optimize_linear_witness(interference2_vertices_3333, compare_test[1:end-1,:][:])
    raw_game_perm = optimize_linear_witness(interference2_vertices_3333, perm_test[1:end-1,:][:])
    raw_game_diff = optimize_linear_witness(interference2_vertices_3333, diff_test[1:end-1,:][:])
    raw_game_cv = optimize_linear_witness(interference2_vertices_3333, cv_test[1:end-1,:][:])

    println(raw_game_mult0)
    bell_game_mult0 = convert(BellGame, round.(Int, 4*raw_game_mult0), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_mult0), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_mult0) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_mult0_match = [
        1  2  2  1  0  0  0  0  1;
        0  0  1  0  3  0  1  2  1;
        0  0  0  0  0  3  1  3  0;
        0  1  2  0  1  2  1  2  1;
        0  1  2  0  1  2  1  2  1;
        0  1  2  0  1  2  1  2  1;
        0  1  2  0  1  2  1  2  1;
        0  1  2  0  1  2  1  2  1;
        1  1  2  0  2  2  1  2  0;
    ]
    @test bell_game_mult0_match == bell_game_mult0
    @test bell_game_mult0.β == 12

    println(raw_game_mult1*7)
    bell_game_mult1 = convert(BellGame, round.(Int, 7*raw_game_mult1), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_mult1), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_mult1) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_mult1_match = [
        4  0  1  1  0  2  0  0  1;
        1  3  0  3  0  0  0  0  1;
        0  0  3  0  2  1  1  0  0;
        2  1  1  2  2  1  0  0  1;
        2  1  2  2  1  2  0  0  1;
        2  1  1  2  0  3  0  1  0;
        2  1  2  2  1  2  0  0  1;
        2  1  2  2  1  2  0  0  1;
        3  2  2  2  1  2  0  0  0;
    ]
    @test bell_game_mult1_match == bell_game_mult1
    @test bell_game_mult1.β == 13

    println(raw_game_swap)
    bell_game_swap = convert(BellGame, round.(Int, 7*raw_game_swap), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_swap), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_swap) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_swap_match = [
        3  0  0  0  0  1  0  0  1;
        0  2  0  2  1  0  0  0  1;
        0  1  0  0  1  2  1  0  0;
        0  3  0  0  0  1  0  0  1;
        1  1  0  1  2  0  0  0  1;
        1  1  0  1  0  2  0  1  0;
        0  1  2  0  1  0  0  1  0;
        1  1  0  1  0  2  0  1  0;
        2  2  1  1  1  1  0  0  0;
    ]
    @test bell_game_swap_match == bell_game_swap
    @test bell_game_swap.β == 9

    
    println(raw_game_adder)
    bell_game_adder = convert(BellGame, round.(Int, 8*raw_game_adder), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_adder), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_adder) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_adder_match = [
        4  0  1  1  0  2  0  0  1;
        1  3  0  3  0  0  0  0  1;
        0  0  3  0  2  1  1  0  0;
        2  1  1  2  0  3  0  1  0;
        2  1  2  2  1  2  0  0  1;
        2  1  2  2  1  2  0  0  1;
        2  1  2  2  1  2  0  0  1;
        2  1  2  2  1  2  0  0  1;
        3  2  2  2  1  2  0  0  0;
    ]
    @test bell_game_adder_match == bell_game_adder
    @test bell_game_adder.β == 13

    println(raw_game_compare)
    bell_game_compare = convert(BellGame, round.(Int, 6*raw_game_compare), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_compare), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_compare) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_compare_match = [
        3  0  0  0  2  0  0  0  1;
        1  0  3  2  1  2  0  0  1;
        1  0  3  2  1  2  0  0  1;
        1  0  3  2  1  2  0  0  1;
        1  0  3  2  1  2  0  0  1;
        0  2  3  1  0  3  0  0  0;
        1  0  3  2  1  2  0  0  1;
        0  0  3  3  0  2  0  1  0;
        2  1  3  2  1  2  0  0  0;
    ]
    @test bell_game_compare_match == bell_game_compare
    @test bell_game_compare.β == 12

    println(raw_game_perm)
    bell_game_perm = convert(BellGame, round.(Int, 7*raw_game_perm), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_perm), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_perm) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_perm_match = [
        3  0  0  0  0  1  0  0  1;
        0  3  0  0  0  1  0  0  1;
        0  1  2  0  1  0  0  1  0;
        1  1  0  1  2  0  0  0  1;
        1  1  0  1  0  2  0  1  0;
        0  2  0  2  1  0  0  0  1;
        1  1  1  1  1  1  0  0  1;
        0  1  0  0  1  2  1  0  0;
        2  2  1  1  1  1  0  0  0;
    ]
    @test bell_game_perm_match == bell_game_perm
    @test bell_game_perm.β == 9

    println(raw_game_diff)
    bell_game_diff = convert(BellGame, round.(Int, 3*raw_game_diff), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_diff), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_diff) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_diff_match = [
        2  0  0  0  2  0  0  0  1;
        0  1  3  1  1  2  0  0  1;
        0  1  3  1  1  2  0  0  1;
        0  1  3  1  1  2  0  0  1;
        0  1  3  2  0  2  0  1  0;
        0  1  3  1  1  2  0  0  1;
        0  1  3  1  1  2  0  0  1;
        0  1  3  1  1  2  0  0  1;
        1  2  3  1  1  2  0  0  0;
    ]
    @test bell_game_diff_match == bell_game_diff
    @test bell_game_diff.β == 11

    println(raw_game_cv)
    bell_game_cv = convert(BellGame, round.(Int, 7*raw_game_cv), BlackBox(9,9), rep="normalized")

    facet_verts = Array{Vector{Int}}([])
    for v in interference2_vertices_3333
        if isapprox(sum([v...,-1].*raw_game_cv), 0, atol=1e-6)
            push!(facet_verts, convert.(Int, v))
        elseif sum([v...,-1].*raw_game_cv) > 0
            println("not a polytope bound")
        end
    end

    facet_dim = BellScenario.dimension(facet_verts)
    @test polytope_dim == facet_dim + 1

    bell_game_cv_match = [
        3  0  0  0  0  1  0  0  1;
        0  3  0  0  0  1  0  0  1;
        0  1  2  0  1  0  0  1  0;
        0  2  0  2  1  0  0  0  1;
        1  1  0  1  2  0  0  0  1;
        1  1  0  1  0  2  0  1  0;
        0  1  0  0  1  2  1  0  0;
        1  1  0  1  0  2  0  1  0;
        2  2  1  1  1  1  0  0  0;
    ]
    @test bell_game_cv_match == bell_game_cv
    @test bell_game_cv.β == 9
end

@testset "(2,2)->(2,2)->(2,2) -> (3,3) " begin
    
    vertices = butterfly_vertices(2,2,2,2,2,2,2,2,2,2,2)

    @test length(vertices) == 256

    length(vertices[1])
    raw_game = optimize_linear_witness(vertices, diff_test[1:3,:][:])

    @test raw_game ≈ [0,0,0,0,0,0,0,0,0,0] 
end