using Test
using BellScenario

include("../src/classical_network_vertices.jl")


@testset "broadcast network linear programming bounds" begin

    function facet_dimension(normalized_vertices, normalized_bell_game)

        facet_verts = Array{Vector{Int}}([])
        for v in normalized_vertices
            if isapprox(sum([v...,-1].*normalized_bell_game), 0, atol=1e-6)
                push!(facet_verts, convert.(Int, v))
            elseif sum([v...,-1].*normalized_bell_game) > 0
                println("not a polytope bound")
            end
        end

        facet_dim = BellScenario.dimension(facet_verts)

        return facet_dim
    end
    
    @testset "4-22-44" begin
        vertices_4_22_44 = broadcast_vertices(4, 4, 4, 2, 2)
        vertices_4_22_44_unnormalized = broadcast_vertices(4, 4, 4, 2, 2, normalize=false)
        
        @test length(vertices_4_22_44) == 7744

        polytope_dim_4_22_44 = BellScenario.dimension(vertices_4_22_44)
        @test polytope_dim_4_22_44 == 60

        @testset "conditional perm game" begin
            cv_test = [
                1 0 0 0;0 0 0 0;0 0 0 0;0 0 0 0;
                0 0 0 0;0 1 0 0;0 0 0 0;0 0 0 0;
                0 0 0 0;0 0 0 0;0 0 1 0;0 0 0 0;
                0 0 0 0;0 0 0 0;0 0 0 0;0 0 0 1;
            ]

            perm_max_violation = max(map(v -> sum(perm_test[:] .* v), vertices_4_22_44_unnormalized)...)
            @test perm_max_violation ==4

            raw_game_cv_4_22_44 = optimize_linear_witness(vertices_4_22_44, cv_test[1:end-1,:][:])

            println(raw_game_perm_4_22_44*2)
            bg_cv_4_22_44 = convert(BellGame, round.(Int, 2*raw_game_cv_4_22_44), BlackBox(16,4), rep="normalized")

            println(bg_cv_4_22_44)
            println(bg_cv_4_22_44.β)

            facet_dim = facet_dimension(vertices_4_22_44, round.(Int, 2*raw_game_cv_4_22_44))
            @test polytope_dim_4_22_44 == facet_dim + 1

            bg_match = [
                3 0 0 0;1 1 0 1; 1 0 2 1; 1 1 2 0;
                1 1 0 1; 0 3 0 0; 0 1 2 1; 1 1 2 0;
                1 0 2 1; 0 1 2 1; 0 0 3 0; 1 1 2 0;
                1 1 2 0; 1 1 2 0; 1 1 2 0; 2 2 2 0
            ]
            @test bg_match == bg_cv_4_22_44
            @test bg_cv_4_22_44.β == 7
        end

        @testset "entangeld receiver game" begin
            test_game = [
                1 0.  0.  0. ;
                0.  0.  0.  0. ;
                0.  0.  0.  0. ;
                0.  1 0.  0. ;
                0.  0.  0.  0. ;
                0.  1 0.  0. ;
                1 0.  0.  0. ;
                0.  0.  0.  0. ;
                0.  0.  0.  0. ;
                0.  0.  1 0. ;
                0.  0.  0.  1;
                0.  0.  0.  0. ;
                0.  0.  0.  1;
                0.  0.  0.  0. ;
                0.  0.  0.  0. ;
                0.  0.  1 0. ;
            ]

            max_violation = max(map(v -> sum(test_game[:] .* v), vertices_4_22_44_unnormalized)...)
            @test max_violation == 4
        end

        @testsett "pr-box assisted receivers" begin
            pr_mat = hcat(
                kron([1;0],[0.5;0;0;0.5],[1;0]),
                kron([1;0],[0.5;0;0;0.5],[0;1]),
                kron([0;1],[0.5;0;0;0.5],[1;0]),
                kron([0;1],[0;.5;.5;0],[0;1])
            )

            raw_game_pr_4_22_44 = optimize_linear_witness(vertices_4_22_44, pr_mat[1:end-1,:][:])

            println(raw_game_pr_4_22_44)
            bg_pr_4_22_44 = convert(BellGame, round.(Int, raw_game_pr_4_22_44), BlackBox(16,4), rep="normalized")
            println(bg_pr_4_22_44)
            println(bg_pr_4_22_44.β)

            bg_pr_match = [
                3 0 0 0; 0 3 0 0; 0 0 0 3; 1 1 0 2;
                1 1 0 2; 1 1 0 2; 2 1 0 2; 1 3 0 2;
                1 0 2 0; 0 1 2 0; 0 0 0 3; 1 1 0 2;
                2 1 0 2; 1 2 0 2; 1 1 1 2; 2 2 1 2;
            ]
            @test bg_pr_4_22_44 == bg_pr_match
            @test bg_pr_4_22_44.β == 8

            facet_dim = facet_dimension(vertices_4_22_44, round.(Int, raw_game_pr_4_22_44))
            @test polytope_dim_4_22_44 == facet_dim + 1


        end

        @testset "chsh-like game is not violated" begin
            
            chsh_test = [
                1 0 0 0;
                0 1 0 0;
                -1 0 0 0;
                0 -1 0 0;
                -1 0 0 0;
                0 -1 0 0;
                1 0 0 0;
                0 1 0 0;
                0 0 1 0;
                0 0 0 -1;
                0 0 -1 0;
                0 0 0 1;
                0 0 -1 0;
                0 0 0 1;
                0 0 1 0;
                0 0 0 -1;
            ]
            chsh_max_violation = max(map(v -> sum(chsh_test[:] .* v), vertices_4_22_44_unnormalized)...)


        end
    end

    @testset "" begin
        vertices_9_22_44 = broadcast_vertices(9,4,4,2,2)
        vertices_9_22_44_unnormalized = broadcast_vertices(9,4,4,2,2, normalize=false)

        @test length(vertices_9_22_44) == 9388096

        bin_vals = [
            [0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],
            [0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],
            [1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],
            [1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1],
        ]

        magic_squares_game = zeros(16,9)
        for x in [1,2,3], y in [1,2,3]
            col_id = 3*(x-1) + y 
            for i in 1:16
                
                a_bits = bin_vals[i][1:2]
                b_bits = bin_vals[i][3:4]

                a3 = (sum(a_bits) % 2 == 0) ? 0 : 1
                b3 = (sum(b_bits) % 2 == 0) ? 1 : 0

                push!(a_bits, a3)
                push!(b_bits, b3)

                if a_bits[y] == b_bits[x]
                    magic_squares_game[i, col_id] = 1
                end
            end
        end
        ms_test = [
            1.0 1.0 1.0 1.0 1.0 1.0 0.0 0.0 0.0;
            1.0 1.0 1.0 0.0 0.0 0.0 1.0 1.0 1.0;
            0.0 0.0 0.0 1.0 1.0 1.0 1.0 1.0 1.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 
            1.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1.0;
            1.0 0.0 0.0 0.0 1.0 1.0 1.0 0.0 0.0;
            0.0 1.0 1.0 1.0 0.0 0.0 1.0 0.0 0.0;
            0.0 1.0 1.0 0.0 1.0 1.0 0.0 1.0 1.0;
            0.0 1.0 0.0 0.0 1.0 0.0 1.0 0.0 1.0;
            0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0;
            1.0 0.0 1.0 0.0 1.0 0.0 0.0 1.0 0.0;
            1.0 0.0 1.0 1.0 0.0 1.0 1.0 0.0 1.0;
            0.0 0.0 1.0 0.0 0.0 1.0 1.0 1.0 0.0;
            0.0 0.0 1.0 1.0 1.0 0.0 0.0 0.0 1.0;
            1.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0;
            1.0 1.0 0.0 1.0 1.0 0.0 1.0 1.0 0.0;
        ]
        println(magic_squares_game)

        # no advantageg
        ms_max_violation = max(map(v -> sum(magic_squares_game[:] .* v), vertices_9_22_44_unnormalized)...)
        @test ms_max_violation == 9


    end

end