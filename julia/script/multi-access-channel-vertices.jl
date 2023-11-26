using Test

include("../src/ClassicalNetworkVertices.jl")

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
