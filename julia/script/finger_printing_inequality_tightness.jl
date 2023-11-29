using Test

include("../src/communication_network_nonclassicality.jl")

"""
This script shows that the equality simulation game G^= is not a tight classical bound
in any easily computable scenarios.
"""

@testset "Equality simulation game tightness" begin

@testset "num_senders=2, num_in=3" begin
    vertices = multi_access_vertices(3, 3, 2, 2, 2, normalize=false)
    fp = finger_printing_game(2, 3)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 5

end

@testset "num_senders=2, num_in=4" begin
    vertices = multi_access_vertices(4, 4, 2, 2, 2, normalize=false)
    fp = finger_printing_game(2, 4)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 3
end

@testset "num_senders=2, num_in=4 qutrit" begin
    vertices = multi_access_vertices(4, 4, 2, 3, 3, normalize=false)
    fp = finger_printing_game(2, 4)

    @test all(v -> fp.game[:]'*v ≤ fp.β+1, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β+1, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 9
end

@testset "num_senders=2, num_in=5" begin
    vertices = multi_access_vertices(5, 5, 2, 2, 2, normalize=false)
    fp = finger_printing_game(2, 5)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 4
end

@testset "num_senders=2, num_in=6" begin
    vertices = multi_access_vertices(6, 6, 2, 2, 2, normalize=false)
    fp = finger_printing_game(2, 6)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 5
end

@testset "num_senders=2, num_in=7" begin
    vertices = multi_access_vertices(7, 7, 2, 2, 2, normalize=false)
    fp = finger_printing_game(2, 7)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 6
end

@testset "num_senders=2, num_in=8" begin
    vertices = multi_access_vertices(8, 8, 2, 2, 2, normalize=false)
    fp = finger_printing_game(2, 8)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 7
end

@testset "num_senders=3, num_in=3" begin
    num_senders = 3
    num_in = 3

    vertices = three_sender_multi_access_vertices(
        num_in, num_in, num_in, 2, 2, 2, 2,
        normalize=false
    )
    fp = finger_printing_game(num_senders, num_in)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 2
end

@testset "num_senders=3, num_in=4" begin
    num_senders = 3
    num_in = 4

    vertices = three_sender_multi_access_vertices(
        num_in, num_in, num_in, 2, 2, 2, 2,
        normalize=false
    )
    fp = finger_printing_game(num_senders, num_in)

    @test all(v -> fp.game[:]'*v ≤ fp.β, vertices)

    fp_vertices = filter(v -> fp.game[:]'*v == fp.β, vertices)

    @test LocalPolytope.dimension(fp_vertices) == 3
end


end
