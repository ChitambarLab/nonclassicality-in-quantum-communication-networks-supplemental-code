using Test
using BellScenario

include("../src/MultiAccessChannels.jl")

@testset "simple polytopes" begin
    @testset "(2,2) -> (2,2) -> 2 only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (2, 2, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 8
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,4),rep="normalized"), facets)

        @test bell_games == [
            [0 0 0 0;1 0 0 0],[0 0 0 0;0 1 0 0],[0 0 0 0;0 0 1 0],[0 0 0 0;0 0 0 1],
            [0 0 0 1;0 0 0 0],[0 0 1 0;0 0 0 0],[0 1 0 0;0 0 0 0],[1 0 0 0;0 0 0 0]
        ]
    end

    @testset "2 -> (2,2) -> (2,2) only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (2, 2, 2, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 8
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,2),rep="normalized"), facets)

        @test bell_games == [
            [0 0; 1 0; 1 0; 1 0],
            [1 0; 0 0; 1 0; 1 0],
            [1 0; 1 0; 0 0; 1 0],
            [0 0; 0 1; 0 1; 0 1],
            [0 1; 0 0; 0 1; 0 1],
            [0 1; 0 1; 0 0; 0 1],
            [0 1; 0 1; 0 1; 0 0],
            [1 0; 1 0; 1 0; 0 0],
        ]
    end

    @testset "(2,2) -> (2,2) -> (2,2) -> (2,2) interference only has non-negativity facets" begin
        (X1, X2, Z1, Z2, dA1, dA2, dB1, dB2) = (2, 2, 2, 2, 2, 2, 2, 2)

        vertices = interference_vertices(X1,X2,Z1,Z2,dA1,dA2,dB1,dB2)

        @test length(vertices) == 256

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 16
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,4),rep="normalized"), facets)
        println(bell_games)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0], [1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0],
            [1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0], [0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0], [0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0],
            [0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0], [0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0], [0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1],
            [0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1], [0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0], [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0],
            [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0], [1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0]
        ]
    end

    @testset "(2,2) -> (2,2) -> 3 only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (2, 2, 3, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 12
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(3,4),rep="normalized"), facets)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0],[1 0 0 0; 0 0 0 0; 1 0 0 0],[0 0 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 0 0 0; 0 1 0 0],[0 0 0 0; 0 0 1 0; 0 0 1 0],[0 0 1 0; 0 0 0 0; 0 0 1 0],
            [0 0 0 0; 0 0 0 1; 0 0 0 1],[0 0 0 1; 0 0 0 0; 0 0 0 1],[0 0 0 1; 0 0 0 1; 0 0 0 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0],[0 1 0 0; 0 1 0 0; 0 0 0 0],[1 0 0 0; 1 0 0 0; 0 0 0 0]
        ]
    end

    @testset "3 -> (2,2) -> (2,2) only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (3, 2, 2, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 64 

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 12
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,3),rep="normalized"), facets)

        @test bell_games == [
            [0 0 0; 1 0 0; 1 0 0; 1 0 0],
            [1 0 0; 0 0 0; 1 0 0; 1 0 0],
            [1 0 0; 1 0 0; 0 0 0; 1 0 0],
            [0 0 0; 0 1 0; 0 1 0; 0 1 0],
            [0 1 0; 0 0 0; 0 1 0; 0 1 0],
            [0 1 0; 0 1 0; 0 0 0; 0 1 0],
            [0 0 0; 0 0 1; 0 0 1; 0 0 1],
            [0 0 1; 0 0 0; 0 0 1; 0 0 1],
            [0 0 1; 0 0 1; 0 0 0; 0 0 1],
            [0 0 1; 0 0 1; 0 0 1; 0 0 0],
            [0 1 0; 0 1 0; 0 1 0; 0 0 0],
            [1 0 0; 1 0 0; 1 0 0; 0 0 0],
        ]
    end

    @testset "(2,2) -> (2,2) -> 4 only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (2, 2, 4, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 16
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,4),rep="normalized"), facets)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0],[1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0],
            [1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0],[0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0],[0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0],
            [0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0],[0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0],[0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1],
            [0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1],[0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0],[0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0],
            [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0],[1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0],
        ]
    end

    @testset "4 -> (2,2) -> (2,2) only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (4, 2, 2, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 256

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 16
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,4),rep="normalized"), facets)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0],[1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0],
            [1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0],[0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0],[0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0],
            [0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0],[0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0],[0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1],
            [0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1],[0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0],[0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0],
            [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0],[1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0],
        ]
    end

    @testset "(2,2) -> (2,2) -> 5 only has non-negativity facets" begin
        (X, Y, Z, dA, dB) = (2, 2, 5, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 20
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(5,4),rep="normalized"), facets)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0], [1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0],
            [1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0], [1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0],
            [0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0], [0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0], [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0],
            [0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 1 0], [0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0], [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0],
            [0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 1], [0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1], [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0], [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0],
            [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0], [1 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0]
        ]
    end

    @testset "(3,2) -> (2,2) -> 2" begin
        (X, Y, Z, dA, dB) = (3, 2, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 40
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 36
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,6),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0;
            1  0  0  0  0  0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  0  1  1  0;
            1  1  1  0  0  0;
        ]
        @test bg2.β == 4
    end

    @testset "(2,3) -> (2,2) -> 2" begin
        (X, Y, Z, dA, dB) = (2, 3, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 40
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 36
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,6),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0;
            1  0  0  0  0  0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  1  0  1  0;
            1  1  0  1  0  0;
        ]
        @test bg2.β == 4
    end

    @testset "(4,2) -> (2,2) -> 2" begin
        (X, Y, Z, dA, dB) = (4, 2, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 88
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 136
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,8),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 3

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0  0  0;
            1  0  0  0  0  0  0  0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  0  1  0  0  1  0;
            1  1  1  0  0  0  0  0;
        ]
        @test bg2.β == 4

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0  0  0  1  1  0  1  1;
            1  1  1  0  0  1  0  0;
        ]
        @test bg3.β == 6
    end

    @testset "(5,2) -> (2,2) -> 2" begin
        (X, Y, Z, dA, dB) = (5, 2, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 184
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 380
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,10),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 3

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0  0  0  0  0;
            1  0  0  0  0  0  0  0  0  0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  0  1  0  0  0  0  1  0;
            1  1  1  0  0  0  0  0  0  0;
        ]
        @test bg2.β == 4

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0  0  0  1  0  0  1  0  1  1;
            1  1  1  0  0  0  0  1  0  0;
        ]
        @test bg3.β == 6
    end

    @testset "(3,3) -> (2,2) -> 2" begin
        (X, Y, Z, dA, dB) = (3, 3, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        @test multi_access_num_vertices(X,Y,Z,dA,dB) == length(vertices)
        @test length(vertices) == 104

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 1230
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,9),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 20

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0  0  0  0;
            1  0  0  0  0  0  0  0  0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  0  0  1  0  1  0  0;
            1  1  0  1  0  0  0  0  0;
        ]
        @test bg2.β == 4

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0  0  1  0  1  0  0  0  0;
            1  1  0  1  0  0  0  0  0;
        ]
        @test bg3.β == 4

        bg4 = bell_games[findfirst(bg -> bg in classes_dict[4], bell_games)]
        @test bg4.β == 5
        @test bg4 == [
            0  0  0  0  0  1  1  0  0;
            1  1  0  1  0  0  0  1  0;
        ]

        bg5 = bell_games[findfirst(bg -> bg in classes_dict[5], bell_games)]
        @test bg5 == [
            0  0  0  0  1  0  0  0  1;
            1  1  0  1  0  1  0  0  0;
        ]
        @test bg5.β == 5

        bg6 = bell_games[findfirst(bg -> bg in classes_dict[6], bell_games)]
        @test bg6.β == 7
        @test bg6 == [
            0  0  1  0  1  0  1  0  0;
            2  1  0  1  0  1  0  1  0;
        ]

        bg7 = bell_games[findfirst(bg -> bg in classes_dict[7], bell_games)]
        @test bg7.β == 7
        @test bg7 == [
            0  0  1  0  1  0  1  0  0;
            2  2  0  1  0  0  0  1  0;
        ]

        bg8 = bell_games[findfirst(bg -> bg in classes_dict[8], bell_games)]
        @test bg8.β == 7
        @test bg8 == [
            0  0  1  0  1  0  1  0  0;
            2  1  0  2  0  1  0  0  0;
        ]

        bg9 = bell_games[findfirst(bg -> bg in classes_dict[9], bell_games)]
        @test bg9.β == 6
        @test bg9 == [
            0  0  0  0  1  1  1  0  1;
            1  1  0  1  0  0  0  1  0;
        ]

        bg10 = bell_games[findfirst(bg -> bg in classes_dict[10], bell_games)]
        @test bg10.β == 8
        @test bg10 ==  [
            0  0  0  0  1  1  2  0  0;
            2  2  0  1  0  0  0  1  0;
        ]

        bg11 = bell_games[findfirst(bg -> bg in classes_dict[11], bell_games)]
        @test bg11.β == 8
        @test bg11 == [
            0  0  0  0  2  0  1  0  1;
            2  2  0  1  0  1  0  0  0;
        ]

        bg12 = bell_games[findfirst(bg -> bg in classes_dict[12], bell_games)]
        @test bg12 == [
            0  0  1  0  2  0  0  0  1
            2  1  0  2  0  1  0  0  0
        ]
        @test bg12.β == 8

        bg13 = bell_games[findfirst(bg -> bg in classes_dict[13], bell_games)]
        @test bg13 == [
            0  0  1  0  2  0  0  0  1;
            2  1  0  2  0  0  0  1  0;
        ]
        @test bg13.β == 8

        bg14 = bell_games[findfirst(bg -> bg in classes_dict[14], bell_games)]
        @test bg14 == [
            0  0  1  0  1  0  2  0  0;
            2  1  0  0  0  1  0  2  0;
        ]
        @test bg14.β == 8

        bg15 = bell_games[findfirst(bg -> bg in classes_dict[15], bell_games)]
        @test bg15 == [
            0  0  1  0  2  0  1  0  1;
            3  2  0  2  0  1  0  0  0;
        ]
        @test bg15.β == 10

        bg16 = bell_games[findfirst(bg -> bg in classes_dict[16], bell_games)]
        @test  bg16 == [
            0  0  1  0  2  0  1  0  1;
            3  2  0  2  0  0  0  1  0;
        ]
        @test bg16.β == 10

        bg17 = bell_games[findfirst(bg -> bg in classes_dict[17], bell_games)]
        @test bg17.β == 11
        @test bg17 == [
            0  0  2  0  1  0  2  0  0;
            3  1  0  1  0  2  0  2  0;
        ]

        bg18 = bell_games[findfirst(bg -> bg in classes_dict[18], bell_games)]
        @test bg18 == [
            0  0  2  0  2  0  2  0  0;
            3  1  0  1  0  3  0  3  1;
        ]
        @test bg18.β == 14

        bg19 = bell_games[findfirst(bg -> bg in classes_dict[19], bell_games)]
        @test bg19 == [
            0  0  2  0  3  0  2  0  1;
            5  3  0  3  0  1  0  1  0;
        ]
        @test bg19.β == 16

        bg20 = bell_games[findfirst(bg -> bg in classes_dict[20], bell_games)]
        @test bg20 == [
            0  0  2  1  2  0  5  0  1;
            4  2  0  0  0  1  0  4  0;
        ]
        @test bg20.β == 17
    end


    @testset "(3,3) -> (2,3) -> 2" begin
        (X, Y, Z, dA, dB) = (3, 3, 2, 2, 3)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 176
        @test length(vertices[1]) == 9

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 186
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,9),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 6

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0 0 0 0;
            1  0  0  0  0  0 0 0 0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  0  0  1  0  1  0  0;
            1  1  0  1  0  0  0  0  0;
        ]
        @test bg2.β == 4

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0 0 1 0 1 0 1 0 0;
            1 1 0 1 0 1 0 1 1;
        ]
        @test bg3.β == 7

        bg4 = bell_games[findfirst(bg -> bg in classes_dict[4], bell_games)]
        @test bg4 == [
            0 0 0 0 1 0 1 0 1;
            1 1 0 0 0 1 0 0 0;
        ]
        @test bg4.β == 5

        bg5 = bell_games[findfirst(bg -> bg in classes_dict[5], bell_games)]
        @test bg5 == [
            0 0 1 0 1 0 1 0 0;
            1 0 0 0 0 1 0 1 0;
        ]
        @test bg5.β == 5

        bg6 = bell_games[findfirst(bg -> bg in classes_dict[6], bell_games)]
        @test bg6 == [
            0 0 0 0 1 1 1 0 1;
            1 1 1 1 0 0 0 1 0;
        ]
        @test bg6.β == 7
    end


    @testset "(3,3) -> (3,2) -> 2" begin
        (X, Y, Z, dA, dB) = (3, 3, 2, 3, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 176
        @test length(vertices[1]) == 9

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 186
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,9),rep="normalized"), facets)

        classes_dict = facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 6

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0 0 0 0;
            1  0  0  0  0  0 0 0 0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  1  0  1  0  0  0  0;
            1  1  0  1  0  0  0  0  0;
        ]
        @test bg2.β == 4

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0 0 1 0 1 0 1 0 0;
            1 1 0 1 0 1 0 1 1;
        ]
        @test bg3.β == 7

        bg4 = bell_games[findfirst(bg -> bg in classes_dict[4], bell_games)]
        @test bg4 == [
            0 0 1 0 1 0 0 0 1;
            1 0 0 1 0 0 0 1 0;
        ]
        @test bg4.β == 5

        bg5 = bell_games[findfirst(bg -> bg in classes_dict[5], bell_games)]
        @test bg5 == [
            0 0 1 0 1 0 1 0 0;
            1 0 0 0 0 1 0 1 0;
        ]
        @test bg5.β == 5

        bg6 = bell_games[findfirst(bg -> bg in classes_dict[6], bell_games)]
        @test bg6 == [
            0 0 1 0 1 0 0 1 1;
            1 1 0 1 0 1 1 0 0;
        ]
        @test bg6.β == 7
    end

    @testset "(3,3) -> (3,2) -> 2 + (3,3) -> (2,3) -> 2" begin

        vertices_a = multi_access_vertices(3, 3, 2, 3, 2)
        vertices_b = multi_access_vertices(3, 3, 2, 2, 3)

        vertices = unique(cat(vertices_a, vertices_b, dims=1))
        @test length(vertices) == 248
        @test length(vertices[1]) == 9

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 330
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(2,9),rep="normalized"), facets)

        classes_dict = facet_classes(3, 3, 2, bell_games)

        @test length(keys(classes_dict)) == 8

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [
            0  0  0  0  0  0 0 0 0;
            1  0  0  0  0  0 0 0 0;
        ]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0  0  0  0  1  0  1  0  0;
            1  1  0  1  0  1  0  0  1;
        ]
        @test bg2.β == 6

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0 0 1 0 1 0 1 0 0;
            1 1 0 1 0 1 0 1 1;
        ]
        @test bg3.β == 7

        bg4 = bell_games[findfirst(bg -> bg in classes_dict[4], bell_games)]
        @test bg4 == [
            0 0 1 0 1 0 1 0 0;
            1 1 0 1 0 0 0 0 0;
        ]
        @test bg4.β == 5

        bg5 = bell_games[findfirst(bg -> bg in classes_dict[5], bell_games)]
        @test bg5 == [
            0 0 1 0 1 0 1 0 0;
            1 0 0 0 0 1 0 1 0;
        ]
        @test bg5.β == 5

        bg6 = bell_games[findfirst(bg -> bg in classes_dict[6], bell_games)]
        @test bg6 == [
            0 0 2 0 1 0 2 0 0;
            2 1 0 1 0 1 0 1 0;
        ]
        @test bg6.β == 9

        bg7 = bell_games[findfirst(bg -> bg in classes_dict[7], bell_games)]
        @test bg7 == [
            0 0 1 1 0 0 2 0 2;
            2 1 0 0 1 2 0 1 0;
        ]
        @test bg7.β == 11

        bg8 = bell_games[findfirst(bg -> bg in classes_dict[8], bell_games)]
        @test bg8 == [
            0 1 2 0 0 0 1 0 2;
            2 0 0 1 1 1 0 2 0;
        ]
        @test bg8.β == 11

    end

    @testset "(3,3) -> (2,2) -> 3" begin
        (X, Y, Z, dA, dB) = (3, 3, 3, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 633
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        # facet computation is too expensive
    end

    @testset "(4,4) -> (2,2) -> 2" begin
        (X, Y, Z, dA, dB) = (4, 4, 2, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        verts = Array{Vector{Int}}([])
        for v in vertices
            test = sum(v'*[2,-1,-1,-1,-1,2,-1,-1,-1,-1,2,-1,-1,-1,-1,2])
            if test +12 == 16
                println(v)
            end
            push!(verts, [test + 12])
        end
        max(verts...)

        @test length(vertices) == 633
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)
        
        vertices[1]'*[2,-1,-1,-1,-1,2,-1,-1,-1,-1,2,-1,-1,-1,-1,2] + 12

        # facet computation is too expensive
    end


    @testset "2 -> (2,2) -> (3,3) positivity only" begin
        (X, Y, Z, dA, dB) = (2, 3, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 81

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 18
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(9,2),rep="normalized"), facets)

        @test bell_games == [
            [0 0;1 0;1 0;1 0;1 0;1 0;1 0;1 0;1 0],
            [1 0;0 0;1 0;1 0;1 0;1 0;1 0;1 0;1 0],
            [1 0;1 0;0 0;1 0;1 0;1 0;1 0;1 0;1 0],
            [1 0;1 0;1 0;0 0;1 0;1 0;1 0;1 0;1 0],
            [1 0;1 0;1 0;1 0;0 0;1 0;1 0;1 0;1 0],
            [1 0;1 0;1 0;1 0;1 0;0 0;1 0;1 0;1 0],
            [1 0;1 0;1 0;1 0;1 0;1 0;0 0;1 0;1 0],
            [1 0;1 0;1 0;1 0;1 0;1 0;1 0;0 0;1 0],
            [0 0;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1],
            [0 1;0 0;0 1;0 1;0 1;0 1;0 1;0 1;0 1],
            [0 1;0 1;0 0;0 1;0 1;0 1;0 1;0 1;0 1],
            [0 1;0 1;0 1;0 0;0 1;0 1;0 1;0 1;0 1],
            [0 1;0 1;0 1;0 1;0 0;0 1;0 1;0 1;0 1],
            [0 1;0 1;0 1;0 1;0 1;0 0;0 1;0 1;0 1],
            [0 1;0 1;0 1;0 1;0 1;0 1;0 0;0 1;0 1],
            [0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 0;0 1],
            [0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 1;0 0],
            [1 0;1 0;1 0;1 0;1 0;1 0;1 0;1 0;0 0],
        ]
        
    end

    @testset "(3,3) -> (2,2) -> (2,2) -> (3,3) interference" begin
        (X1, X2, Z1, Z2, dA1, dA2, dB1, dB2) = (3, 3, 3, 3, 2, 2, 2, 2)

        vertices = interference_vertices(X1,X2,Z1,Z2,dA1,dA2,dB1,dB2)

        ma_vertices = multi_access_vertices(3,3,9,2,2)

        @test length(vertices) == 256

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 16
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,4),rep="normalized"), facets)
        println(bell_games)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0], [1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0],
            [1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0], [0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0], [0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0],
            [0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0], [0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0], [0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1],
            [0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1], [0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0], [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0],
            [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0], [1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0]
        ]
    end

    @testset "(3,3) -> (2,2) -> (2,2) -> (3,3) interference2" begin
        (X1, X2, Z1, Z2, dA1, dA2, dB, dC1, dC2) = (3, 3, 3, 3, 2, 2, 2, 2, 2)

        vertices = interference2_vertices(X1,X2,Z1,Z2,dA1,dA2,dB,dC1,dC2)

        ma_vertices = multi_access_vertices(3,3,9,2,2)

        @test length(vertices) == 256

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 16
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(4,4),rep="normalized"), facets)
        println(bell_games)

        @test bell_games == [
            [0 0 0 0; 1 0 0 0; 1 0 0 0; 1 0 0 0], [1 0 0 0; 0 0 0 0; 1 0 0 0; 1 0 0 0],
            [1 0 0 0; 1 0 0 0; 0 0 0 0; 1 0 0 0], [0 0 0 0; 0 1 0 0; 0 1 0 0; 0 1 0 0],
            [0 1 0 0; 0 0 0 0; 0 1 0 0; 0 1 0 0], [0 1 0 0; 0 1 0 0; 0 0 0 0; 0 1 0 0],
            [0 0 0 0; 0 0 1 0; 0 0 1 0; 0 0 1 0], [0 0 1 0; 0 0 0 0; 0 0 1 0; 0 0 1 0],
            [0 0 1 0; 0 0 1 0; 0 0 0 0; 0 0 1 0], [0 0 0 0; 0 0 0 1; 0 0 0 1; 0 0 0 1],
            [0 0 0 1; 0 0 0 0; 0 0 0 1; 0 0 0 1], [0 0 0 1; 0 0 0 1; 0 0 0 0; 0 0 0 1],
            [0 0 0 1; 0 0 0 1; 0 0 0 1; 0 0 0 0], [0 0 1 0; 0 0 1 0; 0 0 1 0; 0 0 0 0],
            [0 1 0 0; 0 1 0 0; 0 1 0 0; 0 0 0 0], [1 0 0 0; 1 0 0 0; 1 0 0 0; 0 0 0 0]
        ]
    end

    # lifted version of signaling polytope facets
    @testset "3 -> (2,2) -> (3,2)" begin
        (X, Y, Z, dA, dB) = (3, 3, 2, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 168
        @test length(vertices[1]) == 15
        @test length(vertices[1]) == BellScenario.dimension(vertices)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 24
        @test length(facet_dict["equalities"]) == 0


        bell_games = map(f -> convert(BellGame, f, BlackBox(6,3),rep="normalized"), facets)


        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        # nonnegativity game
        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [0 0 1;0 0 1;0 1 0;0 1 0;1 0 0;1 0 0]
        @test bg2.β == 2
    end

    @testset "3 -> (2,2) -> (2,3)" begin
        (X, Y, Z, dA, dB) = (3, 2, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 168
        @test length(vertices[1]) == 15
        @test length(vertices[1]) == BellScenario.dimension(vertices)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 24
        @test length(facet_dict["equalities"]) == 0

        bell_games = map(f -> convert(BellGame, f, BlackBox(6,3),rep="normalized"), facets)

        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        # nonnegativity game
        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [0 0 1;0 1 0;1 0 0;0 0 1;0 1 0;1 0 0]
        @test bg2.β == 2
    end

    @testset "3 -> (2,2) -> (4,2)" begin
        (X, Y, Z, dA, dB) = (3, 4, 2, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 320
        @test length(vertices[1]) == 21
        @test length(vertices[1]) == BellScenario.dimension(vertices)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 84
        @test length(facet_dict["equalities"]) == 0

        bell_games = map(f -> convert(BellGame, f, BlackBox(8,3),rep="normalized"), facets)
        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)
        @test length(keys(classes_dict)) == 3

        # nonnegativity game
        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [0 0 1;0 0 1;0 1 0;0 1 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg2.β == 2

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [0 0 2;0 0 2;0 2 0;0 2 0;2 0 0;2 0 0;1 1 1;1 1 1]
        @test bg3.β == 4
    end

    @testset "3 -> (2,3) -> (3,3)" begin
        (X, Y, Z, dA, dB) = (3, 3, 3, 2, 3)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 567
        @test length(vertices[1]) == 24
        @test length(vertices[1]) == BellScenario.dimension(vertices)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 33
        @test length(facet_dict["equalities"]) == 0


        bell_games = map(f -> convert(BellGame, f, BlackBox(9,3),rep="normalized"), facets)


        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        # nonnegativity game
        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [0 0 1;0 0 1;0 0 1;0 1 0;0 1 0;0 1 0;1 0 0;1 0 0;1 0 0]
        @test bg2.β == 2
    end

    @testset "3 -> (3,2) -> (3,3)" begin
        (X, Y, Z, dA, dB) = (3, 3, 3, 3, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 567
        @test length(vertices[1]) == 24
        @test length(vertices[1]) == BellScenario.dimension(vertices)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 33
        @test length(facet_dict["equalities"]) == 0


        bell_games = map(f -> convert(BellGame, f, BlackBox(9,3),rep="normalized"), facets)


        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        # nonnegativity game
        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [0 0 1;0 1 0;1 0 0;0 0 1;0 1 0;1 0 0;0 0 1;0 1 0;1 0 0]
        @test bg2.β == 2
    end

    @testset "3 -> (3/2,2/3) -> (3,3)" begin

        vertices_a = broadcast_vertices(3,3,3,3,2)
        vertices_b = broadcast_vertices(3,3,3,2,3)

        vertices = cat(vertices_a,vertices_b, dims=1)

        @test length(vertices) == 1134
        @test length(vertices[1]) == 24
        @test length(vertices[1]) == BellScenario.dimension(vertices)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 63
        @test length(facet_dict["equalities"]) == 0


        bell_games = map(f -> convert(BellGame, f, BlackBox(9,3),rep="normalized"), facets)


        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 2

        # nonnegativity game
        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [0 0 1;0 0 0;0 0 0;0 0 0;0 1 0;0 0 0;0 0 0;0 0 0;1 0 0]
        @test bg2.β == 2
    end

    @testset "3 -> (2,2) -> (3,3)" begin
        (X, Y, Z, dA, dB) = (3, 3, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 441
        @test length(vertices[1]) == 24
        BellScenario.dimension(vertices)
        # @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)

        facets = facet_dict["facets"]

        @test length(facets) == 417
        @test facet_dict["equalities"] == []

        bell_games = map(f -> convert(BellGame, f, BlackBox(9,3),rep="normalized"), facets)

        bell_games[1]

        classes_dict = bipartite_broadcast_facet_classes(X, Y, Z, bell_games)

        @test length(keys(classes_dict)) == 7

        bg1 = bell_games[findfirst(bg -> bg in classes_dict[1], bell_games)]
        @test bg1 == [0 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0;1 0 0]
        @test bg1.β == 1

        bg2 = bell_games[findfirst(bg -> bg in classes_dict[2], bell_games)]
        @test bg2 == [
            0 0 1;
            0 0 0;
            1 0 0;
            0 0 0;
            0 1 0;
            1 0 0;
            1 0 0;
            1 0 0;
            1 0 0;
        ]
        @test bg2.β == 2

        bg3 = bell_games[findfirst(bg -> bg in classes_dict[3], bell_games)]
        @test bg3 == [
            0 0 1;
            0 0 1;
            0 0 1;
            0 1 0;
            0 1 0;
            0 1 0;
            1 0 0;
            1 0 0;
            1 0 0;
        ]
        @test bg3.β == 2

        bg4 = bell_games[findfirst(bg -> bg in classes_dict[4], bell_games)]
        @test bg4 == [
            0 0 1;
            0 1 0;
            1 0 0;
            0 0 1;
            0 1 0;
            1 0 0;
            0 0 1;
            0 1 0;
            1 0 0;
        ]
        @test bg4.β == 2

        bg5 = bell_games[findfirst(bg -> bg in classes_dict[5], bell_games)]
        @test bg5 == [
            0 0 1;
            0 0 0;
            0 0 0;
            0 0 0;
            1 0 0;
            0 1 0;
            0 0 0;
            0 1 0;
            1 0 0
        ]
        @test bg5.β == 2

        bg6 = bell_games[findfirst(bg -> bg in classes_dict[6], bell_games)]
        @test bg6 == [
            0 0 2;
            0 0 1;
            0 1 1;
            0 2 0;
            0 1 0;
            0 1 1;
            1 0 0;
            2 0 0;
            1 1 1;
        ]
        @test bg6.β == 4

        bg7 = bell_games[findfirst(bg -> bg in classes_dict[7], bell_games)]
        @test bg7 == [
            0 0 2;
            0 2 0;
            1 0 0;
            0 0 1;
            0 1 0;
            2 0 0;
            0 1 1;
            0 1 1;
            1 1 1;
        ]
        @test bg7.β == 4

    end

    @testset "4 -> (2,2) -> (3,3)" begin
        (X, Y, Z, dA, dB) = (4, 3, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 2025
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

    end

    @testset "3 -> (2,2) -> (4,3)" begin
        (X, Y, Z, dA, dB) = (3, 4, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 840

    end


    @testset "3 -> (2,2) -> (4,3)" begin
        (X, Y, Z, dA, dB) = (3, 4, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 840
        @test length(vertices[1]) == 33
        BellScenario.dimension(vertices)
        # @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

        facet_dict = LocalPolytope.facets(vertices)
    end

    @testset "4 -> (2,2) -> (4,4)" begin
        (X, Y, Z, dA, dB) = (4, 4, 4, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB, normalize=false)
        @test length(vertices) == 7744
        @test length(vertices[1]) == 64
        @test BellScenario.dimension(vertices) == 60

        # embedded CHSH inequaliy
        bg_chsh = BellScenario.BellGame([
            1  0  0  0 ;
            0  1  0  0 ;
            -1 0  0  0 ;
            0  -1 0  0 ;
            -1 0  0  0 ;
            0  -1 0  0 ;
            1  0  0  0 ;
            0  1  0  0 ;
            0  0  1  0 ;
            0  0  0  -1;
            0  0  -1 0 ;
            0  0  0  1 ;
            0  0  -1 0 ;
            0  0  0  1 ;
            0  0  1  0 ;
            0  0  0  -1;
        ], 2)


        # inequality is proper half-space
        @test BellScenario.dimension(filter(v -> bg_chsh.β == bg_chsh[:]'*v, vertices)) == 59

        # inequality does not bound polytope
        @test length(filter(v -> bg_chsh.β < bg_chsh[:]'*v, vertices)) == 24

        bg_correlated = BellScenario.BellGame([
            1 0 0 0;
            0 1 0 0;
            0 0 0 0;
            0 0 0 0;
            0 1 0 0;
            1 0 0 0;
            0 0 0 0;
            0 0 0 0;
            0 0 0 0;
            0 0 0 0;
            0 0 1 0;
            0 0 0 1;
            0 0 0 0;
            0 0 0 0;
            0 0 0 1;
            0 0 1 0;
        ], 2)

        # inequality is proper half-space and tight facet
        @test BellScenario.dimension(filter(v -> bg_correlated.β == bg_correlated[:]'*v, vertices)) == 59
        @test length(filter(v -> bg_correlated.β < bg_correlated[:]'*v, vertices)) == 0


        bg_witness = BellScenario.BellGame([
            3 0 0 0;
            0 3 0 0;
            0 0 0 3;
            1 1 0 2;
            1 1 0 2;
            1 1 0 2;
            2 1 0 2;
            1 3 0 2;
            1 0 2 0;
            0 1 2 0;
            0 0 0 3;
            1 1 0 2;
            2 1 0 2;
            1 2 0 2;
            1 1 1 2;
            2 2 1 2;
        ], 8)

        # inequality is proper half-space and tight facet
        @test BellScenario.dimension(filter(v -> bg_witness.β == bg_witness[:]'*v, vertices)) == 59
        @test length(filter(v -> bg_witness.β < bg_witness[:]'*v, vertices)) == 0

    end


    @testset "bacon toner scenario 22->22" begin

        bt_vertices = bacon_and_toner_vertices(2,2,2,2)

        facet_dict = LocalPolytope.facets(bt_vertices)

        @test facet_dict["equalities"] == []

        facets = facet_dict["facets"] 

        bell_games = map(f -> convert(BellGame, f, BipartiteNonSignaling(2,2,2,2),rep="normalized"), facets)

        bt_facet_classes = bipartite_interference_facet_classes(2,2,2,2, bell_games)

        bell_games[24].β


        bt_facet_classes[6][4]

        bell_game_match1 = BellGame([1 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1], 2)

        @test bell_game_match1 in bell_games


        bell_game_match2 = BellGame([1 0 0 0;0 0 1 0;1 0 0 1;0 0 0 1], 2)
        @test bell_game_match2 in bell_games





    end

    @testset  "bacon toner scenario 33->22" begin
        bt_vertices = bacon_and_toner_vertices(3,3,2,2, true)
        bt_vertices_unnorm = bacon_and_toner_vertices(3,3,2,2, false)
        length(bt_vertices_unnorm[1])


        polytope_dim = LocalPolytope.dimension(bt_vertices)


        bell_game_match1 = BellGame([0 0 1 0 1 1 1 1 1;0 1 0 1 0 0 0 0 0;0 1 0 1 0 0 0 0 0;0 0 1 0 1 1 1 1 1], 7)

        facet_vertices = Vector{Vector{Int64}}(undef, 0)
        for v in bt_vertices_unnorm
            if sum(bell_game_match1[:] .* v) == bell_game_match1.β
                push!(facet_vertices, v)
            end
        end

        facet_dim = LocalPolytope.dimension(facet_vertices)

        @test facet_dim + 1 == polytope_dim

        bell_game_match2 = BellGame([1 2 0 2 1 2 0 2 1;0 0 2 0 0 0 2 0 0;0 0 2 0 0 0 2 0 0;1 2 0 2 1 2 0 2 1], 13)
    
        facet_vertices = Vector{Vector{Int64}}(undef, 0)
        for v in bt_vertices_unnorm
            if sum(bell_game_match2[:] .* v) == bell_game_match2.β
                push!(facet_vertices, v)
            end
        end


        facet_dim = LocalPolytope.dimension(facet_vertices)

        @test facet_dim + 1 == polytope_dim


    end
end

