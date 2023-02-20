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

    @testset "(3,3) -> (2,2) -> 3" begin
        (X, Y, Z, dA, dB) = (3, 3, 3, 2, 2)

        vertices = multi_access_vertices(X,Y,Z,dA,dB)

        @test length(vertices) == 633
        @test length(vertices) == multi_access_num_vertices(X,Y,Z,dA,dB)

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

    @testset "3 -> (2,2) -> (3,3)" begin
        (X, Y, Z, dA, dB) = (3, 3, 3, 2, 2)

        vertices = broadcast_vertices(X,Y,Z,dA,dB)
        @test length(vertices) == 441
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

end
