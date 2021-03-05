using Test
using BellScenario
using QBase

@testset "quantum violations" begin

@testset "(3,2)->(2,2)->2 violations" begin
    @testset "trine states and computational basis" begin
        (X,Y,Z,dA,dB) = (3,2,2,2,2)

        game = BellGame([
            0  0  0  1  1  0;
            1  1  1  0  0  0;
        ], 4)

        ρA_states = States.trine_qubits
        ρB_states = States.basis_states(2)

        opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

        @test isapprox(opt_dict["violation"], 0.366, atol=1e-3)
    end
end

@testset "(3,3)->(2,2)->2 violations" begin

    @testset "trine states and trine states" begin
        (X,Y,Z,dA,dB) = (3,3,2,2,2)

        ρA_states = States.trine_qubits
        ρB_states = States.trine_qubits

        @testset "bg2" begin
            game = BellGame([
                0  0  0  0  1  0  1  0  0;
                1  1  0  1  0  0  0  0  0;
            ], 4)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.253, atol=1e-3)
        end


        @testset "bg3" begin
            game = BellGame([
                0  0  1  0  1  0  0  0  0;
                1  1  0  1  0  0  0  0  0;
            ], 4)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.253, atol=1e-3)
        end

        @testset "bg4" begin
            game = BellGame([
                0  0  0  0  0  1  1  0  0;
                1  1  0  1  0  0  0  1  0;
            ], 5)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.005, atol=1e-3)
        end

        @testset "bg5" begin
            game = BellGame([
                0  0  0  0  1  0  0  0  1;
                1  1  0  1  0  1  0  0  0;
            ], 5)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.005, atol=1e-3)
        end

        @testset "bg6 (no violation with trine)" begin
            game = BellGame([
                0  0  1  0  1  0  1  0  0;
                2  1  0  1  0  1  0  1  0;
            ], 7)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.177, atol=1e-3)
        end

        @testset "bg7 no violation" begin
            game = BellGame([
                0  0  1  0  1  0  1  0  0;
                2  2  0  1  0  0  0  1  0;
            ], 7)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.073, atol=1e-3)
        end

        @testset "bg8 no violation" begin
            game = BellGame([
                0  0  1  0  1  0  1  0  0;
                2  1  0  2  0  1  0  0  0;
            ], 7)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.073, atol=1e-3)
        end

        @testset "bg9" begin
            game = BellGame([
                0  0  0  0  1  1  1  0  1;
                1  1  0  1  0  0  0  1  0;
            ], 6)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.427, atol=1e-3)
        end

        @testset "bg10" begin
            game = BellGame([
                0  0  0  0  1  1  2  0  0;
                2  2  0  1  0  0  0  1  0;
            ], 8)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.325, atol=1e-3)
        end

        @testset "bg11" begin
            game = BellGame([
                0  0  0  0  2  0  1  0  1;
                2  2  0  1  0  1  0  0  0;
            ], 8)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.373, atol=1e-3)
        end

        @testset "bg12" begin
            game = BellGame([
                0  0  1  0  2  0  0  0  1
                2  1  0  2  0  1  0  0  0
            ], 8)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.325, atol=1e-3)
        end

        @testset "bg13" begin
            game = BellGame([
                0  0  1  0  2  0  0  0  1;
                2  1  0  2  0  0  0  1  0;
            ], 8)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.373, atol=1e-3)
        end

        @testset "bg14 no violation" begin
            game = BellGame([
                0  0  1  0  1  0  2  0  0;
                2  1  0  0  0  1  0  2  0;
            ], 8)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.291, atol=1e-3)
        end

        @testset "bg15" begin
            game = BellGame([
                0  0  1  0  2  0  1  0  1;
                3  2  0  2  0  1  0  0  0;
            ], 10)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.362, atol=1e-3)
        end

        @testset "bg16" begin
            game = BellGame([
                0  0  1  0  2  0  1  0  1;
                3  2  0  2  0  0  0  1  0;
            ], 10)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.362, atol=1e-3)
        end

        @testset "bg17 no violation" begin
            game = BellGame([
                0  0  2  0  1  0  2  0  0;
                3  1  0  1  0  2  0  2  0;
            ], 11)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.235, atol=1e-3)
        end

        @testset "bg18 no violation" begin
            game = BellGame([
                0  0  2  0  2  0  2  0  0;
                3  1  0  1  0  3  0  3  1;
            ], 14)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.231, atol=1e-3)
        end

        @testset "bg19" begin
            game = BellGame([
                0  0  2  0  3  0  2  0  1;
                5  3  0  3  0  1  0  1  0;
            ], 16)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], 0.177, atol=1e-3)
        end

        @testset "bg20 no violation" begin
            game = BellGame([
                0  0  2  1  2  0  5  0  1;
                4  2  0  0  0  1  0  4  0;
            ], 17)

            opt_dict = multi_access_optimize_measurement(X,Y,Z,dA,dB, game, ρA_states, ρB_states)

            @test isapprox(opt_dict["violation"], -0.457, atol=1e-3)
        end
    end
end

end
