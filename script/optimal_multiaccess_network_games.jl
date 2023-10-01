using Test
using QBase
using BellScenario

include("../src/MultiAccessChannels.jl")


@testset "Multiaccess Network Games" begin
    
    @testset "trit distance game" begin
        
        ψ = bell_kets()[1]

        prep_settingsA = [(0,0),(0,π),(π,0),(π,π)]
        prep_settingsB = [(0,0),(0,π),(π,0),(π,π)]
        U_setA = map(settings -> qubit_rotation(settings[2], axis="z") * qubit_rotation(settings[1], axis="y"), prep_settingsA)
        U_setB = map(settings -> qubit_rotation(settings[2], axis="z") * qubit_rotation(settings[1], axis="y"), prep_settingsB)

        # U_setA = map(settings -> Unitary(im*qubit_rotation(settings[2], axis="z") * im*qubit_rotation(settings[1], axis="x")), prep_settingsA)
        # U_setB = map(settings -> Unitary(im*qubit_rotation(settings[2], axis="z") * im*qubit_rotation(settings[1], axis="x")), prep_settingsB)

        U_setA = [σI, σz, σx, σy]
        U_setB = U_setA

        bell_basis = PVM(bell_kets())
        probs = zeros((4,16))

        state_ensemble = []
        for x1 in [1,2,3,4], x2 in [1,2,3,4]
            ψ_x1x2 = kron(U_setA[x1], U_setB[x2]) * ψ
            push!(state_ensemble, ψ_x1x2)
            id = (x1-1)*4 + x2
            probs[:,id] = measure(bell_basis, ψ_x1x2)
        end
        round.(probs)

        post_process = [1 0 0 0;0 1 1 0;0 0 0 1]
    
        post_process*probs

    
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

        LocalPolytope.dimension(map(tuple -> tuple[1], game1_scores[findall(tuple -> tuple[2] == 8, game1_scores)]))
        LocalPolytope.dimension(map(tuple -> tuple[1], game2_scores[findall(tuple -> tuple[2] == 8, game1_scores)]))

        

        polytope_dim
    end
end
