using Test
using BellScenario
using QBase
import QBase: measure

include("../src/classical_network_vertices.jl")


@testset "Multiaccess Network Games" begin

    """
    This testset demonstrates nonclassical quantum strategies for multiaccess network simulation games.
    """

    @testset "EATx Finger Printing Strategy with Classical signaling" begin

        ψ = Ket([1,0,0,1]/sqrt(2))
    
        ϕ0 = PVM([[1,0],[0,1]])
        ϕ1 = PVM([[1/2,sqrt(3)/2],[sqrt(3)/2,-1/2]])
        ϕ2 = PVM([[1/2,-sqrt(3)/2],[sqrt(3)/2,1/2]])
    
        joint_pvm(pvm1, pvm2) =  PVM(map(
            (a,b) -> kron(pvm1[a], pvm2[b]), [1,1,2,2], [1,2,1,2]
        ))
    
        local_pvms = [ϕ0, ϕ1, ϕ2]
    
        joint_pvms = collect(flatten(
            map(pvm1 -> map(pvm2 -> joint_pvm(pvm1, pvm2), local_pvms), local_pvms)
        ))
    
        measure(pvm :: QBase.PVM, ket :: QBase.Ket) = map(ϕ -> abs(ϕ'ket)^2, pvm)
    
        @test all(pvm -> measure(pvm, ψ) ≈ [1,0,0,1]/2, joint_pvms[[1,5,9]])
        @test all(pvm -> measure(pvm, ψ) ≈ [1,3,3,1]/8, joint_pvms[[2,3,4,6,7,8]])
    
        # decoding applied by classical MAC
        postmap = [1 0 0 1;0 1 1 0]
    
        P_mac = postmap * hcat(map(
            pvm -> measure(pvm, ψ), joint_pvms
        )...)
    
        @test P_mac ≈ [
            1 0.25 0.25 0.25 1 0.25 0.25 0.25 1;
            0 0.75 0.75 0.75 0 0.75 0.75 0.75 0;
        ]
    end
    
    
    @testset "trit distance game" begin
        
        ψ = bell_kets()[1]

        U_setA = [σI, σz, σy]
        U_setB = U_setA

        bell_basis = PVM(bell_kets())
        probs = zeros((4,9))

        state_ensemble = []
        for x1 in [1,2,3], x2 in [1,2,3]
            ψ_x1x2 = kron(U_setA[x1], U_setB[x2]) * ψ
            push!(state_ensemble, ψ_x1x2)
            id = (x1-1)*3 + x2
            probs[:,id] = measure(bell_basis, ψ_x1x2)
        end

        post_process = [1 0 0 0;0 1 1 0;0 0 0 1]
    
        @test all(post_process*round.(probs) == [
            1 0 0 0 1 0 0 0 1;
            0 1 0 1 0 1 0 1 0;
            0 0 1 0 0 0 1 0 0;
        ])
    end

    @testset "two-bit XOR game (error syndrome)" begin
        
        ψ = bell_kets()[1]

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

        @test all(round.(probs) == [
            1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1;
            0 1 0 0 1 0 0 0 0 0 0 1 0 0 1 0;
            0 0 1 0 0 0 0 1 1 0 0 0 0 1 0 0;
            0 0 0 1 0 0 1 0 0 1 0 0 1 0 0 0;
        ])
    end
end
