using Test
using QBase
using BellScenario

include("../src/MultiAccessChannels.jl")


@testset "Multiaccess Network Games" begin
    
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

    @testset "error-syndrom game" begin
        
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

    @testset "broadcast entanglement-assisted receivers" begin
        
        ρ = bell_kets()[1]
        cnot = Unitary([1. 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0])
        ket0 = Ket([1,0])
        ketp = Ket([1,1]/sqrt(2))
        ketm = Ket([1,-1]/sqrt(2))
        ket1 = Ket([0,1])
        swap = Unitary([1. 0 0 0;0 0 1 0;0 1 0 0;0 0 0 1])

        joint_basis = Vector{Vector{ComplexF64}}(undef, 0)
        for b1 in bell_basis, b2 in bell_basis
            push!(joint_basis, kron(b1, b2)[:])
        end

        joint_pvm = PVM(joint_basis)

        ψ = kron(swap, Matrix{Float64}(I,(4,4)) )*kron(ket0,ρ,ket1)
        probs = measure(joint_pvm, State(ψ*ψ'))

        println(probs)



    end

    @testset "" begin
        bell_state = bell_kets()[1]

        bell_basis = PVM(bell_kets())

        measure(bell_state, bell_basis)

        bell_kets()[2]*bell_kets()[2]'bell_state

        ψ1 = [1,0]
        ψ2 = [sqrt(3)/2, 1/2]

        i1 = 1
        i2 = 3
        meas_proj = kron(bell_kets()[i1]*bell_kets()[i1]', bell_kets()[i2]*bell_kets()[i2]') 
        
        result = meas_proj * kron(ψ1, bell_state, ψ2)

        CNOT1 = [1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0]
        CNOT2 = [1 0 0 0;0 0 0 1;0 0 1 0;0 1 0 0]
        H = [1 1;1 -1] / sqrt(2)

        U1 = kron(H, σI.M) * CNOT1

        U2 = kron(σI.M, H) * CNOT2

        U3 = kron(σI.M, CNOT2*CNOT1, σI.M)

        
        meas1 = partial_trace(partial_trace(result * result', [2, 2, 2, 2], 4), [2,2,2], 3)
        meas2 = partial_trace(partial_trace(result * result', [2, 2, 2, 2], 1), [2,2,2], 1)

        result1 = partial_trace(CNOT * meas1 * CNOT, [2, 2], 1)
        result2 = partial_trace(CNOT * meas2 * CNOT, [2, 2], 1)

        partial_trace([1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0]*bell_states()[1]*[1 0 0 0;0 1 0 0;0 0 0 1;0 0 1 0], [2,2], 1)

        partial_trace(
            partial_trace(U3 * kron(U1, U2) * kron(σI.M, bell_state * bell_state', σI.M) * kron(U1, U2)' * U3', [2,2,2,2], 3),
            [2,2,2],2
        )

        partial_trace(
            partial_trace(kron(U2, U1) * kron(ψ1*ψ1', bell_state * bell_state', σI.M) * kron(U2, U1)', [2,2,2,2], 3),
            [2,2,2],2
        )


    end
end
