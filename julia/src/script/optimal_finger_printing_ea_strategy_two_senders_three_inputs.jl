using Test
using QBase
import QBase: measure

"""
This script verifies the optimal entanglement assisted finger printing strategy
for two senders with three inputs each.
"""


@testset "Verify Optimal Finger Printing Strategy" begin

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
    @test all()

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
