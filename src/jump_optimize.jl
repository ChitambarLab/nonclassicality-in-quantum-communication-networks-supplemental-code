using JuMP
using HiGHS

include("./MultiAccessChannels.jl")


p = [0.2,0.5,0.3,-1]
d1 = [1,0,0,-1]
d2 = [0.5,0.5,0,-1]
d3 = [0.5,0,0.5,-1]

d_set = [d1,d2,d3]

p = [0.3,0.7,0.5,0.5,0.2,0.3,0.4,0.1,-1]
v_set = [
    [-1, -1, -1, -1,  1,  1,  1,  1,-1],
    [-1, -1, -1,  1,  1, -1,  1, -1,-1],
    [-1, -1,  1, -1, -1,  1, -1,  1,-1],
    [-1, -1,  1,  1, -1, -1, -1, -1,-1],
    [-1,  1, -1, -1,  1,  1, -1, -1,-1],
    [-1,  1, -1,  1,  1, -1, -1,  1,-1],
    [-1,  1,  1, -1, -1,  1,  1, -1,-1],
    [-1,  1,  1,  1, -1, -1,  1,  1,-1],
    [ 1, -1, -1, -1, -1, -1,  1,  1,-1],
    [ 1, -1, -1,  1, -1,  1,  1, -1,-1],
    [ 1, -1,  1, -1,  1, -1, -1,  1,-1],
    [ 1, -1,  1,  1,  1,  1, -1, -1,-1],
    [ 1,  1, -1, -1, -1, -1, -1, -1,-1],
    [ 1,  1, -1,  1, -1,  1, -1,  1,-1],
    [ 1,  1,  1, -1,  1, -1,  1, -1,-1],
    [ 1,  1,  1,  1,  1,  1,  1,  1,-1],
]


(X, Y, Z, dA, dB) = (4, 4, 4, 2, 2)

vertices = broadcast_vertices(X,Y,Z,dA,dB)

# p_mat = [
#     1 0 0 0;
#     0 0 0 0;
#     0 0 0 0;
#     0 0 0 0;
#     0 0 0 0;
#     0 1 0 0;
#     1 0 0 0;
#     0 0 0 0;

#     0 0 0 0;
#     0 0 0 0;
#     0 0 1 0;
#     0 0 0 0;
#     0 0 0 0;
#     0 0 0 0;
#     0 0 0 0;
# ]
p_mat = hcat(kron([1;0],[0.5;0;0;0.5],[1;0]), kron([1;0],[0.5;0;0;0.5],[0;1]), kron([0;1],[0.5;0;0;.5],[1;0]), kron([0;1],[0;.5;.5;0],[0;1]))
pq_mat = hcat(kron([1;0],[1/(2*sqrt(2));(1 - 1/sqrt(2))/2;(1 - 1/sqrt(2))/2;1/(2*sqrt(2))],[1;0]), kron([1;0],[1/(2*sqrt(2));(1 - 1/sqrt(2))/2;(1 - 1/sqrt(2))/2;1/(2*sqrt(2))],[0;1]), kron([0;1],[1/(2*sqrt(2));(1 - 1/sqrt(2))/2;(1 - 1/sqrt(2))/2;1/(2*sqrt(2))],[1;0]), kron([0;1],[(1 - 1/sqrt(2))/2;1/(2*sqrt(2));1/(2*sqrt(2));(1 - 1/sqrt(2))/2],[0;1]))


p = p_mat[1:end-1,:][:]
# p = pq_mat[1:end-1,:][:]

num_v = length(vertices)
dim_v = length(vertices[1]) + 1 

model = Model(HiGHS.Optimizer)

@variable(model, s[1:dim_v])


for v in vertices
    va = [v..., -1]
    @constraint(model, sum(s.*va) <= 0)
end

# @constraint(model, c1, sum(s.*d1) <= 0)
# @constraint(model, c2, sum(s.*d2) <= 0)
# @constraint(model, c3, sum(s.*d3) <= 0)




@constraint(model, c, sum(s.*[p..., -1]) <= 1)

@objective(model, Max, sum(s.*[p..., -1]))

optimize!(model)
println(value.(s))

bg = convert(BellGame, round.(Int, 2*value.(s)), BlackBox(16,4), rep="normalized")
bg.β

objective_value(model)

verts = Array{Vector{Int}}([])
for v in vertices
    if sum([v...,-1].*value.(s)) ≈ 0
        push!(verts, convert.(Int, v))
    end
end
verts


BellScenario.dimension(verts)
BellScenario.dimension(vertices)

ρ = State([0.5 0.5im;-0.5im 0.5])

Π = [0.5 0.35im;-0.35im 0.5]

sqrt(Π) * ρ * sqrt(Π)'

sqrt(Π) * ρ * sqrt(Π)'

ρ =State( [1 0 0 1;0 0 0 0;0 0 0 0;1 0 0 1]/2)

partial_trace(kron([1 0;0 1], sqrt(Π)) * ρ * kron([1 0;0 1], sqrt(Π)')  , [2,2], 1)

transpose(sqrt(Π)) *[1 0;0 1]/2 * transpose(sqrt(Π)')

sqrt(Π) *[1 0;0 1]/2 * sqrt(Π)'