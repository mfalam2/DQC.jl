include("sqc_learn.jl")
include("dqc_learn.jl")
include("models.jl")

function complexity(mps; start=0)
    infidelity = 1.0
    n_layers = start
    while infidelity > 1e-3
        n_layers += 1
        infidelities, circ = sqc_svd(mps, n_layers, "complexity.jld"; quiet=true)
        infidelity = infidelities[end]
    end
    return n_layers
end

n_layers = 1
complexities = []
for n in 2:20
    n_layers = complexity(tfim_gs(n); start=n_layers-1) 
    push!(complexities, n_layers)
end

save("complexity.jld", "complexities", complexities)