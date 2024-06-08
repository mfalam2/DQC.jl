"""
train_sqc.jl
Author: Faisal Alam
Date: 6/7/2024
functions for training static quantum circuits 
"""

using Flux: train!
using ProgressMeter
using JLD
include("gates.jl")
include("environments.jl")

" generates either the parameters or gates for SQC ansatz "
function generate_sqc_ansatz(targ_mps, n_layers, data="gates")
    params = rand(n_layers, length(targ_mps)-1, 15)
    if data == "gates"
        return kak_circuit(params)
    else
        return params
    end
end

" infidelity of static quantum circuit with targ_mps given params "
function sqc_cost(params, targ_mps)
    circ = kak_circuit(params) 
    hilbert = siteinds(targ_mps)
    return 1 - abs(inner(runcircuit(hilbert, circ), targ_mps)) # square root infidelity
end

" infidelity of static quantum circuit with targ_mps given circ "
function sqc_cost(circ, targ_mps)
    hilbert = siteinds(targ_mps)
    return 1 - abs(inner(runcircuit(hilbert, circ), targ_mps))
end

" gradient descent optimization of static quantum circuits "
function sqc_gd(targ_mps, ansatz, filename; num_sweeps=5000, eta=0.05)
    n = length(targ_mps)
    hilbert = siteinds("Qubit", n)
    params = ansatz
    
    cost_list = []
    @showprogress for epoch in 1:num_sweeps
        train!(sqc_cost, params, [(targ_mps,)], Descent(eta))
        c = sqc_cost(params, targ_mps)
        push!(cost_list, c)
        if epoch % 100 == 0
            println(c)
        end    
    end
    save(filename, "cost_list", cost_list, "params", params)
    return cost_list, params
end 

" training SQCs with SVD updates "
function sqc_svd(targ_mps, ansatz, filename; num_sweeps=500, quiet=false)
    right_mps = deepcopy(targ_mps)
    left_mps = siteinds(targ_mps)
    circ = ansatz
    
    cost_list = []
    @showprogress for sweep in 1:num_sweeps
        for i in 1:length(circ)
            env = environment(left_mps, right_mps, circ, i)
            circ = svd_update(env, circ, i)
        end
        c = sqc_cost(circ, targ_mps)
        push!(cost_list, c)
        if !quiet
            if sweep%50 == 0
                println(infidelities[end])
            end
        end
    end
    save(filename, "cost_list", infidelities, "circ", circ)
    return infidelities, circ 
end