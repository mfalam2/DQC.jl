"""
environments.jl
Author: Faisal Alam
Date: 6/7/2024
environments of parametrized gates 
"""

include("gates.jl")

" environments of one or two qubit gates "
function environment(left_mps, right_mps, circ, i)
    a_mps = deepcopy(left_mps)
    b_mps = deepcopy(right_mps)
    
    a_circ = [("gate_from_mat", g[2], (mat=conj.(g[3].mat),)) for g in circ[1:i-1]]
    a_mps = runcircuit(a_mps, a_circ)
    
    b_circ = [dag(swapprime(gate(siteinds(b_mps),g), 0,1)) for g in reverse(circ[i+1:end])] # need to reverse later
    b_mps = runcircuit(b_mps, b_circ)
    
    hilbert = siteinds(a_mps)
    for loc in circ[i][2]
        ind = hilbert[loc]
        replaceind!(a_mps[loc], ind, prime(ind))
    end
    
    env = a_mps[1] * b_mps[1] 
    for j in 2:length(hilbert)
        env = env * a_mps[j] * b_mps[j]
    end
    
    if length(circ[i][2]) == 2
        return reshape(permutedims(array(dag(env)),[3, 1, 4, 2]),4,4) 
    else
        return array(env)
    end
end
    
" uses environments to update gate "   
function svd_update(env, circ, gate_idx)
    U,S,V = svd(env)
    new_gate = V * U'
    circ[gate_idx] = ("gate_from_mat", circ[gate_idx][2], (mat=new_gate,))
    return circ 
end