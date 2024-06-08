""" 
mps_utils.jl
Author: Faisal Alam
Date: 6/7/2024
useful functions for ITensors MPS class
"""

using ITensors

" turns an MPS into a statevector "
function mps_to_vec(mps)
    sv = contract(mps)
    idx = inds(sv)
    vec = reshape(Array(sv, idx...), 2^length(idx))
    return vec
end

" turns a qubit index into a binary string "
function index_to_binary_string(index, num_qubits)
    binary_str = ""
    for qubit_idx in 0:num_qubits-1
        binary_str *= (index & 1) == 0 ? "0" : "1"
        index >>= 1
    end
    return reverse(binary_str)
end

" writes a statevector in computational basis notation; qubit 1 in circuit corresponds to rightmost qubit in the ket "
function pretty_state(statevector; precision=1e-8)
    num_qubits = Int(log2(length(statevector)))
    amplitudes = abs.(statevector).^2
    sups = findall(amplitudes .> precision)    
    states = [index_to_binary_string(index - 1, num_qubits) for index in sups]
    amps = [string(round(statevector[index], digits=4)) for index in sups]
    result_string = join([amps[i] * "|" * states[i] * ">" for i in 1:length(sups)], " + ")
    return result_string
end

" returns mps with its indices renamed; often used before calling inner() "
function rename_indices(mps, new_inds)
    s = deepcopy(siteinds(mps))
    for j in 1:length(mps)
      replaceind!(mps[j],s[j],new_inds[j])
    end
    return mps
end

" returns mps with |0> inserted at loc; used to /fake/ measurement followed by overlap; loc cannot be an edge "
function insert_zero_ket(mps, loc)
    if loc in [1, length(mps)]
        error("unimplemented")
    else
        new_mps = deepcopy(mps)
        sn = siteind("Qubit", loc) 
        ln = linkind(new_mps, loc-1)
        ln_sim = sim(ln)
        new_site = onehot(sn => 1) * delta(dag(ln), ln_sim)
        new_mps[loc] *= delta(dag(ln_sim), ln)
        insert!(new_mps.data, loc, new_site)
        return new_mps
    end
end