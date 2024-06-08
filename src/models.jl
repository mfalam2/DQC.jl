""" 
models.jl
Author: Faisal Alam
Date: 6/7/2024
paradigmatic classes of states to use as targets for DQCs
"""

include("gates.jl")

" generates a n_clusters product of GHZ states of size cluster_size "  
function ghz_clusters(cluster_size, n_clusters)
    circuit = []
    for j in 1:n_clusters
        start = 1 + cluster_size*(j-1)
        circuit = [circuit; ("H", start)]
        for i in start:start+cluster_size-2
            circuit = [circuit; ("CX", (i,i+1))]
        end
    end
    
    for j in 1:n_clusters-1
        fused = j*cluster_size
        circuit = [circuit; ("CX", (fused,fused+1))]
        circuit = [circuit; ("H", fused)]
    end
    
    hilbert = qubits(cluster_size*n_clusters)
    mps = runcircuit(hilbert, circuit)
    return mps
end

" returns ground state of the critical transverse-field Ising model " 
function tfim_gs(N)
    sites = siteinds("Qubit",N)
    os = OpSum()
    for j=1:N-1
        os += -4.0,"Sz",j,"Sz",j+1
        os += -2.0,"Sx",j  # set to -2 for critical point
    end
    os += -2.0,"Sx",N   # set to -2 for critical point
    H = MPO(os,sites)
    
    psi0 = randomMPS(sites,N)
    nsweeps = 5
    maxdim = [N,2*N,3*N,4*N,5*N]
    cutoff = [1E-10]
    energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff, outputlevel=0)
    
    return psi
end  

" returns ground state of the long-range Ising model " 
function lrim_gs(N)
    sites = siteinds("S=1/2",N)
    os = OpSum()
    
    for j=1:N
        for k=j+1:N
            os += (1/abs(j-k)),"Sx",j,"Sx",k
        end
    end
    for j=1:N
        os += -1.0,"Sz",j  # set to -2 for critical point
    end
    H = MPO(os,sites)
    
    psi0 = randomMPS(sites,N)
    nsweeps = 5
    maxdim = [N,2*N,3*N,4*N,5*N]
    cutoff = [1E-10]
    energy, psi = dmrg(H,psi0; nsweeps, maxdim, cutoff)
    
    return psi
end    

" returns a random MPS with long-range entanglement "
function lre_mps(N)
    lre = false 
    while !lre 
        mps = randomMPS(siteinds("Qubit", N); linkdims=1)
        circ = [("gate_from_mat", (1,N), (mat=kak_matrix(rand(15)),))]
        targ_mps = runcircuit(deepcopy(mps), circ)
        infidelities, _ = sqc_svd(targ_mps, Int(N/2), "junk.jld"; num_sweeps=500)
        
        if infidelities[end] > 1e-2
            lre = true
        else
            println("Failed! Sampling again!")
        end
    end
    return targ_mps
end