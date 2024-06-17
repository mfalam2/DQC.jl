include("/media/HomeData/mfalam2/current_projects/DQC.jl/src/models.jl")
include("/media/HomeData/mfalam2/current_projects/DQC.jl/src/train_sqc.jl")
include("/media/HomeData/mfalam2/current_projects/DQC.jl/src/train_dqc.jl")

args = ARGS

targ_type = args[1]  # "ghz", "tfi", "subset"
n = parse(Int, args[2]) # number of qubits

n_clusters = parse(Int, args[3]) # this is 1 for static and > 1 for dynamic
n_layers = parse(Int, args[4]) # number of KAK layers
instance = parse(Int, args[5]) 

if targ_type == "tfi"
    filename = string("tfi_", n, "_", n_clusters, "_", n_layers, "_", instance, ".jld")
    targ_mps = tfim_gs(n)
elseif targ_type == "ghz"
    filename = string("ghz_", n, "_", n_clusters, "_", n_layers, "_", instance, ".jld")
    targ_mps = ghz_clusters(n,1)   
else
    filename = string("subset_", n, "_", n_clusters, "_", n_layers, "_", instance, ".jld")
    targ_mps = subset_state(n)
end

if n_clusters > 1
    ansatz =  generate_dqc_ansatz(targ_mps, n_clusters, n_layers)
    dqc_svd(targ_mps, ansatz, filename);
else
    ansatz = generate_sqc_ansatz(targ_mps, n_layers)
    sqc_svd(targ_mps, ansatz, filename);        
end