include("sqc_learn.jl")
include("dqc_learn.jl")
include("models.jl")

args = ARGS

targ_type = args[1]  # "ghz", "tfi"
n = parse(Int, args[2]) # number of qubits

n_clusters = parse(Int, args[3]) # this is 1 for static and > 1 for dynamic
n_layers = parse(Int, args[4]) # number of KAK layers
instance = parse(Int, args[5]) 

if targ_type == "tfi"
    filename = string("tfi_", n, "_", n_clusters, "_", n_layers, "_", instance, ".jld")
    targ_mps = tfim_gs(n)
else
    filename = string("ghz_", n, "_", n_clusters, "_", n_layers, "_", instance, ".jld")
    targ_mps = ghz_clusters(n,1)    
end

if n_clusters > 1
    dqc_lookup_svd(targ_mps, n_clusters, n_layers, filename);
else
    sqc_svd(targ_mps, n_layers, filename);
end