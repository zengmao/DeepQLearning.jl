abstract type AbstractNNPolicy  <: Policy end

## NN Policy interface

"""
    getnetwork(policy)
    return the  value network of the policy 
"""
function getnetwork end 

"""
    resetstate!(policy)
reset the hidden states of a policy
"""
function resetstate! end

struct NNPolicy{P, Q, A} <: AbstractNNPolicy 
    problem::P
    qnetwork::Q
    action_map::Vector{A}
    n_input_dims::Int64
end

NNPolicy(problem::MDPCommonRLEnv, qnetwork, action_map::Vector, n_input_dims::Int) = NNPolicy(convert(MDP, problem), qnetwork, action_map, n_input_dims)
NNPolicy(problem::POMDPCommonRLEnv, qnetwork, action_map::Vector, n_input_dims::Int) = NNPolicy(convert(POMDP, problem), qnetwork, action_map, n_input_dims)


function getnetwork(policy::NNPolicy)
    return policy.qnetwork
end

function resetstate!(policy::NNPolicy)
    Flux.reset!(policy.qnetwork)
end

actionmap(p::NNPolicy) = p.action_map

function _action(policy::NNPolicy, o; random = false, rng = Random.default_rng())
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        vals = policy.qnetwork(obatch)
        if !random
            # println("we're here: b1")
            return policy.action_map[argmax(vals)]
        else
            # println("We're here: b2")
            return policy.action_map[rand(rng, filter(i->vals[i] != typemin(Float32), 1:length(policy.action_map)))]
        end
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function _actionvalues(policy::NNPolicy{P,Q,A}, o::AbstractArray{T,N}) where {P,Q,A,T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        return dropdims(policy.qnetwork(obatch), dims=2)
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

function _value(policy::NNPolicy{P}, o::AbstractArray{T,N}) where {P,T<:Real,N}
    if ndims(o) == policy.n_input_dims
        obatch = reshape(o, (size(o)...,1))
        return maximum(policy.qnetwork(obatch))
    else 
        throw("NNPolicyError: was expecting an array with $(policy.n_input_dims) dimensions, got $(ndims(o))")
    end
end

POMDPs.action(policy::NNPolicy, o; random = false) = _action(policy, o, random=random)
POMDPs.action(policy::NNPolicy{P}, s; random = false) where {P <: MDP} = _action(policy, POMDPs.convert_s(Array{Float32}, s, policy.problem), random=random)
POMDPs.action(policy::NNPolicy{P}, o; random = false) where {P <: POMDP} = _action(policy, POMDPs.convert_o(Array{Float32}, o, policy.problem), random=random)

POMDPTools.actionvalues(policy::NNPolicy, o) = _actionvalues(policy, o)
POMDPTools.actionvalues(policy::NNPolicy{P}, s) where {P<:MDP} = _actionvalues(policy, POMDPs.convert_s(Array{Float32}, s, policy.problem))
POMDPTools.actionvalues(policy::NNPolicy{P}, o) where {P<:POMDP} = _actionvalues(policy, POMDPs.convert_o(Array{Float32}, o, policy.problem))

POMDPs.value(policy::NNPolicy, o) = _value(policy, o)
POMDPs.value(policy::NNPolicy{P}, s) where {P <: MDP} = _value(policy, POMDPs.convert_s(Array{Float32}, s, policy.problem))
POMDPs.value(policy::NNPolicy{P}, o) where {P <: POMDP} = _value(policy, POMDPs.convert_o(Array{Float32}, o, policy.problem))
