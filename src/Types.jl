module Types

using Adapt: @adapt_structure
using CUDA: CuArray
using Parameters: @with_kw
using Colors: HSV

export RewardArgs, FrequencyRewardArgs, LayoutKey, KeyboardData, GPUArgs

@with_kw struct RewardArgs
    effortWeighting::NTuple{6,Float64}
    xBias::Float64
    distanceEffort::Float64
    doubleFingerEffort::Float64
    singleHandEffort::Float64
    rightHandEffort::Float64
    ansKbs::Float64
end

@adapt_structure RewardArgs

@with_kw struct FrequencyRewardArgs
    effortWeighting::NTuple{2,Float64}
    xBias::Float64
    leftHandBias::Float64
    rowCPSBias::NTuple{6,Float64}
    ansKbs::Float64
end

const LayoutKey = Tuple{NTuple{4,Float64},NTuple{2,Int},Int}

@with_kw struct KeyboardData
    keyboardColorMap::Dict{Char,HSV}
    layoutMap::Dict{Int,LayoutKey}
    keyMapCharacters::Set{Char}
    keyMap::Dict{Char,Int}
    noCharKeyMap::Dict{String,Int}
    fixedKeyMap::Dict{Char,Int}
    movableKeyMap::Dict{Char,Int}
    fixedKeys::Vector{Char}
    movableKeys::Vector{Char}
    getFixedMovableKeyMaps::Function # TODO Specify type and return??
    handFingers::Vector{Int}
    numFingers::Int
    numKeys::Int
    numLayoutKeys::Int
    numFixedKeys::Int
    numMovableKeys::Int
end

@with_kw struct GPUArgs
    numThreadsInBlock::Int
    text::CuArray{Char}
    layoutMap::CuArray{LayoutKey}
    handFingers::CuArray{Int}
    fingerEffort::CuArray{Float64}
    rowEffort::CuArray{Float64}
end

end
