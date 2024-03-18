module Types

using Adapt: @adapt_structure
using CUDA: CuArray
using Parameters: @with_kw
using Colors: HSV

export RewardArgs, RewardMapArgs, LayoutKey, KeyboardData, CPUArgs, GPUArgs

@with_kw struct RewardArgs
    effortWeighting::NTuple{4,Float64}
    yScale::Float64
    distGrowthRate::Float64
end

@adapt_structure RewardArgs

@with_kw struct RewardMapArgs
    rewardWeighting::NTuple{3,Float64}
    rowsCPSBias::NTuple{6,Float64}
end

const LayoutKey = Tuple{NTuple{4,Float64},NTuple{2,Int},Int}

@with_kw struct KeyboardData
    keyboardColorMap::Dict{Char,HSV}
    layoutMap::Vector{LayoutKey}
    vertLayoutMap
    keyMapCharacters::Set{Char}
    keyMap::Dict{Char,Int}
    noCharKeyMap::Dict{String,Int}
    fixedKeyMap::Dict{Char,Int}
    movableKeyMap::Dict{Char,Int}
    fixedKeys::Set{Char}
    movableKeys::Vector{Char}
    getFixedMovableKeyMaps::Function
    handFingers::Vector{Int}
    numFingers::Int
    numKeys::Int
    numLayoutKeys::Int
    numFixedKeys::Int
    numMovableKeys::Int
end

@with_kw struct CPUArgs
    text::Vector{Tuple{UInt8,UInt8}}
    layoutMap::Vector{LayoutKey}
    handFingers::Vector{Int}
    rewardMap::Vector{Float64}
end

@with_kw struct GPUArgs
    numThreadsInBlock::Int
    text::CuArray{Tuple{UInt8,UInt8}}
    layoutMap::CuArray{LayoutKey}
    handFingers::CuArray{Int}
    rewardMap::CuArray{Float64}
end

@adapt_structure GPUArgs

end
