import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

#using Revise
using Base.Filesystem: cptree
using Printf: @sprintf
using Random: rand
using StableRNGs: LehmerRNG
using BenchmarkTools: @time
using CUDA: CuArray
using JSON: parse as jparse

include("src/DataProcessing.jl")
include("src/DrawKeyboard.jl")
include("src/Types.jl")
include("src/Utils.jl")
include("src/DataStats.jl")
include("src/FrequencyKeyboard.jl")
include("src/KeyboardGenerator.jl")

include("src/Genome.jl")
include("src/KeyboardObjective.jl")
include("src/SimulatedAnnealing.jl")

using .Utils: conditionalSplit, dictToArray, dictToNamedTuple, minMaxScale
using .DataProcessing: processDataFolderIntoTextFile
using .DataStats: computeStats
using .KeyboardGenerator: layoutGenerator, keyMapGenerator
using .DrawKeyboard: computeKeyboardColorMap
using .Types: RewardArgs, FrequencyRewardArgs, LayoutKey, KeyboardData, CPUArgs, GPUArgs
using .FrequencyKeyboard: createFrequencyKeyMap, createFrequencyGenome, drawFrequencyKeyboard
using .KeyboardObjective: objectiveFunction
using .SimulatedAnnealing: chooseSA
using .Genome: shuffleKeyMap

function main()
    jsonData = dictToNamedTuple(open(f -> jparse(f), "persistent/data.json", "r"))
    (;
        textPath,
        dataPaths,
        keyMap,
        noCharKeyMap,
        randomSeed,
        dataStats,
        keyboardLayout,
        fixedKeys,
        handFingers,
        algorithmArgs,
        keyboardSize,
        frequencyRewardArgs,
        rewardArgs,
    ) = jsonData

    dataPaths = dictToNamedTuple(dataPaths)
    (; persistentPath, rawDataPath, dataPath, lastRunsPath, finalResultsPath, startResultsPath, endResultsPath) = dataPaths

    # Creating folders and removing old data
    map(i -> rm(joinpath(dataPath, "$i"), recursive=true), filter(s -> occursin(r"result", s), readdir(dataPath)))
    for path in dataPaths
        mkpath(path)
    end
    runId = 1 + last(sort(vcat([0], collect(map(i -> parse(Int, replace(i, r"[^0-9]" => "")), readdir(lastRunsPath))))))


    # TODO Split layout into list of keys with same size so that they can be shuffled
    keyMap = dictToNamedTuple(keyMap)
    keyMap = keyMapGenerator(
        startIndices=Vector{Int}(keyMap.startIndices),
        keys=Vector{String}(keyMap.keys)
    )
    keyMapCharacters = Set(keys(keyMap))
    noCharKeyMap = dictToNamedTuple(noCharKeyMap)
    noCharKeyMap = keyMapGenerator(
        startIndices=Vector{Int}(noCharKeyMap.startIndices),
        keys=Vector{Vector{String}}(noCharKeyMap.keys)
    )

    # Processing data
    processDataFolderIntoTextFile(rawDataPath, textPath, keyMapCharacters, overwrite=false, verbose=true)

    # Getting data
    textData = open(io -> read(io, String), textPath, "r")

    (; fingersCPS, rowsCPS) = dictToNamedTuple(dataStats)
    dataStats = computeStats(;
        text=textData,
        fingersCPS=Vector{Float64}(fingersCPS),
        rowsCPS=Vector{Float64}(rowsCPS)
    )

    (; textStats) = dataStats
    (; charFrequency) = textStats
    keyboardColorMap = computeKeyboardColorMap(charFrequency)

    layoutMap = layoutGenerator(; dictToNamedTuple(keyboardLayout)...)

    fixedKeys = Set(fixedKeys)
    # const fixedKeys = collect("\t\n ") # Numbers also change
    getFixedMovableKeyMaps(keyMap) = conditionalSplit(((k, v),) -> k in fixedKeys, keyMap)
    fixedKeyMap, movableKeyMap = getFixedMovableKeyMaps(keyMap)
    movableKeys = [k for (k, v) in movableKeyMap]
    handFingers = Vector{Int}(handFingers)
    numFingers = length(handFingers)
    numKeys = length(keyMap)
    numLayoutKeys = length(layoutMap)
    numFixedKeys = length(fixedKeyMap)
    numMovableKeys = length(movableKeyMap)

    # Total number of iterations will be -epoch * log(t) / log(coolingRate)
    algorithmArgs = dictToNamedTuple(algorithmArgs)

    (; effortWeighting, xBias, leftHandBias, rowCPSBias) = dictToNamedTuple(frequencyRewardArgs)
    frequencyRewardArgs = FrequencyRewardArgs(;
        effortWeighting=NTuple{2,Float64}(effortWeighting),
        xBias=xBias,
        leftHandBias=leftHandBias,
        rowCPSBias=NTuple{6,Float64}(rowCPSBias),
        ansKbs=1 / keyboardSize,
    )

    (; effortWeighting, doubleFingerEffort, singleHandEffort, rewardMapEffort) = dictToNamedTuple(rewardArgs)
    rewardArgs = RewardArgs(;
        effortWeighting=NTuple{3,Float64}(effortWeighting),
        doubleFingerEffort=doubleFingerEffort,
        singleHandEffort=singleHandEffort,
        rewardMapEffort=rewardMapEffort
    )

    keyboardData = KeyboardData(
        keyboardColorMap,
        layoutMap,
        keyMapCharacters,
        keyMap,
        noCharKeyMap,
        fixedKeyMap,
        movableKeyMap,
        fixedKeys,
        movableKeys,
        getFixedMovableKeyMaps,
        handFingers,
        numFingers,
        numKeys,
        numLayoutKeys,
        numFixedKeys,
        numMovableKeys,
    )

    rewardKeyMap = createFrequencyKeyMap(dataStats, keyboardData, frequencyRewardArgs)
    frequencyGenome, freqKeyMap = createFrequencyGenome(dataStats, keyboardData, rewardKeyMap)

    td = collect(textData)
    lm = dictToArray(layoutMap)
    rkm = minMaxScale(dictToArray(rewardKeyMap), 1, 0)

    cpuArgs = CPUArgs(
        text=td,
        layoutMap=lm,
        handFingers=handFingers,
        rewardMap=rkm,
    )

    gpuArgs = GPUArgs(
        numThreadsInBlock=512,
        text=CuArray(td),
        layoutMap=CuArray(lm),
        handFingers=CuArray(handFingers),
        rewardMap=CuArray(rkm),
    )

    useGPU = true
    (; numKeyboards) = algorithmArgs

    computationArgs = useGPU ? gpuArgs : cpuArgs

    genomes = Dict{Any,Any}()
    objectives = Dict{Any,Any}()
    rngs = LehmerRNG.(rand(LehmerRNG(randomSeed), 1:typemax(Int), numKeyboards))
    generators = vcat([() -> frequencyGenome], [() -> shuffleKeyMap(rngs[i], keyMap, fixedKeys) for i in 1:(numKeyboards-1)])
    baselineScore = objectiveFunction(keyMap, computationArgs, rewardArgs)

    println("Drawing frequency keymap...")
    drawFrequencyKeyboard(joinpath(finalResultsPath, "frequencyKeyboard.png"), frequencyGenome, freqKeyMap, keyboardData, useFrequencyColorMap=true)

    println(@sprintf "Raw baseline: %.2f" baselineScore)
    println("From here everything is reletive with + % worse and - % better than this baseline")

    saArgs = (
        baselineScore=baselineScore,
        computationArgs=computationArgs,
        rewardArgs=rewardArgs,
        keyboardData=keyboardData,
        algorithmArgs=algorithmArgs,
        dataPaths=dataPaths
    )

    # TODO Use Distributed.@distributed to get results
    @time chooseSA(numKeyboards, genomes, objectives, rngs, generators, saArgs, Val(useGPU))

    bestI, bestG, bestO = reduce(((i, g, o), (i2, g2, o2)) -> o < o2 ? (i, g, o) : (i2, g2, o2), ((i, genomes[i], objectives[i]) for i in filter(x -> haskey(genomes, x), eachindex(genomes))))

    println("Best overall: $bestI; Score: $bestO")

    drawKeyboard(bestG, joinpath(finalResultsPath, "bestOverall.png"), keyboardData)
    cptree(finalResultsPath, joinpath(lastRunsPath, "$runId"))
end
