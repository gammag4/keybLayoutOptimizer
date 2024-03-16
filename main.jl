import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

#using Revise
using Base.Filesystem: cptree
using Printf: @sprintf
using Random: rand
using StableRNGs: LehmerRNG
using BenchmarkTools: @time

include("src/DataProcessing.jl")
include("src/DrawKeyboard.jl")
include("src/Types.jl")
include("src/Utils.jl")
include("src/DataStats.jl")
include("src/FrequencyKeyboard.jl")
include("src/KeyboardGenerator.jl")
include("src/Presets.jl")

include("src/Genome.jl")
include("src/KeyboardObjective.jl")
include("src/SimulatedAnnealing.jl")

using .Presets: runId, randomSeed, dataStats, keyboardData, frequencyRewardArgs, algorithmArgs, cpuArgs, gpuArgs, rewardArgs, dataPaths
using .Types: GPUArgs, CPUArgs
using .FrequencyKeyboard: createFrequencyGenome, drawFrequencyKeyboard
using .DrawKeyboard: drawKeyboard
using .KeyboardObjective: objectiveFunction
using .SimulatedAnnealing: chooseSA

useGPU = true

const (lastRunsPath, finalResultsPath) = dataPaths
const (; keyMap) = keyboardData

const (; numKeyboards) = algorithmArgs

frequencyGenome, freqKeyMap = createFrequencyGenome(dataStats, keyboardData, frequencyRewardArgs)
drawFrequencyKeyboard(joinpath(finalResultsPath, "frequencyKeyboard.png"), frequencyGenome, freqKeyMap, keyboardData, useFrequencyColorMap=false)

function multipleSA(numKeyboards, useGPU)
    computationArgs = useGPU ? gpuArgs : cpuArgs

    genomes = Dict{Any,Any}()
    objectives = Dict{Any,Any}()
    rngs = LehmerRNG.(rand(LehmerRNG(randomSeed), 1:typemax(Int), numKeyboards))
    #genomeGenerator=() -> shuffleKeyMap(rngs[i], keyMap, fixedKeys)
    generators = [() -> frequencyGenome for _ in 1:numKeyboards]
    baselineScore = objectiveFunction(keyMap, computationArgs, rewardArgs)

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
    chooseSA(numKeyboards, genomes, objectives, rngs, generators, saArgs, Val(useGPU))

    bestI, bestG, bestO = reduce(((i, g, o), (i2, g2, o2)) -> o < o2 ? (i, g, o) : (i2, g2, o2), ((i, genomes[i], objectives[i]) for i in filter(x -> haskey(genomes, x), eachindex(genomes))))

    println("Best overall: $bestI; Score: $bestO")

    drawKeyboard(bestG, joinpath(finalResultsPath, "bestOverall.png"), keyboardData)
    cptree(finalResultsPath, joinpath(lastRunsPath, "$runId"))
end

# TODO Use gpu in objective function
# Run julia --threads=<num threads your processor can run -1>,1 --project=. main.jl
# Example for 12 core processor, 2 threads per core, total 24 threads: julia --threads=23,1 --project=. main.jl
# Genomes are keymaps
@time multipleSA(numKeyboards, useGPU)
