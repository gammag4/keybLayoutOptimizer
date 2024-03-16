import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

#using Revise
using Base.Filesystem: cptree
using Base.Iterators: countfrom
using Base.Threads: @spawn, @threads, threadid, nthreads
using Printf: @sprintf
using Random: rand
using StableRNGs: LehmerRNG
using BenchmarkTools: @time

include("src/Presets.jl")
include("src/Genome.jl")
include("src/FrequencyKeyboard.jl")
include("src/DrawKeyboard.jl")
include("src/KeyboardObjective.jl")
include("src/Utils.jl")

using .Presets: runId, randomSeed, dataStats, keyboardData, frequencyRewardArgs, algorithmArgs, gpuArgs, rewardArgs, dataPaths
using .Genome: shuffleGenomeKeyMap
using .FrequencyKeyboard: createFrequencyGenome, drawFrequencyKeyboard
using .DrawKeyboard: drawKeyboard
using .KeyboardObjective: objectiveFunction
using .Utils: dictToArray

# Has probability 0.5 of changing current genome to best when starting new epoch
# Has probability e^(-delta/t) of changing current genome to a worse when not the best genome
function runSA(;
    keyboardId,
    rng,
    gpuArgs,
    rewardArgs,
    baselineScore,
    keyboardData,
    genomeGenerator,
    algorithmArgs,
    dataPaths,
)
    (; fixedKeys) = keyboardData
    (; t, e, nIter, tShuffleMultiplier) = algorithmArgs
    (; startResultsPath, endResultsPath) = dataPaths

    coolingRate = (1 / t)^(e / nIter)

    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(currentGenome, gpuArgs, rewardArgs, baselineScore)

    bestGenome, bestObjective = currentGenome, currentObjective

    drawKeyboard(bestGenome, joinpath(startResultsPath, "$keyboardId.png"), keyboardData)

    baseT = t
    try
        for iteration in countfrom(1)
            t ≤ 1 && break

            # TODO Shuffle genome in GPU
            # Create new genome
            newGenome = shuffleGenomeKeyMap(rng, currentGenome, fixedKeys, floor(Int, t * tShuffleMultiplier))

            # Asess
            newObjective = objectiveFunction(newGenome, gpuArgs, rewardArgs, baselineScore)
            delta = newObjective - currentObjective

            if iteration % 1000 == 1
                skid = lpad(keyboardId, 3, " ")
                sit = lpad(iteration, 6, " ")
                stemp = lpad((@sprintf "%.2f" t), 7, " ")
                sobj = lpad((@sprintf "%.3f" bestObjective), 8, " ")
                snobj = lpad((@sprintf "%.3f" newObjective), 8, " ")
                updateLine = "Keyboard: $skid, Iteration: $sit, temp: $stemp, best obj: $sobj, new obj: $snobj"

                println(updateLine)
            end

            # If new keyboard is better (less objective is better)
            if delta < 0
                currentGenome, currentObjective = newGenome, newObjective

                if newObjective < bestObjective
                    bestGenome, bestObjective = newGenome, newObjective
                end

                # Changes genome with probability e^(-delta/t)
            elseif exp(-delta / t) > rand(rng)
                currentGenome, currentObjective = newGenome, newObjective
            end

            # Starting new epoch
            if iteration > e && (iteration % e == 1)
                t = baseT * coolingRate^floor(Int, iteration / e)

                # Changes genome with probability 0.5
                if rand(rng) < 0.5
                    currentGenome, currentObjective = bestGenome, bestObjective
                end
            end
        end
    catch e
        if !(e isa InterruptException)
            rethrow(e)
        end
    end

    drawKeyboard(bestGenome, joinpath(endResultsPath, "$keyboardId.png"), keyboardData)

    return bestGenome, bestObjective
end

const (lastRunsPath, finalResultsPath) = dataPaths
const (; keyMap) = keyboardData

const (; numKeyboards) = algorithmArgs
rngs = LehmerRNG.(rand(LehmerRNG(randomSeed), 1:typemax(Int), numKeyboards))
genomes = Dict{Any,Any}()
objectives = Dict{Any,Any}()

frequencyGenome, freqKeyMap = createFrequencyGenome(dataStats, keyboardData, frequencyRewardArgs)
drawFrequencyKeyboard(joinpath(finalResultsPath, "frequencyKeyboard.png"), frequencyGenome, freqKeyMap, keyboardData, useFrequencyColorMap=false)

# TODO Use gpu in objective function
# Run julia --threads=<num threads your processor can run -1>,1 --project=. main.jl
# Example for 12 core processor, 2 threads per core, total 24 threads: julia --threads=23,1 --project=. main.jl
# Genomes are keymaps
@time begin
    baselineScore = objectiveFunction(keyMap, gpuArgs, rewardArgs)
    println(@sprintf "Raw baseline: %.2f" baselineScore)
    println("From here everything is reletive with + % worse and - % better than this baseline")

    # TODO Use Distributed.@distributed to get results
    for i in 1:numKeyboards
        genomes[i], objectives[i] = runSA(
            keyboardId=i,
            rng=rngs[i],
            gpuArgs=gpuArgs,
            rewardArgs=rewardArgs,
            baselineScore=baselineScore,
            keyboardData=keyboardData,
            #genomeGenerator=() -> shuffleKeyMap(rngs[i], keyMap, fixedKeys),
            genomeGenerator=() -> frequencyGenome,
            algorithmArgs=algorithmArgs,
            dataPaths=dataPaths,
        )
    end

    bestI, bestG, bestO = reduce(((i, g, o), (i2, g2, o2)) -> o < o2 ? (i, g, o) : (i2, g2, o2), ((i, genomes[i], objectives[i]) for i in filter(x -> haskey(genomes, x), eachindex(genomes))))

    println("Best overall: $bestI; Score: $bestO")

    drawKeyboard(bestG, joinpath(finalResultsPath, "bestOverall.png"), keyboardData)
    cptree(finalResultsPath, joinpath(lastRunsPath, "$runId"))
end
