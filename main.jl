push!(LOAD_PATH, "src/")

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Random, StableRNGs
using .Threads
using BenchmarkTools
using Revise
using Base.Filesystem

using Presets
using Genome
using FrequencyKeyboard
using DrawKeyboard
using KeyboardObjective

const (; fingerEffort, rowEffort, textStats) = dataStats
const (; charHistogram, charFrequency, usedChars) = textStats

const (;
    keyboardColorMap,
    layoutMap,
    keyMap,
    noCharKeyMap,
    fixedKeys,
    fingersHome,
    handFingers,
    numFingers,
    numKeys,
    numLayoutKeys,
    numMovableKeys
) = keyboardData

const (;
    temperature,
    epoch,
    numIterations,
    coolingRate,
    maxIterations,
    temperatureKeyShuffleMultiplier
) = algorithmArgs

# Has probability 0.5 of changing current genome to best when starting new epoch
# Has probability e^(-delta/t) of changing current genome to a worse when not the best genome
function runSA(;
    threadId,
    lk,
    rng,
    text,
    keyboardData,
    baselineGenome,
    genomeGenerator,
    temperature,
    epoch,
    coolingRate,
    maxIterations,
    temperatureKeyShuffleMultiplier,
    verbose,
)
    mkpath("data/result$threadId")

    verbose && println("Running code...")
    verbose && print("Calculating raw baseline: ")
    baselineScore = objectiveFunction(text, baselineGenome, keyboardData)
    verbose && println(baselineScore)
    verbose && println("From here everything is reletive with + % worse and - % better than this baseline \n Note that best layout is being saved as a png at each step. Kill program when satisfied.")

    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(text, currentGenome, keyboardData, baselineScore)

    bestGenome = currentGenome
    bestObjective = currentObjective

    Threads.@spawn :interactive drawKeyboard(bestGenome, "data/result/first/$threadId.png", keyboardData, lk)

    baseTemp = temperature
    try
        for iteration in 1:maxIterations
            if temperature ≤ 1
                break
            end

            # Create new genome
            newGenome = shuffleGenomeKeyMap(rng, currentGenome, fixedKeys, floor(Int, temperature * temperatureKeyShuffleMultiplier))

            # Asess
            newObjective = objectiveFunction(text, newGenome, keyboardData, baselineScore)
            delta = newObjective - currentObjective

            srid = lpad(threadId, 3, " ")
            sit = lpad(iteration, 6, " ")
            stemp = lpad(round(temperature, digits=2), 7, " ")
            sobj = lpad(round(bestObjective, digits=3), 8, " ")
            snobj = lpad(round(newObjective, digits=3), 8, " ")
            updateLine = "Thread: $srid, Iteration: $sit, temp: $stemp, best obj: $sobj, new obj: $snobj"

            (verbose || iteration % 100 == 1) && println(updateLine)

            if delta < 0
                # If new keyboard is better (less objective is better)

                currentGenome = newGenome
                currentObjective = newObjective

                open(f -> write(f, updateLine, "\n"), "data/result/iterationScores$threadId.txt", "a")

                if newObjective < bestObjective
                    bestGenome = newGenome
                    bestObjective = newObjective

                    verbose && println("(new best, text being saved)")
                    Threads.@spawn :interactive drawKeyboard(bestGenome, "data/result$threadId/$iteration.png", keyboardData, lk)
                    # open("data/result/bestGenomes.txt", "a") do io
                    #     print(io, iteration, ":")
                    #     for c in bestGenome
                    #         print(io, c)
                    #     end
                    #     println(io)
                    # end
                end
            elseif exp(-delta / temperature) > rand(rng)
                # Changes genome with probability e^(-delta/t)

                currentGenome = newGenome
                currentObjective = newObjective
            end

            # Starting new epoch
            if iteration > epoch && (iteration % epoch == 1)
                temperature = baseTemp * coolingRate^floor(Int, iteration / epoch)

                # Changes genome with probability 0.5
                if rand(rng) < 0.5
                    currentGenome = bestGenome
                    currentObjective = bestObjective
                end
            end
        end
    catch e
        if !(e isa InterruptException)
            rethrow(e)
        end
    end

    Threads.@spawn :interactive drawKeyboard(bestGenome, "data/result/final/$threadId.png", keyboardData, lk)

    return bestGenome, bestObjective
end

frequencyGenome, freqKeyMap = createFrequencyGenome(dataStats, keyboardData)
drawFrequencyKeyboard("data/frequencyKeyboard.png", frequencyGenome, freqKeyMap, keyboardData, useFrequencyColorMap=false)

const nts = Threads.nthreads()

const rng = StableRNGs.LehmerRNG(randomSeed)
const rngs = StableRNGs.LehmerRNG.(rand(rng, 1:typemax(Int), nts))
genomes = Dict{Any,Any}()
objectives = Dict{Any,Any}()

# TODO Use gpu in objective function
# Run julia --threads=<num threads your processor can run -1>,1 --project=. main.jl
# Example for 12 core processor, 2 threads per core, total 24 threads: julia --threads=23,1 --project=. main.jl
# Genomes are keymaps
@time begin
    @sync begin
        lk = ReentrantLock()
        Threads.@threads :static for i in 1:nts
            tid = Threads.threadid()

            genome, objective = runSA(
                threadId=tid,
                lk=lk,
                rng=rngs[i],
                text=textData,
                keyboardData=keyboardData,
                baselineGenome=keyMap,
                #genomeGenerator=() -> shuffleKeyMap(rngs[i], keyMap, fixedKeys),
                genomeGenerator=() -> frequencyGenome,
                temperature=temperature,
                epoch=epoch,
                coolingRate=coolingRate,
                maxIterations=maxIterations,
                temperatureKeyShuffleMultiplier=temperatureKeyShuffleMultiplier,
                verbose=false
            )

            # TODO Use Distributed.@distributed to get results
            genomes[tid] = genome
            objectives[tid] = objective
        end
    end

    bestI, bestG, bestO = reduce(((i, g, o), (i2, g2, o2)) -> o < o2 ? (i, g, o) : (i2, g2, o2), ((i, genomes[i], objectives[i]) for i in filter(x -> haskey(genomes, x), eachindex(genomes))))

    println("Best overall: $bestI; Score: $bestO")
    drawKeyboard(bestG, "data/result/bestOverall - $bestI.png", keyboardData)

    cptree("data/result", "data/lastRuns/$runId")
end
