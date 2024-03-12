push!(LOAD_PATH, "src/")

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Random, StableRNGs
using .Threads
using BenchmarkTools
using Revise

using Presets
using Genome
using FrequencyKeyboard
using DrawKeyboard

const (; fingerEffort, rowEffort, textStats, effortWeighting) = dataStats
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
    xMoveMultiplier,
    distanceEffort,
    doubleFingerEffort,
    singleHandEffort,
    rightHandEffort,
) = rewardArgs

const (;
    temperature,
    epoch,
    numIterations,
    coolingRate,
    maxIterations,
    temperatureKeyShuffleMultiplier
) = algorithmArgs

function doKeypress(keyPress, fingerData, oldFinger, oldHand, keyboardData)
    (; layoutMap, numFingers, handFingers) = keyboardData
    (x, y, _), (finger, _), row = layoutMap[keyPress]
    currentHand = handFingers[finger]
    homeX, homeY, currentX, currentY, objectiveCounter = fingerData[finger]

    # Sets other fingers back to home position
    for fingerID in 1:numFingers
        hx, hy, _, _, _ = fingerData[fingerID]

        fingerData[fingerID][3] = hx
        fingerData[fingerID][4] = hy
    end

    fingerData[finger][3] = currentX
    fingerData[finger][4] = currentY

    dx, dy = x - currentX, y - currentY
    distance = sqrt((dx * xMoveMultiplier)^2 + dy^2)

    distancePenalty = (distance + 1)^distanceEffort - 1 # This way, distanceEffort always increases even if in [0, 1]

    # Double finger
    doubleFingerPenalty = 0
    if finger == oldFinger && distance > 0.01
        doubleFingerPenalty = doubleFingerEffort
    end

    # Single hand
    singleHandPenalty = 0
    if currentHand == oldHand
        singleHandPenalty = singleHandEffort
    end

    # Right hand
    rightHandPenalty = 0
    if currentHand == 2
        rightHandPenalty = rightHandEffort
    end

    fingerPenalty = fingerEffort[finger]
    rowPenalty = rowEffort[row]

    # Combined weighting
    penalties = (distancePenalty, doubleFingerPenalty, singleHandPenalty, rightHandPenalty, fingerPenalty, rowPenalty) .* effortWeighting
    #println(penalties)
    newObjective = objectiveCounter + sum(penalties)

    fingerData[finger][3] = x
    fingerData[finger][4] = y
    fingerData[finger][5] = newObjective

    return fingerData, finger, currentHand
end

function baselineObjectiveFunction(text, genome, keyboardData)
    (; layoutMap, numFingers) = keyboardData
    # homeX, homeY, currentX, currentY, objectiveCounter
    fingerData = repeat([zeros(5)], 10)

    for i in 1:numFingers
        (x, y, _), (finger, _), _ = layoutMap[i]
        fingerData[finger][1:4] = [x, y, x, y]
    end

    objective = 0
    oldFinger = 0
    oldHand = 0

    for currentCharacter in text
        # determine keypress (nothing if there is no key)
        keyPress = get(genome, currentCharacter, nothing)

        if isnothing(keyPress)
            continue
        end

        # do keypress
        fingerData, oldFinger, oldHand = doKeypress(keyPress, fingerData, oldFinger, oldHand, keyboardData)
    end

    # calculate and return objective
    objective = sum(map(x -> x[5], fingerData))
    return objective
end

function objectiveFunction(text, genome, keyboardData, baselineScore)
    objective = (baselineObjectiveFunction(text, genome, keyboardData) / baselineScore - 1) * 100
    return objective
end

# Has probability 0.5 of changing current genome to best when starting new epoch
# Has probability e^(-delta/t) of changing current genome to a worse when not the best genome
function runSA(;
    runid,
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
    mkpath("data/result$runid")

    verbose && println("Running code...")
    verbose && print("Calculating raw baseline: ")
    baselineScore = baselineObjectiveFunction(text, baselineGenome, keyboardData)
    verbose && println(baselineScore)
    verbose && println("From here everything is reletive with + % worse and - % better than this baseline \n Note that best layout is being saved as a png at each step. Kill program when satisfied.")

    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(text, currentGenome, keyboardData, baselineScore)

    bestGenome = currentGenome
    bestObjective = currentObjective

    Threads.@spawn :interactive drawKeyboard(bestGenome, "data/result/$runid - first.png", keyboardData, lk)

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

            srid = lpad(runid, 3, " ")
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

                open(f -> write(f, updateLine, "\n"), "data/result/iterationScores$runid.txt", "a")

                if newObjective < bestObjective
                    bestGenome = newGenome
                    bestObjective = newObjective

                    verbose && println("(new best, text being saved)")
                    Threads.@spawn :interactive drawKeyboard(bestGenome, "data/result$runid/$iteration.png", keyboardData, lk)
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

    Threads.@spawn :interactive drawKeyboard(bestGenome, "data/result/$runid - final.png", keyboardData, lk)

    return bestGenome, bestObjective
end

const nts = Threads.nthreads()

seed = 563622
const rng = StableRNGs.LehmerRNG(seed)
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
                runid=tid,
                lk=lk,
                rng=rngs[i],
                text=textData,
                keyboardData=keyboardData,
                baselineGenome=keyMap,
                genomeGenerator=() -> shuffleKeyMap(rngs[i], keyMap, fixedKeys),
                #genomeGenerator=() -> keyMapDict,
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
end
