push!(LOAD_PATH, "src/")

import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()
using Plots
using Colors
using Random, StableRNGs
using Base.Threads
using BenchmarkTools
using Revise

using Presets
using DataStats
using DataProcessing
using Genome

# TODO Put code after functions and remove globals

textpath = "data/dataset.txt" # File to save/get text data

# Processing data
mkpath("data/result")
processDataFolderIntoTextFile("data/raw_dataset", textpath, overwrite=false, verbose=true)

# Getting data
textdata = open(io -> read(io, String), textpath, "r")

const (; fingerEffort, rowEffort, textStats, effortWeighting) = computeStats(
    text=textdata,
    fingersCPS=[5.5, 5.9, 6.3, 6.2, 6.4, 5.3, 7.0, 6.7, 5.2, 6.2], # Tested by just pressing the home key of each finger
    rowsCPS=[2.36, 2.4, 5.07, 2.6, 2.47, 1.87], # Top to bottom, tested by bringing the pinky to the respective key and going back to the home key
    effortWeighting=(1.2, 1, 1, 0.7, 0.4, 0.4), # dist, double finger, single hand, right hand, finger cps, row cps
    #effortWeighting=(0.7917, 1, 0, 0, 0.4773, 0.00), # dist, double finger, single hand, right hand, finger cps, row cps
)
const (; charHistogram, charFrequency, usedChars) = textStats

const distanceEffort = 1.2 # Always positive. At 2, distance penalty is squared
const doubleFingerEffort = 1 # Positive prevents using same finger more than once
const singleHandEffort = 1 # Positive prefers double hand, negative prefers single hand
const rightHandEffort = 1 # Has to use the mouse

# Hue goes from 170 (min f) to 0 (max f)
# Saturation is the normalized frequency of each key
normFreqToHSV(f) = HSV((1.0 - f) * 170, f * 0.7 + 0.3, 1.0)

function computeKeyboardColorMap(charFrequency)
    charFrequency = filter(((k, v),) -> !isspace(k), charFrequency)
    # Normalizes char frequency
    maxf = maximum(values(charFrequency))
    minf = minimum(values(charFrequency))

    return Dict(k => normFreqToHSV(log2(1 + (v - minf) / (maxf - minf))) for (k, v) in charFrequency)
end

const keyboardColorMap = computeKeyboardColorMap(charFrequency)

function drawKey(key, letter)
    (x, y, w), (finger, home), row = key
    h = 1 # TODO Add h to layout
    color = get(keyboardColorMap, lowercase(letter), HSV(220, 0.2, 1))
    border = Shape((x - 0.5 * w) .+ [0, w, w, 0], (y - 0.5 * h) .+ [0, 0, h, h])
    rect = Shape((x - 0.5 * w + 0.03) .+ ((w - 0.06) .* [0, 1, 1, 0]), (y - 0.5 * h + 0.03) .+ ((h - 0.06) .* [0, 0, 1, 1]))

    plot!(border, fillalpha=1, linecolor=nothing, color=HSV((finger - 1) * 720 / numFingers, 1, 1), label="", dpi=100) # Border
    plot!(rect, fillalpha=1, linecolor=nothing, color=HSVA(color, 0.5), label="", dpi=100)

    if home == 1
        #plot!(rect, fillalpha=0.2, linecolor=nothing, color=HSVA(0, 0, 0, 0.3), label="", dpi=100)
        plot!([x], [y - 0.33], shape=:rect, fillalpha=0.2, linecolor=nothing, color=HSV(0, 0, 0), label="", markersize=1.5, dpi=100)
    end

    # Draws character
    annotate!(x, y, text(uppercase(strip(string(letter))), :black, :center, 8))
end

function drawKeyboard(myGenome, filepath, layoutMap)
    plot(axis=([], false))

    for (letter, i) in myGenome
        drawKey(layoutMap[i], letter)
    end

    for (name, i) in noCharKeys
        drawKey(layoutMap[i], name)
    end

    plot!(aspect_ratio=1, legend=false)
    savefig(filepath)
end

function drawKeyboardAtomic(lk, genome, filepath, layoutMap)
    lock(lk) do
        drawKeyboard(genome, filepath, layoutMap)
    end
end

function doKeypress(myFingerList, keyPress, oldFinger, oldHand, layoutMap)
    (x, y, _), (finger, _), row = layoutMap[keyPress]
    currentHand = handList[finger]
    homeX, homeY, currentX, currentY, distanceCounter, objectiveCounter = myFingerList[finger]

    # Sets other fingers back to home position
    for fingerID in 1:numFingers
        hx, hy, _, _, _, _ = myFingerList[fingerID]

        myFingerList[fingerID][3] = hx
        myFingerList[fingerID][4] = hy
    end

    myFingerList[finger][3] = currentX
    myFingerList[finger][4] = currentY

    dx, dy = x - currentX, y - currentY
    distance = sqrt(dx^2 + dy^2)

    distancePenalty = (distance + 1)^distanceEffort - 1 # This way, distanceEffort always increases even if in [0, 1]
    newDistance = distanceCounter + distance

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
    newObjective = objectiveCounter + sum(penalties)

    myFingerList[finger][3] = x
    myFingerList[finger][4] = y
    myFingerList[finger][5] = newDistance
    myFingerList[finger][6] = newObjective

    return myFingerList, finger, currentHand
end

function baselineObjectiveFunction(file, myGenome, layoutMap)
    # homeX, homeY, currentX, currentY, distanceCounter, objectiveCounter
    myFingerList = repeat([zeros(6)], 10)

    for i in 1:numFingers
        (x, y, _), (finger, _), _ = layoutMap[i]
        myFingerList[finger][1:4] = [x, y, x, y]
    end

    objective = 0
    oldFinger = 0
    oldHand = 0

    for currentCharacter in file
        # determine keypress (nothing if there is no key)
        keyPress = get(myGenome, currentCharacter, nothing)

        if isnothing(keyPress)
            continue
        end

        # do keypress
        myFingerList, oldFinger, oldHand = doKeypress(myFingerList, keyPress, oldFinger, oldHand, layoutMap)
    end

    # calculate and return objective
    objective = sum(map(x -> x[6], myFingerList))
    return objective
end

function objectiveFunction(file, genome, layoutMap, baselineScore)
    objective = (baselineObjectiveFunction(file, genome, layoutMap) / baselineScore - 1) * 100
    return objective
end

# Has probability 0.5 of changing current genome to best when starting new epoch
# Has probability e^(-delta/t) of changing current genome to a worse when not the best genome
function runSA(;
    runid,
    lk,
    rng,
    text,
    layoutMap,
    baselineGenome,
    genomeGenerator,
    temperature,
    epoch,
    coolingRate,
    num_iterations,
    verbose,
)
    mkpath("data/result$runid")

    verbose && println("Running code...")
    verbose && print("Calculating raw baseline: ")
    baselineScore = baselineObjectiveFunction(text, baselineGenome, layoutMap)
    verbose && println(baselineScore)
    verbose && println("From here everything is reletive with + % worse and - % better than this baseline \n Note that best layout is being saved as a png at each step. Kill program when satisfied.")

    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(text, currentGenome, layoutMap, baselineScore)

    bestGenome = currentGenome
    bestObjective = currentObjective

    Threads.@spawn :interactive drawKeyboardAtomic(lk, bestGenome, "data/result/$runid - first.png", layoutMap)

    baseTemp = temperature
    for iteration in 1:num_iterations
        if temperature ≤ 1
            break
        end

        # Create new genome
        newGenome = shuffleGenomeKeyMap(rng, currentGenome, fixedKeys, temperature)

        # Asess
        newObjective = objectiveFunction(text, newGenome, layoutMap, baselineScore)
        delta = newObjective - currentObjective

        sit = lpad(iteration, 6, " ")
        stemp = lpad(round(temperature, digits=2), 7, " ")
        sobj = lpad(round(bestObjective, digits=3), 8, " ")
        snobj = lpad(round(newObjective, digits=3), 8, " ")
        updateLine = "Iteration: $sit, temp: $stemp, best obj: $sobj, new obj: $snobj"

        verbose && println(updateLine)

        if delta < 0
            # If new keyboard is better (less objective is better)

            currentGenome = newGenome
            currentObjective = newObjective

            open(f -> write(f, updateLine, "\n"), "data/result/iterationScores$runid.txt", "a")

            if newObjective < bestObjective
                bestGenome = newGenome
                bestObjective = newObjective

                verbose && println("(new best, text being saved)")
                Threads.@spawn :interactive drawKeyboardAtomic(lk, bestGenome, "data/result$runid/$iteration.png", layoutMap)
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
            temperature = baseTemp * coolingRate^floor(Int64, iteration / epoch)

            # Changes genome with probability 0.5
            if rand(rng) < 0.5
                currentGenome = bestGenome
                currentObjective = bestObjective
            end
        end
    end

    Threads.@spawn :interactive drawKeyboardAtomic(lk, bestGenome, "data/result/$runid - final.png", layoutMap)

    return bestGenome, bestObjective
end

const nts = Threads.nthreads()

seed = 563622
const rng = StableRNGs.LehmerRNG(seed)
const rngs = StableRNGs.LehmerRNG.(rand(rng, 1:typemax(Int64), nts))
genomes = Dict{Any,Any}()
objectives = Dict{Any,Any}()

# TODO Use gpu in objective function
# Run julia --threads=<num threads your processor can run -1>,1 --project=. main.jl
# Example for 12 core processor, 2 threads per core, total 24 threads: julia --threads=23,1 --project=. main.jl
# Genomes are keymaps
@time begin
    @sync begin
        Threads.@threads :static for i in 1:nts
            tid = Threads.threadid()
            lk = ReentrantLock()

            # Total number of iterations will be -epoch * log(t) / log(coolingRate)
            # Compute cooling rate = 1/(t)^(epoch/i)
            t = 500
            e = 20
            niter = 1000
            cr = (1 / t)^(e / niter)

            genome, objective = runSA(
                runid=tid,
                lk=lk,
                rng=rngs[i],
                text=textdata,
                layoutMap=defaultLayoutMap,
                baselineGenome=keyMapDict,
                genomeGenerator=() -> shuffleKeyMap(rngs[i], keyMapDict, fixedKeys),
                #genomeGenerator=() -> keyMapDict,
                temperature=t,
                epoch=e,
                coolingRate=cr,
                num_iterations=100000,
                verbose=false
            )

            # TODO Use Distributed.@distributed to get results
            genomes[tid] = genome
            objectives[tid] = objective
        end
    end

    bestI, bestG, bestO = reduce(((i, g, o), (i2, g2, o2)) -> o < o2 ? (i, g, o) : (i2, g2, o2), ((i, genomes[i], objectives[i]) for i in filter(x -> haskey(genomes, x), 1:nts)))

    println("Best overall: $bestI; Score: $bestO")
    drawKeyboard(bestG, "data/result/bestOverall - $bestI.png", defaultLayoutMap)
end
