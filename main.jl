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
kbResultsFilePath = "data/result/iterationScores.txt" # File to save keyboard results data

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

function drawKeyboard(myGenome, id, layoutMap)
    plot(axis=([], false))

    for (letter, i) in myGenome
        drawKey(layoutMap[i], letter)
    end

    for (name, i) in noCharKeys
        drawKey(layoutMap[i], name)
    end

    plot!(aspect_ratio=1, legend=false)
    savefig("data/result/$id.png")
end

appendUpdates(updateLine, kbResultsFilePath) = open(f -> write(f, updateLine, "\n"), kbResultsFilePath, "a")

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

# TODO Move to Genome
function shuffleGenomeKeyMap(rng, genomeArray, fixedKeys, temperature)
    noSwitches = Int(maximum([2, minimum([floor(temperature / 100), numFixedKeys])]))

    return shuffleKeyMap(rng, genomeArray, fixedKeys; numKeys=noSwitches)
end

function runSA(;
    rng,
    text,
    layoutMap,
    baselineGenome,
    genomeGenerator,
    kbResultsFilePath,
    temperature,
    epoch,
    coolingRate,
    num_iterations,
    save_current_best,
    verbose,
)
    verbose && println("Running code...")
    verbose && print("Calculating raw baseline: ")
    baselineScore = baselineObjectiveFunction(text, baselineGenome, layoutMap)
    verbose && println(baselineScore)
    verbose && println("From here everything is reletive with + % worse and - % better than this baseline \n Note that best layout is being saved as a png at each step. Kill program when satisfied.")

    verbose && println("Temperature \t Best Score \t New Score")

    # setup
    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(text, currentGenome, layoutMap, baselineScore)

    bestGenome = currentGenome
    bestObjective = currentObjective

    drawKeyboard(bestGenome, 0, layoutMap)

    # run SA
    staticCount = 0.0
    iteration = 0
    while iteration <= num_iterations && temperature > 1.0
        iteration += 1
        # ~ create new genome ~
        newGenome = shuffleGenomeKeyMap(rng, currentGenome, fixedKeys, 2)

        # ~ asess ~
        newObjective = objectiveFunction(text, newGenome, layoutMap, baselineScore)
        delta = newObjective - currentObjective

        verbose && println(round(temperature, digits=2), "\t", round(bestObjective, digits=2), "\t", round(newObjective, digits=2))


        if delta < 0
            currentGenome = deepcopy(newGenome)
            currentObjective = newObjective

            updateLine = string(round(temperature, digits=2), ", ", iteration, ", ", round(bestObjective, digits=5), ", ", round(newObjective, digits=5))
            appendUpdates(updateLine, kbResultsFilePath)

            if newObjective < bestObjective
                bestGenome = newGenome
                bestObjective = newObjective

                #staticCount = 0.0

                if save_current_best === :plot
                    verbose && println("(new best, png being saved)")
                    drawKeyboard(bestGenome, iteration, layoutMap)
                else
                    verbose && println("(new best, text being saved)")
                    open("bestGenomes.txt", "a") do io
                        print(io, iteration, ":")
                        for c in bestGenome
                            print(io, c)
                        end
                        println(io)
                    end
                end
            end
        elseif exp(-delta / temperature) > rand(rng)
            #print(" *")
            currentGenome = newGenome
            currentObjective = newObjective
        end

        #print("\n")


        staticCount += 1.0

        if staticCount > epoch
            staticCount = 0.0
            temperature = temperature * coolingRate

            if rand(rng) < 0.5
                currentGenome = bestGenome
                currentObjective = bestObjective
            end
        end
    end

    drawKeyboard(bestGenome, "final", layoutMap)

    return bestGenome

end

seed = 123456
const rng = StableRNGs.LehmerRNG(seed)

# TODO Run multiple times to create many different keyboard layouts, compare them and get the best
# Genomes are keymaps
@time runSA(
    rng=rng,
    text=textdata,
    layoutMap=defaultLayoutMap,
    baselineGenome=keyMapDict,
    genomeGenerator=() -> shuffleKeyMap(rng, keyMapDict, fixedKeys),
    #genomeGenerator=() -> keyMapDict,
    kbResultsFilePath=kbResultsFilePath,
    temperature=500, # TODO 1000
    epoch=20,
    coolingRate=0.99, # TODO 0.9999
    num_iterations=25000, # TODO 500000
    save_current_best=:plot,
    verbose=true
)
