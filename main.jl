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

# TODO Put code after functions and remove globals

textpath = "data/dataset.txt" # File to save/get text data
kbResultsFilePath = "data/result/iterationScores.txt" # File to save keyboard results data

# Processing data
mkpath("data/result")
processDataFolderIntoTextFile("data/raw_dataset", textpath, overwrite=false, verbose=true)

# Getting data
textdata = open(io -> read(io, String), textpath, "r")

# OLD CODE

const (; fingerEffort, rowEffort, textStats, effortWeighting) = computeStats(
    text=textdata,
    fingersCPS=[5.5, 5.9, 6.3, 6.2, 6.4, 5.3, 7.0, 6.7, 5.2, 6.2], # How many clicks can you do in a minute
    rowsCPS=[4.1, 5.0, 5.4, 4.7, 5.0], # How many clicks can you do in a minute in each row
    effortWeighting=(0.7917, 1, 0, 0.4773, 0.00), # dist, finger, row. Also had room for other weightings but removed for simplicity
)
const (; charHistogram, charFrequency, usedChars) = textStats

const distanceEffort = 1 # at 2 distance penalty is squared
const doubleFingerEffort = 1
const doubleHandEffort = -1

# Hue goes from 170 (min f) to 0 (max f)
# Saturation is the normalized frequency of each key
const normFreqToHSV = f -> HSV((1.0 - f) * 170, f * 0.7 + 0.3, 1.0)

function computeKeyboardColorMap(charFrequency)
    charFrequency = filter(((k, v),) -> !isspace(k), charFrequency)
    # Normalizes char frequency
    maxf = maximum(values(charFrequency))
    minf = minimum(values(charFrequency))

    return Dict(k => normFreqToHSV((v - minf) / (maxf - minf)) for (k, v) in charFrequency)
end

const keyboardColorMap = computeKeyboardColorMap(charFrequency)

function drawKeyboard(myGenome, id, currentLayoutMap)
    plot()

    for i in 1:46
        letter = myGenome[i]

        x, y, row, finger, home = currentLayoutMap[i]
        color = keyboardColorMap[lowercase(letter)]

        plot!([x], [y], shape=:rect, fillalpha=0.2, linecolor=nothing, color=HSVA(color, 0.5), label="", markersize=16.5, dpi=100)

        # Draws four lines using their points (that's why there is 5 xs and ys)
        plot!([x - 0.45, x + 0.45, x + 0.45, x - 0.45, x - 0.45], [y - 0.45, y - 0.45, y + 0.45, y + 0.45, y - 0.45], color=HSV(0, 0, 0), fillalpha=0.2, label="", dpi=100)

        # Draws character
        annotate!(x, y, text(letter, :black, :center, 10))
    end

    plot!(aspect_ratio=1, legend=false)
    savefig("data/result/$id.png")
end

function appendUpdates(updateLine, kbResultsFilePath)
    open(f -> write(f, updateLine, "\n"), kbResultsFilePath, "a")
end

function determineKeypress(currentCharacter)
    # setup
    keyPress = nothing

    # proceed if valid key (e.g. we dont't care about spaces now)
    if haskey(keyMapDict, currentCharacter)
        keyPress, _ = keyMapDict[currentCharacter]
    end

    # return
    return keyPress
end

function doKeypress(myFingerList, myGenome, keyPress, oldFinger, oldHand, currentLayoutMap)
    # setup
    # ~ get the key being pressed ~
    namedKey = letterList[keyPress]
    actualKey = findfirst(x -> x == namedKey, myGenome)

    # ~ get its location ~
    x, y, row, finger, home = currentLayoutMap[actualKey]
    currentHand = handList[finger]

    # loop through fingers
    for fingerID in 1:8
        # load
        homeX, homeY, currentX, currentY, distanceCounter, objectiveCounter = ntuple(i -> myFingerList[fingerID, i], Val(6))

        if fingerID == finger # move finger to key and include penalty
            # ~ distance
            distance = sqrt((x - currentX)^2 + (y - currentY)^2)

            distancePenalty = distance^distanceEffort # i.e. squared
            newDistance = distanceCounter + distance

            # ~ double finger ~
            doubleFingerPenalty = 0
            if finger != oldFinger && oldFinger != 0 && distance != 0
                doubleFingerPenalty = doubleFingerEffort
            end
            oldFinger = finger


            # ~ double hand ~
            doubleHandPenalty = 0
            if currentHand != oldHand && oldHand != 0
                doubleHandPenalty = doubleHandEffort
            end
            oldHand = currentHand

            # ~ finger
            fingerPenalty = fingerEffort[fingerID]

            # ~ row
            rowPenalty = rowEffort[row]

            # ~ combined weighting
            penalties = (distancePenalty, doubleFingerPenalty, doubleHandPenalty, fingerPenalty, rowPenalty)
            penalty = sum(penalties .* effortWeighting)
            newObjective = objectiveCounter + penalty

            # ~ save
            myFingerList[fingerID, 3] = x
            myFingerList[fingerID, 4] = y
            myFingerList[fingerID, 5] = newDistance
            myFingerList[fingerID, 6] = newObjective
        else # re-home unused finger
            myFingerList[fingerID, 3] = homeX
            myFingerList[fingerID, 4] = homeY
        end
    end

    # return
    return myFingerList, oldFinger, oldHand
end

function baselineObjectiveFunction(file, myGenome, currentLayoutMap)
    # setup
    objective = 0

    # ~ create hand ~
    myFingerList = zeros(8, 6) # (homeX, homeY, currentX, currentY, distanceCounter, objectiveCounter)

    for i in 1:46
        x, y, _, finger, home = currentLayoutMap[i]

        if home == 1.0
            myFingerList[finger, 1:4] = [x, y, x, y]
        end
    end

    # load text
    oldFinger = 0
    oldHand = 0

    for currentCharacter in file
        # determine keypress
        keyPress = determineKeypress(currentCharacter)

        # do keypress
        if keyPress !== nothing
            myFingerList, oldFinger, oldHand = doKeypress(myFingerList, myGenome, keyPress, oldFinger, oldHand,
                currentLayoutMap)
        end
    end

    # calculate and return objective
    objective = sum(myFingerList[:, 6])
    return objective
end

function objectiveFunction(file, myGenome, currentLayoutMap, baselineScore)
    objective = (baselineObjectiveFunction(file, myGenome, currentLayoutMap) / baselineScore - 1) * 100
    return objective
end

function shuffleGenome(rng, currentGenome, temperature)
    # setup
    noSwitches = Int(maximum([2, minimum([floor(temperature / 100), numKeys])]))

    # positions of switched letterList
    switchedPositions = randperm(rng, numKeys)[1:noSwitches]
    newPositions = shuffle(rng, copy(switchedPositions))

    # create new genome by shuffeling
    newGenome = copy(currentGenome)
    for i in 1:noSwitches
        og = switchedPositions[i]
        ne = newPositions[i]

        newGenome[og] = currentGenome[ne]
    end

    # return
    return newGenome

end

function runSA(;
    rng,
    text,
    layoutMap,
    baselineLayout,
    genomeGenerator,
    kbResultsFilePath,
    temperature,
    epoch,
    coolingRate,
    num_iterations,
    save_current_best,
    verbose,
)
    currentLayoutMap = layoutMap

    verbose && println("Running code...")
    # baseline
    verbose && print("Calculating raw baseline: ")
    baselineScore = baselineObjectiveFunction(text, baselineLayout, currentLayoutMap) # ok me fixeded
    verbose && println(baselineScore)

    verbose && println("From here everything is reletive with + % worse and - % better than this baseline \n Note that best layout is being saved as a png at each step. Kill program when satisfied.")

    verbose && println("Temperature \t Best Score \t New Score")


    # setup
    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(text, currentGenome, currentLayoutMap, baselineScore)

    bestGenome = currentGenome
    bestObjective = currentObjective

    drawKeyboard(bestGenome, 0, currentLayoutMap)

    # run SA
    staticCount = 0.0
    iteration = 0
    while iteration <= num_iterations && temperature > 1.0
        iteration += 1
        # ~ create new genome ~
        newGenome = shuffleGenome(rng, currentGenome, 2)

        # ~ asess ~
        newObjective = objectiveFunction(text, newGenome, currentLayoutMap, baselineScore)
        delta = newObjective - currentObjective

        verbose && println(round(temperature, digits=2), "\t", round(bestObjective, digits=2), "\t", round(newObjective, digits=2))


        if delta < 0
            currentGenome = copy(newGenome)
            currentObjective = newObjective

            updateLine = string(round(temperature, digits=2), ", ", iteration, ", ", round(bestObjective, digits=5), ", ", round(newObjective, digits=5))
            appendUpdates(updateLine, kbResultsFilePath)

            if newObjective < bestObjective
                bestGenome = newGenome
                bestObjective = newObjective

                #staticCount = 0.0

                if save_current_best === :plot
                    verbose && println("(new best, png being saved)")
                    drawKeyboard(bestGenome, iteration, currentLayoutMap)
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

    # save
    drawKeyboard(bestGenome, "final", currentLayoutMap)

    # return
    return bestGenome

end

seed = 123456
const rng = StableRNGs.LehmerRNG(seed)

# TODO Run multiple times to create many different keyboard layouts, compare them and get the best
@time runSA(
    rng=rng,
    text=textdata,
    layoutMap=defaultLayoutMap,
    baselineLayout=defaultGenome,
    genomeGenerator=() -> shuffle(rng, letterList),
    #genomeGenerator=() -> defaultGenome,
    kbResultsFilePath=kbResultsFilePath,
    temperature=500, # TODO 1000
    epoch=20,
    coolingRate=0.99, # TODO 0.9999
    num_iterations=25000, # TODO 500000
    save_current_best=:plot,
    verbose=true
)
