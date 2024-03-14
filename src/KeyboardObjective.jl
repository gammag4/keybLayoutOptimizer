module KeyboardObjective

include("Presets.jl")

using .Presets: dataStats, rewardArgs

export objectiveFunction

const (; fingerEffort, rowEffort, textStats) = dataStats

const (;
    effortWeighting,
    xBias,
    xMoveMultiplier,
    distanceEffort,
    doubleFingerEffort,
    singleHandEffort,
    rightHandEffort,
    keyboardSize,
) = rewardArgs

const anskbs = 1 / keyboardSize

# TODO Compute rewards in bulk for all keys and use the whole data to compute and normalize rewards in the end
# TODO Split dataset into array of smaller pieces and use it to parallelize,
# creating also another array with the transitions between keys of the end of a piece and the start of another
# Then, compute everything in bulk for the whole dataset and summarize all reward in the end from all data

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
    distance = 1 - sqrt((dx * xBias * 2)^2 + (dy * (1 - xBias) * 2)^2) * anskbs

    distancePenalty = (distance + 1)^distanceEffort - 1 # This way, distanceEffort always increases even if in [0, 1]

    # Double finger
    doubleFingerPenalty = 0
    if finger == oldFinger && distance ≈ 0
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

function objectiveFunction(text, genome, keyboardData)
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
    objective = (objectiveFunction(text, genome, keyboardData) / baselineScore - 1) * 100
    return objective
end

end
