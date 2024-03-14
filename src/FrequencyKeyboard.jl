module FrequencyKeyboard

using Setfield: @set
using LinearAlgebra: normalize

include("Presets.jl")
include("Utils.jl")
include("DrawKeyboard.jl")

using .Presets: frequencyKeyboardArgs
using .Utils: conditionalSplit
using .DrawKeyboard: computeKeyboardColorMap, drawKeyboard

export createFrequencyGenome, drawFrequencyKeyboard

const (; xBias, distanceBias, leftHandBias, rowCPSBias, keyboardSize) = frequencyKeyboardArgs
const anskbs = 1 / keyboardSize

# TODO Optimize code here: Change <something> in <list> to dictionary haskey or something

function objective(key, dataStats, keyboardData)
    (; fingersCPS, rowsCPS) = dataStats
    (; fingersHome, layoutMap, handFingers) = keyboardData

    (x, y, _), (finger, _), row = layoutMap[key]
    home = fingersHome[finger]
    (hx, hy, _), _, _ = layoutMap[home]

    # Distance penalty
    dx, dy = x - hx, y - hy
    distancePenalty = 1 - sqrt((dx * xBias * 2)^2 + (dy * (1 - xBias) * 2)^2) * anskbs

    # Finger and row reward
    # TODO change to bounds [0,1] and compute outside function
    fingerReward = normalize(fingersCPS)[finger] * normalize(rowsCPS .* rowCPSBias)[row]

    # 1 for right hand, > 1 for left hand
    leftHandReward = 1 - leftHandBias + (2 - handFingers[finger]) * (2 * leftHandBias - 1)

    reward = (fingerReward * (1 - distanceBias) + distancePenalty * distanceBias) * leftHandReward
    return reward
end

# Maps key ids to their rewards
function createFrequencyKeyMap(dataStats, keyboardData)
    (; layoutMap) = keyboardData
    return Dict(k => objective(k, dataStats, keyboardData) for k in keys(layoutMap))
end

getSorted(keyMap) = map(((c, f),) -> c, sort(by=((c, f),) -> f, collect(keyMap)))

# Maps chars to their frequencies
function getFrequencyKeyMap(keyMap, charFrequency)
    keyMap1, keyMap2 = conditionalSplit(((k, v),) -> k in keys(charFrequency), keyMap)
    keyMap1 = Dict(map(((k, v),) -> k => charFrequency[k], collect(keyMap1)))
    keyMap2 = Dict(map(((k, v),) -> k => 0, collect(keyMap2)))
    return merge(keyMap1, keyMap2)
end

function createFrequencyGenome(dataStats, keyboardData)
    (; textStats) = dataStats
    (; charFrequency) = textStats
    (; keyMap, getFixedMovableKeyMaps, fixedKeys, fixedKeyMap) = keyboardData
    revFixedKeys = (keyMap[c] for c in fixedKeys) # Keys instead of chars

    rewardKeyMap = createFrequencyKeyMap(dataStats, keyboardData) # Keys to rewards from entire layout
    freqKeyMap, _ = conditionalSplit(((k, v),) -> k in values(keyMap), rewardKeyMap)
    _, movableFreqKeyMap = conditionalSplit(((k, v),) -> k in revFixedKeys, freqKeyMap)

    charFrequency = getFrequencyKeyMap(keyMap, charFrequency) # Chars to real frequencies
    _, movableCharFrequency = getFixedMovableKeyMaps(charFrequency)

    mchars = getSorted(movableCharFrequency)
    mkeys = getSorted(movableFreqKeyMap)

    kmap = merge(Dict(zip(mchars, mkeys)), fixedKeyMap) # char => key
    freqkmap = Dict(c => freqKeyMap[k] for (c, k) in collect(kmap)) # char => f
    return kmap, freqkmap
end

function drawFrequencyKeyboard(filepath, genome, freqKeyMap, keyboardData; useFrequencyColorMap=false)
    kbData = keyboardData
    if useFrequencyColorMap == true
        kbData = @set keyboardData.keyboardColorMap = computeKeyboardColorMap(freqKeyMap)
    end

    drawKeyboard(genome, filepath, kbData)
end

end
