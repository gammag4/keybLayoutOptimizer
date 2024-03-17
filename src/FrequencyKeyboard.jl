module FrequencyKeyboard

using Setfield: @set
using LinearAlgebra: normalize

using ..Utils: conditionalSplit, minMaxScale
using ..DrawKeyboard: computeKeyboardColorMap, drawKeyboard

export createFrequencyKeyMap, createFrequencyGenome, drawFrequencyKeyboard

# x will always have scale one, y will be scaled by yScale
biasedDistance(x, y, yScale) = sqrt(x^2 + (y * yScale)^2)

# Array that maps key ids to their rewards
function createFrequencyKeyMap(dataStats, keyboardData, frequencyRewardArgs)
    (; xBias, rowsCPSBias, effortWeighting, ansKbs) = frequencyRewardArgs
    (; handFingers, vertLayoutMap) = keyboardData
    (; fingersCPS, rowsCPS) = dataStats
    (; xMap, yMap, hxMap, hyMap, fingerMap, rowMap) = vertLayoutMap

    dx, dy = xMap .- hxMap, yMap .- hyMap
    distanceReward = sqrt.((dx .* (xBias * 2)) .^ 2 + (dy .* ((1 - xBias) * 2)) .^ 2) .* ansKbs
    distanceReward = (x -> 2^1.5 - (1 + x)^1.5 / (2^1.5)).(distanceReward)

    fingersCPSReward = minMaxScale(fingersCPS)[fingerMap]
    rowsCPSReward = minMaxScale(minMaxScale(rowsCPS) .* rowsCPSBias)[rowMap]
    leftHandReward = Float64.(handFingers[fingerMap] .== 1)

    rewards = zip(fingersCPSReward, rowsCPSReward, leftHandReward, distanceReward)
    return minMaxScale((x -> sum(x .* effortWeighting)).(rewards))
end

getSorted(keyMap) = map(((c, f),) -> c, sort(by=((c, f),) -> f, collect(keyMap)))

# Maps chars to their frequencies
function getFrequencyKeyMap(keyMap, charFrequency)
    cfks = Set(keys(charFrequency))
    keyMap1, keyMap2 = conditionalSplit(((k, v),) -> k in cfks, keyMap)
    keyMap1 = Dict(map(((k, v),) -> k => charFrequency[k], collect(keyMap1)))
    keyMap2 = Dict(map(((k, v),) -> k => 0, collect(keyMap2)))
    return merge(keyMap1, keyMap2)
end

function createFrequencyGenome(dataStats, keyboardData, rewardKeyMap)
    (; textStats) = dataStats
    (; charFrequency) = textStats
    (; keyMap, getFixedMovableKeyMaps, fixedKeys, fixedKeyMap) = keyboardData
    revFixedKeys = Set((keyMap[c] for c in fixedKeys)) # Keys instead of chars

    svkm = Set(values(keyMap))
    freqKeyMap, _ = conditionalSplit(((k, v),) -> k in svkm, Dict(enumerate(rewardKeyMap)))
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
