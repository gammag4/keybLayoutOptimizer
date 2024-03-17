module FrequencyKeyboard

using Setfield: @set
using LinearAlgebra: normalize

using ..Utils: conditionalSplit, minMaxScale
using ..DrawKeyboard: computeKeyboardColorMap, drawKeyboard

export createFrequencyKeyMap, createFrequencyGenome, drawFrequencyKeyboard

# Array that maps key ids to their rewards
function createFrequencyKeyMap(dataStats, keyboardData, rewardMapArgs)
    (; rowsCPSBias, rewardWeighting) = rewardMapArgs
    (; handFingers, vertLayoutMap) = keyboardData
    (; fingersCPS, rowsCPS) = dataStats
    (; fingerMap, rowMap) = vertLayoutMap

    fingersCPSReward = minMaxScale(fingersCPS)[fingerMap]
    rowsCPSReward = minMaxScale(minMaxScale(rowsCPS) .* rowsCPSBias)[rowMap]
    leftHandReward = Float64.(handFingers[fingerMap] .== 1)

    rewards = zip(fingersCPSReward, rowsCPSReward, leftHandReward)
    return (x -> sum(x .* rewardWeighting)).(rewards) # Not scaled because the weights will scale it (weights that sum up to 1 will have it in range [0,1])
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
