module KeyboardObjective

using CUDA

using ..Types: RewardArgs, LayoutKey, CPUArgs, GPUArgs
using ..Utils: dictToArray

export objectiveFunction

# TODO Compute rewards in bulk for all keys and use the whole data to compute, normalize rewards and compute all reward

function threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, rewardMap)
    (;
        effortWeighting,
        doubleFingerEffort,
        singleHandEffort,
        rewardMapEffort,
    ) = rewardArgs

    char1 = text[i]
    char2 = text[i+1]
    key1 = genome[Int(char1)]
    key2 = genome[Int(char2)]
    _, (finger1, _), _ = layoutMap[key1]
    _, (finger2, _), _ = layoutMap[key2]
    hand1 = handFingers[finger1]
    hand2 = handFingers[finger2]

    # Old code would also consider distance to prevent counting when pressing same key as before,
    # but this doesn't change the result of the algorithm, since it will just increase the objective of all genomes,
    # hence, not changing the ordering of the set of possible genomes, so it is useless computation
    sameFinger = finger1 == finger2 # Used same finger as previous
    sameHand = hand1 == hand2 # Used same hand as previous

    doubleFingerPenalty = sameFinger * doubleFingerEffort
    singleHandPenalty = sameHand * singleHandEffort
    rewardMapPenalty = rewardMap[key2] * rewardMapEffort

    # TODO Put in output array instead of summing here
    # Combined weighting
    penalties = (doubleFingerPenalty, singleHandPenalty, rewardMapPenalty) .* effortWeighting
    return sum(penalties)
end

function cudaCall!(out, genome, rewardArgs, text, layoutMap, handFingers, rewardMap)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:(length(text)-1)
        out[i] = @inbounds threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, rewardMap)
    end

    return
end

function cpuCall(genome, rewardArgs, text, layoutMap, handFingers, rewardMap)
    objective = 0.0
    for i in 1:(length(text)-1)
        objective += threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, rewardMap)
    end

    return objective
end

# GPU
function computeRawObjective(genome, computationArgs::GPUArgs, rewardArgs)
    (; numThreadsInBlock, text, layoutMap, handFingers, rewardMap) = computationArgs

    out = CuArray{Float64}(undef, length(text) - 1)

    blocks = ceil(Int, length(out) / numThreadsInBlock)
    # Last character is not considered, since there is no next to move to
    @cuda threads = numThreadsInBlock blocks = blocks cudaCall!(out, CuArray{Int}(dictToArray(genome)), rewardArgs, text, layoutMap, handFingers, rewardMap)

    # calculate and return objective
    return sum(out)
end

# CPU
function computeRawObjective(genome, computationArgs::CPUArgs, rewardArgs)
    (; text, layoutMap, handFingers, rewardMap) = computationArgs
    return cpuCall(dictToArray(genome), rewardArgs, text, layoutMap, handFingers, rewardMap)
end

checkNeighbor(char1, char2, genome) = abs(genome[char1] - genome[char2] + 1)

checkNeighborsFunc(chars, genome) = sum((checkNeighbor(i, j, genome) for (i, j) in chars)) / length(chars)

function objectiveFunction(genome, computationArgs, rewardArgs)
    # Chooses cpu or gpu based on type of args used
    objective = computeRawObjective(genome, computationArgs, rewardArgs)

    # TODO Move
    # (; nonNeighborsEffort, ansKbs) = rewardArgs

    # # Checks for [], <> and () being neighbors
    # checkNeighbors = checkNeighborsFunc(vcat(["[]", ",."], ["$i$(i+1)" for i in 0:8]), genome)
    # objective = objective * (1 + checkNeighbors * nonNeighborsEffort / ansKbs)

    return objective
end

function objectiveFunction(genome, computationArgs, rewardArgs, baselineScore)
    objective = (objectiveFunction(genome, computationArgs, rewardArgs) / baselineScore - 1) * 100
    return objective
end

end
