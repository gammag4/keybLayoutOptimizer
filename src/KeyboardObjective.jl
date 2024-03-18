module KeyboardObjective

using CUDA

using ..Types: RewardArgs, LayoutKey, CPUArgs, GPUArgs
using ..Utils: dictToArray, binrTupleToModTuple

export objectiveFunction

# TODO Compute rewards in bulk for all keys and use the whole data to compute, normalize rewards and compute all reward

# Higher is worse
function threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, rewardMap)
    (;
        effortWeighting,
        yScale,
        distGrowthRate
    ) = rewardArgs

    # TODO create penalty for mod keys distance
    (char1, t1) = text[i]
    (char2, t2) = text[i+1]
    _, _, _ = (t1 & 4) >> 2, (t1 & 2) >> 1, t1 & 1
    _, _, _ = (t2 & 4) >> 2, (t2 & 2) >> 1, t2 & 1
    key1 = genome[Int(char1)]
    key2 = genome[Int(char2)]
    (x1, y1, _, _), (finger1, _), _ = layoutMap[key1]
    (x2, y2, _, _), (finger2, home), _ = layoutMap[key2]
    (hx, hy, _, _), _, _ = layoutMap[home]
    hand1 = handFingers[finger1]
    hand2 = handFingers[finger2]

    sameFinger = finger1 == finger2 # Used same finger as previous

    # If same finger, uses old position, else, uses home position (assuming fingers go home when you use another finger)
    x1, y1 = sameFinger * x1 + (!sameFinger) * hx, sameFinger * y1 + (!sameFinger) * hy

    # x will always have scale one, y will be scaled by yScale
    dx, dy = x2 - x1, y2 - y1
    distance = (1 + sqrt(dx^2 + (dy * yScale)^2))^distGrowthRate - 1

    doubleFingerPenalty = sameFinger
    singleHandPenalty = hand1 == hand2 # Used same hand as previous
    distancePenalty = distance
    rewardMapPenalty = -rewardMap[key2] # Negative, since higher is worse here and in reward map, higher is better

    # TODO Put in output array instead of summing here
    # Combined weighting
    penalties = (doubleFingerPenalty, singleHandPenalty, distancePenalty, rewardMapPenalty) .* effortWeighting
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
    # (; nonNeighborsEffort) = rewardArgs

    # # Checks for [], <> and () being neighbors
    # checkNeighbors = checkNeighborsFunc(vcat(["[]", ",."], ["$i$(i+1)" for i in 0:8]), genome)
    # objective = objective * (1 + checkNeighbors * nonNeighborsEffort)

    return objective
end

end
