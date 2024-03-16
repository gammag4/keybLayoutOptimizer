module KeyboardObjective

using CUDA

using ..Types: RewardArgs, LayoutKey, CPUArgs, GPUArgs
using ..Utils: dictToArray

export objectiveFunction

# TODO Compute rewards in bulk for all keys and use the whole data to compute, normalize rewards and compute all reward

function threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)
    (;
        effortWeighting,
        xBias,
        distanceEffort,
        doubleFingerEffort,
        singleHandEffort,
        rightHandEffort,
        ansKbs,
    ) = rewardArgs

    # Thread memory:
    # layoutMap, handfingers, text[i:i + 1], fingerEffort, rowEffort, genome, xBias, ansKbs

    char1 = text[i]
    char2 = text[i+1]
    key1 = genome[Int(char1)]
    key2 = genome[Int(char2)]
    (x1, y1, _, _), (finger1, _), _ = layoutMap[key1]
    (x2, y2, _, _), (finger2, home), row = layoutMap[key2]
    (homeX, homeY, _, _), _, _ = layoutMap[home]
    hand1 = handFingers[finger1]
    hand2 = handFingers[finger2]

    # Old code would also consider distance to prevent counting when pressing same key as before,
    # but this doesn't change the result of the algorithm, since it will just increase the objective of all genomes,
    # hence, not changing the ordering of the set of possible genomes, so it is useless computation
    sameFinger = finger1 == finger2 # Used same finger as previous
    sameHand = hand1 == hand2 # Used same hand as previous
    rightHand = hand2 == 2 # Used right hand

    # If same finger, uses old position, else, uses home position (assuming fingers go home when you use another finger)
    x1, y1 = sameFinger * x1 + (!sameFinger) * homeX, sameFinger * y1 + (!sameFinger) * homeY

    # Distance (normalized by keyboard size)
    dx, dy = x2 - x1, y2 - y1
    distance = sqrt((dx * xBias * 2)^2 + (dy * (1 - xBias) * 2)^2) * ansKbs

    distancePenalty = (distance + 1)^distanceEffort - 1 # This way, distanceEffort always increases even if in [0, 1]
    doubleFingerPenalty = sameFinger * doubleFingerEffort
    singleHandPenalty = sameHand * singleHandEffort
    rightHandPenalty = rightHand * rightHandEffort
    fingerPenalty = fingerEffort[finger2]
    rowPenalty = rowEffort[row]

    # TODO Put in output array instead of summing here
    # Combined weighting
    penalties = (distancePenalty, doubleFingerPenalty, singleHandPenalty, rightHandPenalty, fingerPenalty, rowPenalty) .* effortWeighting
    return sum(penalties)
end

function cudaCall!(out, genome, rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:(length(text)-1)
        out[i] = @inbounds threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)
    end

    return
end

function cpuCall(genome, rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)
    objective = 0.0
    for i in 1:(length(text)-1)
        objective += threadExec(i, genome, rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)
    end

    return objective
end

# GPU
function computeRawObjective(genome, computationArgs::GPUArgs, rewardArgs)
    (; numThreadsInBlock, text, layoutMap, handFingers, fingerEffort, rowEffort) = computationArgs

    out = CuArray{Float64}(undef, length(text) - 1)

    blocks = ceil(Int, length(out) / numThreadsInBlock)
    # Last character is not considered, since there is no next to move to
    @cuda threads = numThreadsInBlock blocks = blocks cudaCall!(out, CuArray{Int}(dictToArray(genome)), rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)

    # calculate and return objective
    return sum(out)
end

# CPU
function computeRawObjective(genome, computationArgs::CPUArgs, rewardArgs)
    (; text, layoutMap, handFingers, fingerEffort, rowEffort) = computationArgs
    return cpuCall(dictToArray(genome), rewardArgs, text, layoutMap, handFingers, fingerEffort, rowEffort)
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
