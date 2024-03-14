module KeyboardObjective

using CUDA

include("Presets.jl")
include("Utils.jl")

using .Presets: dataStats, rewardArgs, gpuArgs, LayoutKey
using .Utils: dictToArray

export objectiveFunction

const (; fingerEffort, rowEffort) = dataStats

const (;
    effortWeighting,
    xBias,
    distanceEffort,
    doubleFingerEffort,
    singleHandEffort,
    rightHandEffort,
    keyboardSize,
) = rewardArgs

const ansKbs = 1 / keyboardSize

# TODO Compute rewards in bulk for all keys and use the whole data to compute, normalize rewards and compute all reward

function threadExec!(i::Int, text::CUDA.CuDeviceVector{Char,1}, out::CUDA.CuDeviceVector{Float64,1}, layoutMap::CUDA.CuDeviceVector{LayoutKey,1}, handFingers::CUDA.CuDeviceVector{Int,1}, genome::CUDA.CuDeviceVector{Int,1}, fingerEffort::CUDA.CuDeviceVector{Float64,1}, rowEffort::CUDA.CuDeviceVector{Float64,1})
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
    distance = 1 - sqrt((dx * xBias * 2)^2 + (dy * (1 - xBias) * 2)^2) * ansKbs

    distancePenalty = (distance + 1)^distanceEffort - 1 # This way, distanceEffort always increases even if in [0, 1]
    doubleFingerPenalty = sameFinger * doubleFingerEffort
    singleHandPenalty = singleHandEffort
    rightHandPenalty = rightHandEffort
    fingerPenalty = fingerEffort[finger2]
    rowPenalty = rowEffort[row]

    # TODO Put in output array instead of summing here
    # Combined weighting
    penalties = (distancePenalty, doubleFingerPenalty, singleHandPenalty, rightHandPenalty, fingerPenalty, rowPenalty) .* effortWeighting
    out[i] = sum(penalties)

    return
end

function cudaCall!(text::CUDA.CuDeviceVector{Char,1}, out::CUDA.CuDeviceVector{Float64,1}, layoutMap::CUDA.CuDeviceVector{LayoutKey,1}, handFingers::CUDA.CuDeviceVector{Int,1}, genome::CUDA.CuDeviceVector{Int,1}, fingerEffort::CUDA.CuDeviceVector{Float64,1}, rowEffort::CUDA.CuDeviceVector{Float64,1})
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i in index:stride:(length(text)-1)
        @inbounds threadExec!(i, text, out, layoutMap, handFingers, genome, fingerEffort, rowEffort)
    end

    return
end

function objectiveFunction(genome, keyboardData, gpuArgs)
    (; handFingers) = keyboardData
    (; numThreadsInBlock, text, layoutMap, handFingers, fingerEffort, rowEffort) = gpuArgs

    out = CuArray{Float64}(undef, length(text) - 1)

    blocks = ceil(Int, length(out) / numThreadsInBlock)
    # Last character is not considered, since there is no next to move to
    @cuda threads = numThreadsInBlock blocks = blocks cudaCall!(text, out, layoutMap, handFingers, CuArray{Int}(dictToArray(genome)), fingerEffort, rowEffort)

    # calculate and return objective
    objective = sum(out)

    return objective
end

function objectiveFunction(genome, keyboardData, gpuArgs, baselineScore)
    objective = (objectiveFunction(genome, keyboardData, gpuArgs) / baselineScore - 1) * 100
    return objective
end

end
