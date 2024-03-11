module Genome

using Random
using Utils

# Num keys should not exceed length(keyMap) - length(fixedKeys)
function shuffleKeyMap(rng, keyMap, fixedKeys; numKeys=nothing)
    fkm, skm = conditionalSplit(((k, v),) -> k in fixedKeys, keyMap)
    ks, vs = collect(keys(skm)), collect(values(skm))

    shuffledKeys = randperm(rng, length(vs))
    shuffledKeys = isnothing(numKeys) ? shuffledKeys : shuffledKeys[1:numKeys]
    vs[shuffledKeys] = shuffle(rng, copy(vs[shuffledKeys])) # Permutates keys

    d = Dict(zip(ks, vs))
    merge!(d, fkm)
    return d
end

export shuffleKeyMap

end
