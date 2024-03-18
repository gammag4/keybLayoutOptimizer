module SimulatedAnnealing

using Base.Threads: @spawn, @threads, threadid, nthreads
using Base.Iterators: countfrom
using Printf: @sprintf
using Random: rand

using ..Genome: shuffleGenomeKeyMap
using ..DrawKeyboard: drawKeyboard
using ..KeyboardObjective: objectiveFunction
using ..Utils: dictToArray

export chooseSA!

# Has probability 0.5 of changing current genome to best when starting new epoch
# Has probability e^(-delta/t) of changing current genome to a worse when not the best genome
function runSA(;
    keyboardId,
    rng,
    genomeGenerator,
    lk=nothing,
    saArgs
)
    (;
        baselineScore,
        computationArgs,
        rewardArgs,
        keyboardData,
        algorithmArgs,
        dataPaths,
        findWorst
    ) = saArgs
    (; fixedKeys) = keyboardData
    (; t, e, nIter, tShuffleMultiplier) = algorithmArgs
    (; startResultsPath, endResultsPath) = dataPaths

    baselineScale = max(abs(baselineScore), 1)

    compare = findWorst ? (>) : (<)

    coolingRate = (1 / t)^(e / nIter)

    currentGenome = genomeGenerator()
    currentObjective = objectiveFunction(currentGenome, computationArgs, rewardArgs, baselineScale)

    bestGenome, bestObjective = currentGenome, currentObjective

    drawKeyboard(bestGenome, joinpath(startResultsPath, "$keyboardId.png"), keyboardData, lk)

    baseT = t
    try
        for iteration in countfrom(1)
            t ≤ 1 && break

            # Create new genome
            newGenome = shuffleGenomeKeyMap(rng, currentGenome, fixedKeys, floor(Int, t * tShuffleMultiplier))

            # Asess
            newObjective = objectiveFunction(newGenome, computationArgs, rewardArgs, baselineScale)
            delta = newObjective - currentObjective

            if iteration % 1000 == 1
                skid = lpad(keyboardId, 3, " ")
                sit = lpad(iteration, 6, " ")
                stemp = lpad((@sprintf "%.2f" t), 7, " ")
                sobj = lpad((@sprintf "%.3f" bestObjective), 8, " ")
                snobj = lpad((@sprintf "%.3f" newObjective), 8, " ")
                updateLine = "Keyboard: $skid, Iteration: $sit, temp: $stemp, best obj: $sobj, new obj: $snobj"

                println(updateLine)
            end

            # If new keyboard is better (less objective is better)
            if compare(delta, 0)
                currentGenome, currentObjective = newGenome, newObjective

                if compare(newObjective, bestObjective)
                    bestGenome, bestObjective = newGenome, newObjective
                end

                # Changes genome with probability e^(-delta/t)
            elseif exp(-delta / t) > rand(rng)
                currentGenome, currentObjective = newGenome, newObjective
            end

            # Starting new epoch
            if iteration > e && (iteration % e == 1)
                t = baseT * coolingRate^floor(Int, iteration / e)

                # Changes genome with probability 0.5
                if rand(rng) < 0.5
                    currentGenome, currentObjective = bestGenome, bestObjective
                end
            end
        end
    catch e
        if !(e isa InterruptException)
            rethrow(e)
        end
    end

    drawKeyboard(bestGenome, joinpath(endResultsPath, "$keyboardId.png"), keyboardData, lk)

    return bestGenome, bestObjective
end

# GPU
function chooseSA!(numKeyboards, genomes, objectives, rngs, generators, saArgs, useGPU::Val{true})
    for i in 1:numKeyboards
        genomes[i], objectives[i] = runSA(
            keyboardId=i,
            rng=rngs[i],
            genomeGenerator=generators[i],
            saArgs=saArgs
        )
    end
end

# CPU
function chooseSA!(numKeyboards, genomes, objectives, rngs, generators, saArgs, useGPU::Val{false})
    lk = ReentrantLock()
    nts = nthreads()

    for j in 1:nts:numKeyboards
        @sync begin
            @threads for k in 1:min(nts, numKeyboards - j + 1)
                i = j + k - 1
                genomes[i], objectives[i] = runSA(
                    keyboardId=i,
                    rng=rngs[i],
                    genomeGenerator=generators[i],
                    lk=lk,
                    saArgs=saArgs
                )
            end
        end
    end
end

end
