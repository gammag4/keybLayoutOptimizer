module SimulatedAnnealing

using Base.Threads: @spawn, @threads, threadid, nthreads
using Base.Iterators: countfrom
using Printf: @sprintf
using Random: rand

using ..Genome: shuffleGenomeKeyMap
using ..KeyboardObjective: objectiveFunction
using ..Utils: dictToArray

export runSA

# Has probability 0.5 of changing current genome to best when starting new epoch
# Has probability e^(-delta/t) of changing current genome to a worse when not the best genome
function sa(;
    keyboardId,
    saArgs
)
    (;
        computationArgs,
        rewardArgs,
        algorithmArgs,
        keyboardData,
        compareGenomes,
        rngs,
        genomeGenerator,
        keyboardUpdatesArgs
    ) = saArgs
    (; t, e, nIter, tShuffleMultiplier) = algorithmArgs
    (; fixedKeys) = keyboardData
    (; viewKeyboardUpdates, nIterBeforeNextUpdate) = keyboardUpdatesArgs

    rng = rngs[keyboardId]
    @inline generator() = genomeGenerator(keyboardId, rng)

    coolingRate = (1 / t)^(e / nIter)

    bestGenome = currentGenome = startGenome = generator()
    bestObjective = currentObjective = startObjective = objectiveFunction(currentGenome, computationArgs, rewardArgs)

    baseT = t
    try
        for iteration in countfrom(1)
            t ≤ 1 && break

            viewKeyboardUpdates && iteration % nIterBeforeNextUpdate == 1 && drawKeyboard(currentGenome, keyboardData)

            # Create new genome
            newGenome = shuffleGenomeKeyMap(rng, currentGenome, fixedKeys, floor(Int, t * tShuffleMultiplier))

            # Asess
            newObjective = objectiveFunction(newGenome, computationArgs, rewardArgs)
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
            if compareGenomes(newObjective, currentObjective)
                currentGenome, currentObjective = newGenome, newObjective

                if compareGenomes(newObjective, bestObjective)
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

    return (startGenome, startObjective), (bestGenome, bestObjective)
end

# GPU
function chooseSA!(startGenomes, startObjectives, genomes, objectives, numKeyboards, saArgs, useGPU::Val{true})
    for i in 1:numKeyboards
        (startGenomes[i], startObjectives[i]), (genomes[i], objectives[i]) = sa(
            keyboardId=i,
            saArgs=saArgs
        )
    end
end

# CPU
function chooseSA!(startGenomes, startObjectives, genomes, objectives, numKeyboards, saArgs, useGPU::Val{false})
    nts = nthreads()

    for j in 1:nts:numKeyboards
        @sync begin
            @threads for k in 1:min(nts, numKeyboards - j + 1)
                i = j + k - 1
                (startGenomes[i], startObjectives[i]), (genomes[i], objectives[i]) = sa(
                    keyboardId=i,
                    saArgs=saArgs
                )
            end
        end
    end
end

genomesToArray(numKeyboards, genomes, objectives) = filter(((i, a, b),) -> !isnothing(a) && !isnothing(b), [(i, get(genomes, i, nothing), get(objectives, i, nothing)) for i in 1:numKeyboards])

function runSA(saArgs, useGPU)
    (; algorithmArgs, compareGenomes) = saArgs
    (; numKeyboards) = algorithmArgs

    # These dicts are used so that we can cancel the operation in the middle and still get the keyboards that were computed up to that time
    genomes = Dict{Any,Any}()
    objectives = Dict{Any,Any}()
    startGenomes = Dict{Any,Any}()
    startObjectives = Dict{Any,Any}()

    try
        # TODO Use Distributed.@distributed to get endResultsPath
        chooseSA!(startGenomes, startObjectives, genomes, objectives, numKeyboards, saArgs, Val(useGPU))
    catch e
        if !(e isa InterruptException)
            rethrow(e)
        end
    end

    startGenomes = genomesToArray(numKeyboards, startGenomes, startObjectives)
    endGenomes = genomesToArray(numKeyboards, genomes, objectives)
    bestGenome = reduce(((i, g, o), (i2, g2, o2)) -> compareGenomes(o, o2) ? (i, g, o) : (i2, g2, o2), endGenomes)

    return startGenomes, endGenomes, bestGenome
end

end
