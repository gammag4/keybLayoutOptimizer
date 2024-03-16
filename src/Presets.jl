module Presets

using CUDA: CuArray

using ..Utils: conditionalSplit, dictToArray
using ..DataProcessing: processDataFolderIntoTextFile
using ..DataStats: computeStats
using ..KeyboardGenerator: layoutGenerator, keyMapGenerator, sp, vsp
using ..DrawKeyboard: computeKeyboardColorMap
using ..Types: RewardArgs, FrequencyRewardArgs, LayoutKey, KeyboardData, CPUArgs, GPUArgs

export runId, randomSeed, textData, dataStats, rewardArgs, algorithmArgs, frequencyRewardArgs, keyboardData, cpuArgs, gpuArgs, dataPaths

# TODO Turn all this into a function

const dataPath = "data" # Path for generated data (everything here can be deleted)
const textPath = joinpath(dataPath, "dataset.txt") # File to save/get text data

const lastRunsPath = joinpath(dataPath, "lastRuns") # History of last runs
const finalResultsPath = joinpath(dataPath, "result") # Results from current run
const startResultsPath = joinpath(finalResultsPath, "start") # Starting keyboards
const endResultsPath = joinpath(finalResultsPath, "end") # End keyboards

const dataPaths = (
    lastRunsPath=lastRunsPath,
    finalResultsPath=finalResultsPath,
    startResultsPath=startResultsPath,
    endResultsPath=endResultsPath,
)

# TODO Move this
# Creating folders and removing old data
map(i -> rm("data/$i", recursive=true), filter(s -> occursin(r"result", s), readdir("data/")))
for path in dataPaths
    mkpath(path)
end
const runId = 1 + last(sort(vcat([0], collect(map(i -> parse(Int, replace(i, r"[^0-9]" => "")), readdir("data/lastRuns/"))))))

# TODO Split layout into list of keys with same size so that they can be shuffled
const keyMap = keyMapGenerator(
    keys=[" ", "zxcvbnm,./", "asdfghjkl;'\n", "\tqwertyuiop[]\\", "`1234567890-="], # Rows of the keyboard layout that are actual characters
    startIndices=[4, 12, 25, 38, 53] # Indices of keys of the first key of each row in the layout map
)

const keyMapCharacters = Set(keys(keyMap))

const noCharKeyMap = keyMapGenerator(
    keys=[["ctrl", "win", "alt", "space", "agr", "fn", "rctl", "lf", "dn", "rt"], ["shift"], ["rshift", "up"], ["caps"], ["enter", "del", "tab"], ["ins"], ["bsp", ""], ["esc"], ["f$i" for i in 1:12], ["psc"]],
    startIndices=[1, 11, 22, 24, 36, 52, 66, 68, 69, 81]
)

# TODO Move this
# Processing data
processDataFolderIntoTextFile("_raw_dataset", textPath, keyMapCharacters, overwrite=false, verbose=true)

# Getting data
const textData = open(io -> read(io, String), textPath, "r")

const randomSeed = 563622

const dataStats = computeStats(
    text=textData,
    fingersCPS=Vector{Float64}([5.5, 5.9, 6.3, 6.2, 6.4, 5.3, 7.0, 6.7, 5.2, 6.2]), # Tested by just pressing the home key of each finger
    rowsCPS=Vector{Float64}([2.27, 3.0, 6.07, 3.0, 2.73, 2.67]), # Bottom to top, tested by bringing the pinky to the respective key and going back to the home key
    # rowsCPS=Vector{Float64}([2.27, 3.07, 6.07, 2.93, 2.73, 2.67]), # Bottom to top, tested by bringing the pinky to the respective key and going back to the home key
    # effortWeighting=(0.7917, 1, 0, 0, 0.4773, 0.00), # dist, double finger, single hand, right hand, finger cps, row cps
)

const (; textStats) = dataStats
const (; charFrequency) = textStats
const keyboardColorMap = computeKeyboardColorMap(charFrequency)

# Keychron Q1 Pro layout
const layoutMap = layoutGenerator(
    rowsList=([
            [(3, 1.25), (1, 6.25), (3,), (sp, 0.25), (3, 1, -0.25)],
            [(1, 2.25), (10,), (1, 1.75), (sp, 0.25), (1, 1, -0.25)],
            [(1, 1.75), (11,), (1, 2.25), (sp, 0.25), (1,)],
            [(1, 1.5), (12,), (1, 1.5), (sp, 0.25), (1,)],
            [(13,), (1, 2), (sp, 0.25), (1,)],
            [(vsp, 0.25)],
            [(1,), (sp, 0.25), (4,), (sp, 0.25), (4,), (sp, 0.25), (4,), (sp, 0.25), (1,)],
        ], (0.5, 0.5)),
    keysFingersList=[
        [1, 11, 24, 25, 38, 39, 53, 54, 68, 69],
        [2, 12, 26, 40, 55, 70],
        [13, 27, 41, 56, 71],
        [14, 15, 16, 28, 29, 42, 43, 57, 58, 72, 73],
        [3, 4],
        [5],
        [17, 18, 30, 31, 44, 45, 59, 60, 74, 75],
        [19, 32, 46, 61, 76],
        [6, 20, 33, 47, 62, 77],
        [7, 8, 9, 10, 21, 22, 23, 34, 35, 36, 37, 48, 49, 50, 51, 52, 63, 64, 65, 66, 67, 78, 79, 80, 81],
    ],
    fingersHome=[25, 26, 27, 28, 4, 4, 31, 32, 33, 34]
)

lmval = first(values(layoutMap))

const fixedKeys = Set("1234567890\t\n\\ ") # Keys that will not change on shuffle
# const fixedKeys = collect("\t\n ") # Numbers also change
getFixedMovableKeyMaps(keyMap) = conditionalSplit(((k, v),) -> k in fixedKeys, keyMap)
const fixedKeyMap, movableKeyMap = getFixedMovableKeyMaps(keyMap)
const movableKeys = [k for (k, v) in movableKeyMap]
const handFingers = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] # What finger is with which hand
const numFingers = length(handFingers)
const numKeys = length(keyMap)
const numLayoutKeys = length(layoutMap)
const numFixedKeys = length(fixedKeyMap)
const numMovableKeys = length(movableKeyMap)

# Total number of iterations will be -epoch * log(t) / log(coolingRate)
const algorithmArgs = (
    t=500, # Temperature
    e=20, # Epoch size (number of iterations in each epoch, before going down in temperature
    nIter=10000, # Number of iterations
    tShuffleMultiplier=0.01, # Is multiplied by temperature to give number of keys shuffled (for 0.01 and t=1000, 10 keys shuffled)
    numKeyboards=1 # Number of keyboards to generate
)

const keyboardSize = 16

const frequencyRewardArgs = FrequencyRewardArgs(
    effortWeighting=NTuple{2,Float64}((0.3, 0.7)), # dist, finger
    xBias=0.75, # [0,1], 0.5 is equal for both
    leftHandBias=0.503, # [0,1], 0.5 is equal for both
    rowCPSBias=(1, 1, 0.8, 1, 1, 1),
    ansKbs=1 / keyboardSize,
)

const rewardArgs = RewardArgs(
    effortWeighting=NTuple{6,Float64}((0.7, 1, 0.2, 0.3, 0.2, 0.15)), # dist, double finger, single hand, right hand, finger cps, row cps
    xBias=0.95, # Lateral movement penalty
    distanceEffort=1.5, # Always positive. At 2, distance penalty is squared
    doubleFingerEffort=1, # Positive prevents using same finger more than once
    singleHandEffort=1, # Positive prefers double hand, negative prefers single hand
    rightHandEffort=1, # Right hand also uses the mouse
    nonNeighborsEffort=0, # Penalty if keys for [], <> and () are not neighbors (0 is no penalty)
    ansKbs=1 / keyboardSize,
)

using Colors

const keyboardData = KeyboardData(
    keyboardColorMap,
    layoutMap,
    keyMapCharacters,
    keyMap,
    noCharKeyMap,
    fixedKeyMap,
    movableKeyMap,
    fixedKeys,
    movableKeys,
    getFixedMovableKeyMaps,
    handFingers,
    numFingers,
    numKeys,
    numLayoutKeys,
    numFixedKeys,
    numMovableKeys,
)

const (; fingerEffort, rowEffort) = dataStats

const cpuArgs = CPUArgs(
    text=collect(textData),
    layoutMap=dictToArray(layoutMap),
    handFingers=handFingers,
    fingerEffort=fingerEffort,
    rowEffort=rowEffort,
)

const gpuArgs = GPUArgs(
    numThreadsInBlock=512,
    text=CuArray(collect(textData)),
    layoutMap=CuArray(dictToArray(layoutMap)),
    handFingers=CuArray(handFingers),
    fingerEffort=CuArray(fingerEffort),
    rowEffort=CuArray(rowEffort),
)

end
