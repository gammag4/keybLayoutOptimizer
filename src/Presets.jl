module Presets

using DataProcessing
using DataStats
using KeyboardGenerator
using DrawKeyboard

# TODO Turn all this into a function

textpath = "data/dataset.txt" # File to save/get text data

# Processing data
mkpath("data/result")
processDataFolderIntoTextFile("data/raw_dataset", textpath, overwrite=false, verbose=true)

# Getting data
textData = open(io -> read(io, String), textpath, "r")

const dataStats = computeStats(
    text=textData,
    fingersCPS=Vector{Float64}([5.5, 5.9, 6.3, 6.2, 6.4, 5.3, 7.0, 6.7, 5.2, 6.2]), # Tested by just pressing the home key of each finger
    rowsCPS=Vector{Float64}([2.36, 2.4, 5.07, 2.6, 2.47, 1.87]), # Top to bottom, tested by bringing the pinky to the respective key and going back to the home key
    effortWeighting=NTuple{6,Float64}((0.05, 1, 1, 0.8, 0.4, 0.3),), # dist, double finger, single hand, right hand, finger cps, row cps
    #effortWeighting=(0.7917, 1, 0, 0, 0.4773, 0.00), # dist, double finger, single hand, right hand, finger cps, row cps
)

const (; textStats) = dataStats
const (; charFrequency) = textStats

const keyboardColorMap = computeKeyboardColorMap(charFrequency)

const fingersHome = [25, 26, 27, 28, 4, 4, 31, 32, 33, 34]

# Keychron Q1 Pro layout
const layoutMap = layoutGenerator(
    rowsList=([
            [(3, 1.25, 0), (1, 6.25, 0), (3, 1, 0), (sp, 0.25, 0), (3, 1, -0.25)],
            [(1, 2.25, 0), (10, 1, 0), (1, 1.75, 0), (sp, 0.25, 0), (1, 1, -0.25)],
            [(1, 1.75, 0), (11, 1, 0), (1, 2.25, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(1, 1.5, 0), (12, 1, 0), (1, 1.5, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(13, 1, 0), (1, 2, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(vsp, 0.25, 0)],
            [(1, 1, 0), (sp, 0.25, 0), (4, 1, 0), (sp, 0.25, 0), (4, 1, 0), (sp, 0.25, 0), (4, 1, 0), (sp, 0.25, 0), (1, 1, 0)],
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
    fingersHome=fingersHome
)

const keyMap = keyMapGenerator(
    keys=[" ", "zxcvbnm,./", "asdfghjkl;'\n", "\tqwertyuiop[]\\", "`1234567890-="], # Rows of the keyboard layout that are actual characters
    startIndices=[4, 12, 25, 38, 53] # Indices of keys of the first key of each row in the layout map
)

const noCharKeyMap = keyMapGenerator(
    keys=[["ctrl", "win", "alt"], ["agr", "fn", "rctl", "lf", "dn", "rt"], ["shift"], ["rshift", "up"], ["caps"], ["enter", "del"], ["ins"], ["bsp", ""], ["esc"], ["f$i" for i in 1:12], ["psc"]],
    startIndices=[1, 5, 11, 22, 24, 36, 52, 66, 68, 69, 81]
)

const fixedKeys = collect("1234567890\t\n ") # Keys that will not change on shuffle
#const fixedKeys = collect("\t\n ") # Numbers also change
const handFingers = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] # What finger is with which hand
const numFingers = length(handFingers)
const numKeys = length(keyMap)
const numLayoutKeys = length(layoutMap)
const numMovableKeys = length(keyMap) - length(fixedKeys)

using Colors

struct KeyboardData
    keyboardColorMap::Dict{Char,HSV}
    layoutMap::Dict{Int,Tuple{NTuple{3,Float64},NTuple{2,Int},Int}}
    keyMap::Dict{Char,Int}
    noCharKeyMap::Dict{String,Int}
    fixedKeys::Vector{Char}
    fingersHome::Vector{Int}
    handFingers::Vector{Int}
    numFingers::Int
    numKeys::Int
    numLayoutKeys::Int
    numMovableKeys::Int
end

const keyboardData = KeyboardData(
    keyboardColorMap,
    layoutMap,
    keyMap,
    noCharKeyMap,
    fixedKeys,
    fingersHome,
    handFingers,
    numFingers,
    numKeys,
    numLayoutKeys,
    numMovableKeys,
)

export textData, dataStats, keyboardData, drawKeyboard

end
