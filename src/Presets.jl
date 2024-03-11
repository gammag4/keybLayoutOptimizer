module Presets

include("./Presets/Generators.jl")

using .Generators

# TODO Change to 'a' => 1, ... and let everything lowercase

const fingersHome = [25, 26, 27, 28, 4, 4, 31, 32, 33, 34]

# Keychron Q1 Pro layout
const defaultLayoutMap = layoutGenerator(
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
        [1, 2, 3, 11, 12, 24, 25, 38, 39, 53, 54, 68, 69],
        [13, 26, 40, 55, 70],
        [14, 27, 41, 56, 71],
        [15, 16, 28, 29, 42, 43, 57, 58, 72, 73],
        [4],
        [],
        [17, 18, 30, 31, 44, 45, 59, 60, 74, 75],
        [5, 19, 32, 46, 61, 76],
        [6, 20, 33, 47, 62, 77],
        [7, 8, 9, 10, 21, 22, 23, 34, 35, 36, 37, 48, 49, 50, 51, 52, 63, 64, 65, 66, 67, 78, 79, 80, 81],
    ],
    fingersHome=fingersHome
)

const keyMapDict = keyMapGenerator(
    keys=[" ", "zxcvbnm,./", "asdfghjkl;'\n", "\tqwertyuiop[]\\", "`1234567890-="], # Rows of the keyboard layout that are actual characters
    startIndices=[4, 12, 25, 38, 53] # Indices of keys of the first key of each row in the layout map
)

const noCharKeys = keyMapGenerator(
    keys=[["ctrl", "win", "alt"], ["agr", "fn", "rctl", "lf", "dn", "rt"], ["shift"], ["rshift", "up"], ["caps"], ["enter", "del"], ["ins"], ["bsp", ""], ["esc"], ["f$i" for i in 1:12], ["psc"]],
    startIndices=[1, 5, 11, 22, 24, 36, 52, 66, 68, 69, 81]
)

const fixedKeys = collect("1234567890\t\n ") # Keys that will not change on shuffle
#const fixedKeys = collect("\t\n ") # Numbers also change

const handList = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] # What finger is with which hand

const numFingers = length(handList)

const numLayoutKeys = length(defaultLayoutMap)

const numKeys = length(keyMapDict)

const numFixedKeys = numKeys - length(fixedKeys)

export fingersHome, defaultLayoutMap, keyMapDict, noCharKeys, fixedKeys, handList, numFingers, numLayoutKeys, numKeys, numFixedKeys

end
