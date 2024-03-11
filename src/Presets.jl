module Presets

include("./Presets/Generators.jl")

using .Generators

# Alphabet, n-th letter corresponds to
const letterList = [
    'A',
    'B',
    'C',
    'D',
    'E',
    'F',
    'G',
    'H',
    'I',
    'J',
    'K',
    'L',
    'M',
    'N',
    'O',
    'P',
    'Q',
    'R',
    'S',
    'T',
    'U',
    'V',
    'W',
    'X',
    'Y',
    'Z',
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '~',
    '-',
    '+',
    '[',
    ']',
    ';',
    ''',
    '<',
    '>',
    '?'
]

# TODO Change to 'a' => 1, ... and let everything lowercase

# Keychron Q1 Pro layout
const defaultLayoutMap = layoutGenerator(
    rowsList=([
            [(1, 1, 0), (sp, 0.25, 0), (4, 1, 0), (sp, 0.25, 0), (4, 1, 0), (sp, 0.25, 0), (4, 1, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(vsp, 0.25, 0)],
            [(13, 1, 0), (1, 2, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(1, 1.5, 0), (12, 1, 0), (1, 1.5, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(1, 1.75, 0), (11, 1, 0), (1, 2.25, 0), (sp, 0.25, 0), (1, 1, 0)],
            [(1, 2.25, 0), (10, 1, 0), (1, 1.75, 0), (sp, 0.25, 0), (1, 1, 0.25)],
            [(3, 1.25, 0), (1, 6.25, 0), (3, 1, 0), (sp, 0.25, 0), (3, 1, 0.25)],
        ], (0.5, 0.5)),
    keysFingersList=[
        [1, 15, 30, 45, 59, 72, 2, 16, 31, 46, 60, 73, 74],
        [3, 17, 32, 47, 61],
        [4, 18, 33, 48, 62],
        [5, 19, 34, 49, 63, 6, 20, 35, 50, 64],
        [75],
        [],
        [7, 21, 36, 51, 65, 8, 22, 37, 52, 66],
        [9, 23, 38, 53, 67, 76],
        [10, 24, 39, 54, 68, 77],
        [11, 12, 13, 14, 25, 26, 27, 28, 29, 40, 41, 42, 43, 44, 55, 56, 57, 58, 69, 70, 71, 78, 79, 80, 81]
    ],
    fingersHome=[46, 47, 48, 49, 75, 75, 52, 53, 54, 55]
)

const keyMapDict = keyMapGenerator(
    keys=["`1234567890-=", "\tqwertyuiop[]\\", "asdfghjkl;'\n", "zxcvbnm,./", " "], # Rows of the keyboard layout that are actual characters
    startIndices=[15, 30, 46, 60, 75] # Indices of keys of the first key of each row in the layout map
)

const fixedKeys = collect("1234567890\t\n ") # Keys that will not change on shuffle
#const fixedKeys = collect("\t\n ") # Numbers also change

const handList = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2] # What finger is with which hand

const numFingers = length(handList)

const numLayoutKeys = length(defaultLayoutMap)

const numKeys = length(keyMapDict)

const numFixedKeys = numKeys - length(fixedKeys)

export defaultLayoutMap, letterList, keyMapDict, fixedKeys, handList, numFingers, numLayoutKeys, numKeys, numFixedKeys

end
