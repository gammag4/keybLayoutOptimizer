module DataProcessing

using ..Utils: conditionalSplit

# TODO Check mimetype without python
using PyCall: pyimport

export processDataFolderIntoTextFile

const mimetypes = pyimport("mimetypes")

# Returns all text files in path and its subfolders as one string
function createDatasetFromFolder(path; verbose=false)
    contents = String[]
    mimes = String[]
    for (root, dirs, files) in walkdir(path)
        for (i, file) in enumerate(files)
            fpath = joinpath(root, file)
            mime = mimetypes.guess_type(fpath)[1]
            # Uses only text data
            # The image part removes images like svg (which are xml)
            # TODO Remove all svg types
            if mime !== nothing && Base.Multimedia.istextmime(mime) && !occursin("image", mime)
                fcont = open(f -> read(f, String), fpath, "r")
                push!(contents, fcont * "\n\n")
                push!(mimes, mime)
            end
        end
    end

    verbose && println("Mimetypes processed:")
    verbose && (x -> println(x)).(mimes)

    return strip(join(contents))
end

function removeLinks(dataset)
    # TODO instead of removing links, put shortcuts for ctrl+c and ctrl+v
    # TODO map all data into shortcuts
    urlregex = r"(http(s)?:\/\/.)(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
    return replace(dataset, urlregex => "")
end

bb(i, c, s, a) = (UInt8(i), UInt8(c * 4 | s * 2 | a))
n(i) = bb(i, 0, 0, 0) # No mod
c(i) = bb(i, 1, 0, 0) # Ctrl
s(i) = bb(i, 0, 1, 0) # Shift
ag(i) = bb(i, 0, 0, 1) # AltGR
cs(i) = bb(i, 1, 1, 0) # Ctrl + Shift
ags(i) = bb(i, 0, 1, 1) # AltGr + Shift

# Creates the map that maps normal and special chars to keys with modifiers, returning for each a Int16 with the info (char key, ctrl, shift, altgr)
function createCharMap(keyMapCharacters)
    # TODO this is specific to language and system layout
    # TODO Put in presets
    # Based on the English International Layout with AltGr Dead Keys

    # First, adds the accents
    # First ā is tilde, second ã is chinese
    ks = collect("áéíóúÁÉÍÓÚàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛãẽĩõũÃẼĨÕŨäëïöüÄËÏÖÜǎěǐǒǔǍĚǏǑǓāēīōūĀĒĪŌŪ")
    accs = repeat(vcat(ag.(collect("'`6")), ags.(collect("`'.3"))), 10)
    cs = repeat(vcat(n.(collect("aeiou")), s.(collect("aeiou"))), 7)
    charMap = Dict(ks[i] => [accs[i], cs[i]] for i in eachindex(ks))

    # Then, adds the rest of the characters
    ks = collect(raw"!@#$%^&*()" * "~_+{}:\"|<>?" * "ç¿¥æœ" * "Ç¡°º" * "“”ʼ’\r")
    cs = vcat(
        s.(collect("1234567890")),
        s.(collect("`-=[];'\\,./")),
        ag.(collect(",/-zx")),
        ags.(collect(",1;;")),
        n.(collect("\"\"''\n"))
    )
    merge!(charMap, Dict(ks[i] => [cs[i]] for i in eachindex(ks)))
    merge!(charMap, Dict(i => [s(lowercase(i))] for i in 'A':'Z')) # Uppercase letters
    merge!(charMap, Dict(i => [n(i)] for i in keyMapCharacters))

    return charMap
end

# Only processes data if target file does not exist yet
function processDataFolderIntoTextFile(srcfolder, destfile, keyMapCharacters; overwrite=false, verbose=false)
    if !overwrite && isfile(destfile)
        verbose && println("Skipped processing data, Processed data file found")
        return
    end

    dataset = createDatasetFromFolder(srcfolder)
    dataset = removeLinks(dataset) # Remove links, since we normally copy paste them instead of writing them
    #TODO = createCharMap() # Maps special characters to their respective keys pressed
    #TODO dataset = replace(dataset, !isascii => "") # Remove non-ascii characters

    charMap = createCharMap(keyMapCharacters)

    binr = Vector{Tuple{UInt8,UInt8}}()

    noKeysSet = Set()

    for c in dataset
        if !haskey(charMap, c)
            push!(noKeysSet, c)
            continue
        end
        for i in charMap[c]
            push!(binr, i)
        end
    end

    println("These keys are not in the keyMap provided, you might want to include them: \"$(join(noKeysSet))\"")

    write(destfile, binr)
end

end
