module DataProcessing

using PyCall

export processDataFolderIntoTextFile

# TODO Check mimetype without python
# PyCall block
function __init__()
    py"""
    import mimetypes

    # No mimetype library in Julia (???)
    def get_mime(x):
        return mimetypes.guess_type(x)[0]
    """
end

# Returns all text files in path and its subfolders as one string
function createDatasetFromFolder(path; verbose=false)
    contents = String[]
    mimes = String[]
    for (root, dirs, files) in walkdir(path)
        for (i, file) in enumerate(files)
            fpath = joinpath(root, file)
            mime = py"get_mime"(fpath)
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

function mapSpecialCharacters(dataset)
    # TODO this is specific to language and system layout
    # TODO Put in presets
    # Based on the English International Layout with AltGr Dead Keys

    # First, adds the accents
    # First ā is tilde, second ã is chinese
    ks = collect("áéíóúÁÉÍÓÚàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛãẽĩõũÃẼĨÕŨäëïöüÄËÏÖÜǎěǐǒǔǍĚǏǑǓāēīōūĀĒĪŌŪ")
    cs = collect(repeat("aeiouAEIOU", 7))
    accs = collect(join((x -> repeat(x, 10)).(collect("'`6`'.3"))))
    keyMap = Dict(ks[i] => accs[i] * cs[i] for i in eachindex(ks))

    # Then, adds the rest of the characters
    ks = collect("ç¿¡¥çæœ°º" * raw"!@#$%^&*()" * "~_+{}:\"|<>?“”\r")
    cs = collect(",?!-,zx;;" * raw"1234567890" * "`-=[];'\\,./\"\"\n")
    merge!(keyMap, Dict(ks[i] => string(cs[i]) for i in eachindex(ks)))

    return replace(dataset, keyMap...)
end

# Only processes data if target file does not exist yet
function processDataFolderIntoTextFile(srcfolder, destfile; overwrite=false, verbose=false)
    if !overwrite && isfile(destfile)
        verbose && println("Skipped processing data, Processed data file found")
        return
    end

    dataset = createDatasetFromFolder(srcfolder)
    dataset = removeLinks(dataset) # Remove links, since we normally copy paste them instead of writing them
    dataset = mapSpecialCharacters(dataset) # Maps special characters to their respective keys pressed
    dataset = lowercase(dataset) # Uppercase or lowercase to use a single letter for each key
    dataset = strip(dataset)

    open(f -> write(f, dataset), destfile, "w")
end

end
