module KeyboardGenerator

using ..Utils: rpadIter, rpadIterArr

export layoutGenerator, keyMapGenerator

removeVSp(rowslist) = filter(x -> x[1][1] != "vsp", rowslist[1])
getCurrentRow(y) = floor(Int, y)
getNumRows(rowslist) = getCurrentRow(rowslist[2][2]) + length(removeVSp(rowslist))
getTotalNumRows(rowslistlist) = maximum(getNumRows.(rowslistlist))

function getLayoutRow(row, x, y)
    pos = x
    lrow = NTuple{4,Float64}[]

    for batch in row
        # number of keys in batch, width of each key, delta position in y axis (when not aligned vertically to others), height (default 1)
        (n, w, dh, h) = rpadIterArr(batch, 4, [1, 0, 1])
        if n == "sp"
            pos += w
            continue
        end

        append!(lrow, map(((i, (x, y, w)),) -> (x + (i - 1) * w, y, w, h), enumerate(repeat([(pos + (w / 2) - 0.5, y + dh, w)], n))))
        pos += w * n
    end

    return lrow
end

function getLayoutRowsList(rowslist)
    rowslist, (x, y) = rowslist
    pos = y
    lrows = repeat([[]], getCurrentRow(y))

    for row in rowslist
        (n, l) = rpadIter(row[1], 2, 0)
        if n == "vsp"
            pos += l
            continue
        end

        push!(lrows, getLayoutRow(row, x, pos))
        pos += 1
    end

    return lrows
end

function getFingerData(keysFingersList, fingersHome, key)
    for f in 1:10
        if key in keysFingersList[f]
            return (f, fingersHome[f])
        end
    end
end

addFingers(layoutTuples, keysFingersList, fingersHome) =
    map(((i, ((x, y, w, h), r)),) -> ((x, y, w, h), getFingerData(keysFingersList, fingersHome, i), r), enumerate(layoutTuples))

# Returns dictionary that maps key numbers to tuples with ((x, y, w, h), (finger, home key), row number)
function layoutGenerator(; rowsList, keysFingersList, fingersHome)
    # Rows list: Tuple(array(rows/vspacers), (starting x, starting y))
    # Row: array(key batch/spacers)
    # Vspacer: [(vsp, vlength, 0)]
    # Key batch: (numkeys, length, ydelta)
    # Spacer: (sp, length)

    layoutRowsList = getLayoutRowsList(rowsList)

    maprows = ((i, r),) -> map(((x, y, w, h),) -> ((x, y, w, h), i), r)
    layoutTuples = reduce(vcat, map(maprows, enumerate(layoutRowsList)))
    layoutTuples2 = addFingers(layoutTuples, keysFingersList, fingersHome)
    layout = Dict(i => t for (i, t) in enumerate(layoutTuples2))
    return layout
end

function keyMapGenerator(; keys, startIndices)
    sts = collect(map(((s, i),) -> collect(i:(i+length(s)-1)), zip(keys, startIndices)))
    ks = typeof(keys) == Vector{String} ? join(keys) : reduce(vcat, keys)
    ids = reduce(vcat, sts)
    return Dict(ks[i] => ids[i] for i in eachindex(ks))
end

end
