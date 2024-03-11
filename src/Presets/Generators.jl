module Generators

const sp = -2
const vsp = -4

removeVSp(rowslist) = filter(x -> x[1][1] != vsp, rowslist[1])
getCurrentRow(y) = floor(Int, y)
getNumRows(rowslist) = getCurrentRow(rowslist[2][2]) + length(removeVSp(rowslist))
getTotalNumRows(rowslistlist) = maximum(getNumRows.(rowslistlist))

function getLayoutRow(row, x, y)
    pos = x
    lrow = Tuple{Float64,Float64,Float64}[]

    for (n, l, h) in row
        if n == sp
            pos += l
            continue
        end

        append!(lrow, map(((i, (x, y, l)),) -> (x + (i - 1) * l, y, l), enumerate(repeat([(pos + (l / 2) - 0.5, y + h, l)], n))))
        pos += l * n
    end

    return lrow
end

function getLayoutRowsList(rowslist)
    rowslist, (x, y) = rowslist
    pos = y
    lrows = repeat([[]], getCurrentRow(y))

    for row in rowslist
        (n, l) = row[1]
        if n == vsp
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
            keyFinger = f
            if key == fingersHome[f]
                return (keyFinger, 1)
            else
                return (keyFinger, 0)
            end
        end
    end
end

addFingers(layoutTuples, keysFingersList, fingersHome) =
    map(((i, ((x, y, l), r)),) -> ((x, y, l), getFingerData(keysFingersList, fingersHome, i), r), enumerate(layoutTuples))

# Returns list of tuples with ((x, y), (finger, ishome), row number)
function layoutGenerator(; rowsList, keysFingersList, fingersHome)
    # Rows list: Tuple(array(rows/vspacers), (starting x, starting y))
    # Row: array(key batch/spacers)
    # Vspacer: [(vsp, vlength, 0)]
    # Key batch: (numkeys, length, ydelta)
    # Spacer: (sp, length)

    layoutRowsList = getLayoutRowsList(rowsList)

    maprows = ((i, r),) -> map(((x, y, l),) -> ((x, y, l), i), r)
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

export sp, vsp, layoutGenerator, keyMapGenerator

end
