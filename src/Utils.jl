module Utils

using Statistics: mean, std
using Base.Iterators: flatten, take, drop, repeated

export conditionalSplit, lpadIter, rpadIter, mergeIter

function conditionalSplit(f, v)
    a = typeof(v)()
    b = typeof(v)()

    for i in v
        f(i) ? push!(a, i) : push!(b, i)
    end

    return a, b
end

lpadIter(v, n, i) = flatten((take(repeated(i), n - length(v)), v))

rpadIter(v, n, i) = flatten((v, take(repeated(i), n - length(v))))

lpadIterArr(v, n, v1) = flatten((take(v1, n - length(v)), v))

rpadIterArr(v, n, v1) = flatten((v, drop(v1, length(v1) - n + length(v))))

mergeVal(a, b) = isnothing(b) ? a : b

mergeIter(v...) = reduce((v1, v2) -> (mergeVal(a, b) for (a, b) in zip(v1, v2)), v)

function dictToArray(d)
    v = first(d)
    res::Vector{typeof(v[2])} = [get(d, typeof(v[1])(i), v[2]) for i in 1:maximum(Int.(keys(d)))]
    return res
end

# Use this to insert one variable at a time in a multivariable function, where g is the function and i is the total number of arguments
# The arguments are put in the order they appear
# The precedence is from right to left
# E.g.: 3 → 78 → 21 → (+, 3) returns 102
→(x, (g, i)) = i == 1 ? g(x) : ((args...) -> g(args..., x), i - 1)

# Scales to range [a,b], can also reverse the order (a > b)
function minMaxScale(v, a, b)
    mi, ma = minimum(v), maximum(v)
    return (x -> (b - a) * (x - mi) / (ma - mi) + a).(v)
end

minMaxScale(v) = minMaxScale(v, 0, 1)

lpTransform(v, k) = minMaxScale(minMaxScale(v, 0, 1) .^ k, minimum(v), maximum(v))

function splitDict(d)
    l = length(d)
    a, b = Vector{keytype(d)}(undef, l), Vector{valtype(d)}(undef, l)
    i = 1
    for p in d
        a[i], b[i] = p
        i += 1
    end

    return a, b
end

function zscore(v)
    m = mean(v)
    st = std(v)
    return (x -> (x - m) / st).(v)
end

end
