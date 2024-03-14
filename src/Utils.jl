module Utils

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

end
