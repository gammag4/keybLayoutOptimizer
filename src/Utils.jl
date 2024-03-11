module Utils

function conditionalSplit(f, v)
    a = typeof(v)()
    b = typeof(v)()

    for i in v
        if f(i)
            push!(a, i)
        else
            push!(b, i)
        end
    end

    return a, b
end

export conditionalSplit

end
