module DataStats

using Statistics: mean, std

export computeStats

# TODO Reverse reward
function computeEffort(arrCPM)
    m = mean(arrCPM)
    st = std(arrCPM)
    nzScore = -(arrCPM .- m) ./ st # negative since higher is better
    effort = nzScore .- minimum(nzScore)
    return effort
end

function computeCharHistogram(text)
    chars = Dict{Char,Int}()
    for i in text
        chars[i] = get(chars, i, 0) + 1
    end
    return chars
end

function computeCharFrequency(charHistogram)
    charHistogram = copy(charHistogram)
    total = sum(values(charHistogram))
    freqs = Dict(c => i / total for (c, i) in charHistogram)
    return freqs
end

function computeUsedChars(charHistogram)
    return collect(keys(charHistogram))
end

struct TextStats
    charHistogram::Dict{Char,Int}
    charFrequency::Dict{Char,Float64}
    usedChars::Vector{Char}
end

function computeTextStats(text)
    charHistogram = computeCharHistogram(text)

    return TextStats(
        charHistogram,
        computeCharFrequency(charHistogram),
        computeUsedChars(charHistogram),
    )
end

struct DataStatsType
    fingersCPS::Vector{Float64}
    rowsCPS::Vector{Float64}
    fingerEffort::Vector{Float64}
    rowEffort::Vector{Float64}
    textStats::TextStats
end

function computeStats(;
    text::String,
    fingersCPS::Vector{Float64},
    rowsCPS::Vector{Float64},
)
    return DataStatsType(
        fingersCPS,
        rowsCPS,
        computeEffort(fingersCPS),
        computeEffort(rowsCPS),
        computeTextStats(text),
    )
end

end
