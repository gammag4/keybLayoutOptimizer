import Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Revise
includet("main.jl")
