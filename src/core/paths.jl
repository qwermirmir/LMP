# Lattice paths along which fields are parallel transported.
#
# A step is an integer: +mu moves in direction mu (1 = x, 2 = y, 3 = z, 4 = t),
# -mu moves against it, and 0 is the empty step (no transport). A Path is a
# sequence of steps, and Paths is a list of Path. Each PTLayer input uses a
# Paths with a single Path, paths[i].various[1].

"""
    path_number(f)

One step of a path: `f` in `-4:4`, see the file header.
"""
struct path_number
    f::Int64

    function path_number(x::Int64)
        if x in (0, 1, 2, 3, 4, -1, -2, -3, -4)
            return new(x)
        else
            error("path_number must be one of 0, 1, 2, 3, 4, -1, -2, -3, -4; got $x.")
        end
    end
end

struct Path
    steps::Vector{path_number}
end

# Path([1, 2, -4]) builds the path from integer steps.
function Path(ints::Vector{Int})
    pn_array = [path_number(i) for i in ints]
    return Path(pn_array)
end

"""
    Paths(steps1, steps2, ...)
    Paths([steps1, steps2, ...])
    Paths(various::Vector{Path})

A list of paths, each given by its integer steps, e.g. `Paths([[1, 1]])` for
two steps in +x.
"""
struct Paths
    various::Vector{Path}

    function Paths(paths::Vararg{Vector{Int64}})
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end

    function Paths(paths::Vector{Vector{Int64}})
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end

    function Paths(various::Vector{Path})
        return new(various)
    end
end

length(ps::Paths) = length(ps.various)

# Shows a path as its list of steps.
function Base.show(io::IO, p::Path)
    print(io, [step.f for step in p.steps])
end

# paths[i] gives the steps of path i as integers.
function Base.getindex(paths::Paths, i::Int)
    return [step.f for step in paths.various[i].steps]
end

# The inverse path: the steps in reverse order with opposite directions.
function reverse_path(path::Path)
    reversed_steps = [path_number(-step.f) for step in reverse(path.steps)]
    return Path(reversed_steps)
end

"""
    reverse_paths(paths)

Inverts every path, so that transport along `reverse_paths(p)` undoes
transport along `p`.
"""
function reverse_paths(paths::Paths)
    reversed_various = [reverse_path(path) for path in paths.various]
    return Paths([[step.f for step in p.steps] for p in reversed_various])
end

"""
    _default_lmp_paths(max_len=4; path_variant=:long)

The path sets of the LMP models. Every set starts with the empty path.

- `:long`: the eight single steps, and for each of the eight directions a
  straight path of every length 2, 4, 8, ... up to `max_len` (and `max_len`
  itself if it is not a power of two). `max_len = 2` gives 17 paths,
  `max_len = 4` gives 25.
- `:long_only1248`: straight paths of length 1, 2, 4 and 8 in each direction
  (33 paths); `max_len` is ignored.
- `:long_only2468`: straight paths of length 2, 4, 6 and 8 (33 paths).
- `:corner33_ordered`: the eight single steps and the 24 two-step corners
  (+-mu, +-nu) for mu < nu (33 paths).
- `:short_repeat4`: the eight single steps, repeated four times (33 paths).
"""
function _default_lmp_paths(max_len::Int=4; path_variant::Symbol=:long)
    if path_variant == :long_only1248
        P = Paths[Paths([[0]])]
        for len in (1, 2, 4, 8), d in (1, 2, 3, 4, -1, -2, -3, -4)
            push!(P, Paths([fill(d, len)]))
        end
        return P
    end
    if path_variant == :corner33_ordered
        P = Paths[Paths([[0]])]
        for d in (1, 2, 3, 4, -1, -2, -3, -4)
            push!(P, Paths([[d]]))
        end
        for (mu, nu) in ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
            for (smu, snu) in ((1, 1), (1, -1), (-1, 1), (-1, -1))
                push!(P, Paths([[smu * mu, snu * nu]]))
            end
        end
        return P
    end
    if path_variant == :long_only2468
        P = Paths[Paths([[0]])]
        for len in (2, 4, 6, 8), d in (1, 2, 3, 4, -1, -2, -3, -4)
            push!(P, Paths([fill(d, len)]))
        end
        return P
    end
    if path_variant == :short_repeat4
        P = Paths[Paths([[0]])]
        for _ in 1:4, d in (1, 2, 3, 4, -1, -2, -3, -4)
            push!(P, Paths([[d]]))
        end
        return P
    end
    path_variant == :long || throw(ArgumentError("Unknown path variant: $path_variant"))
    P = Vector{Paths}(undef, 9)
    for i in 1:9
        if i <= 5
            P[i] = Paths([[i - 1]])
        else
            P[i] = Paths([[-(i - 5)]])
        end
    end

    dirs = [1, 2, 3, 4, -1, -2, -3, -4]
    lengths = [2^k for k in 1:floor(Int, log2(max_len))]
    if !(max_len in lengths)
        push!(lengths, max_len)
    end
    for d in dirs, len in lengths
        push!(P, Paths([fill(d, len)]))
    end
    return P
end
