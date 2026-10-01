# Lattice paths. A path is a sequence of unit steps (+mu or -mu, mu = 1..4)
# along which a fermion field is parallel transported. Includes the default
# path sets used by the LMP models.

struct path_number
    f::Int64

    function path_number(x::Int64)
        # 허용되는 값인지 확인
        if x in (0, 1, 2, 3, 4, -1, -2, -3, -4)
            return new(x)
        else
            error("path_number는 0,1,2,3,4,-1,-2,-3,-4 중 하나만 허용합니다.")
        end
    end
end

struct Path
    steps::Vector{path_number}
end

function Path(ints::Vector{Int})
    # ints: [1, 2, -4, ...]
    # 변환 -> path_number(...) 배열
    pn_array = [path_number(i) for i in ints]
    return Path(pn_array)
end

struct Paths
    various::Vector{Path}

    # ✅ 여러 개의 개별 벡터를 받음
    function Paths(paths::Vararg{Vector{Int64}})  # 개별 벡터를 받음
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end

    # ✅ 기존 방식: 한 번에 묶인 Vector{Vector{Int64}} 받음
    function Paths(paths::Vector{Vector{Int64}})
        various = [Path([path_number(x) for x in path]) for path in paths]
        return new(various)
    end

    # 기존 Path 리스트를 받는 생성자 유지
    function Paths(various::Vector{Path})
        return new(various)
    end
end

length(ps::Paths) = length(ps.various)

# ✅ 개별 Path를 리스트 형태로 출력
function Base.show(io::IO, p::Path)
    print(io, [step.f for step in p.steps])
end

# ✅ 개별 Path의 steps를 Int 리스트로 반환
function Base.getindex(paths::Paths, i::Int)
    return [step.f for step in paths.various[i].steps]
end

function reverse_path(path::Path)
    reversed_steps = [path_number(-step.f) for step in reverse(path.steps)]
    return Path(reversed_steps)
end

function reverse_paths(paths::Paths)
    reversed_various = [reverse_path(path) for path in paths.various]
    return Paths([[step.f for step in p.steps] for p in reversed_various])  # 개별 원소에 `.f` 적용
end

function _default_lmp_paths(max_len::Int=4)
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
