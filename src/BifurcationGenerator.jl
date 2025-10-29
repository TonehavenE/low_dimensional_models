module BifurcationGenerator
export generate_branch, generate_branches, generate_psi, generate_psi_ijkl, generate_psi_ijkl_a_b_n, map_x, reverse_mapping, ode_matrices, fuzz_guess, fuzz_guesses, power_input, SymmetryGroup, eqb_matrices, tw_matrices, get_α_γ, get_nsymm

using Revise, Polynomials, BifurcationKit, Plots, DelimitedFiles, Statistics, WGLMakie, LinearAlgebra, Base.Threads
include("BasisFunctions.jl")
include("Hookstep.jl")
using .BasisFunctions
using .Hookstep
myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')

###########################################
# SymmetryGroup is just a list of all the symmetries 
"""
A Struct that contains all of the symmetries we want to explore. In particular:
- A:  <sxyz, txz>
- B:  <sxy, sz>
- C:  <sxytz, sz>
- D:  <sxy, sztx>
- E:  <sxyz, sztxz> EQ1
- E': <sxytz, sztx>
- F:  <sxy, sz, txz> EQ7
"""
struct SymmetryGroup
    value::Symbol
    function SymmetryGroup(value::Symbol)
        allowed = (:A, :B, :C, :D, :E, :Eprime, :F, :TW1, :TW2, :sztxz)
        if value in allowed
            new(value)
        else
            error("Invalid symmetry group value: $value")
        end
    end
end

function get_nsymm(ijklfull, symmetry)
    # Generate ±1 vectors showing symmetry/antisymmetry of each basis element Ψᵢⱼₖₗ
    # w.r.t. the generators of the symmetry group. 
    sx = xreflection(ijklfull)
    sy = yreflection(ijklfull)
    sz = zreflection(ijklfull)
    tx = xtranslationLx2(ijklfull)
    tz = ztranslationLz2(ijklfull)

    # Determine ±1 symmetry/antisymmetry vectors for some other group elements
    sxy = sx .* sy
    txz = tx .* tz
    sxyz = sx .* sy .* sz
    sztxz = sz .* tx .* tz
    sztx = sz .* tx
    sxytz = sxy .* tz
    sxytxz = sxy .* txz
    
    if symmetry.value == :A
        nsymm = findsymmetric(sxy, sztx)   # A = <sxyz, txz>
    elseif symmetry.value == :B
        nsymm = findsymmetric(sxy, sz)     # B = <sxy, sz>
    elseif symmetry.value == :C
        nsymm = findsymmetric(sxytz, sz)   # C = <sxytz, sz>
    elseif symmetry.value == :D
        nsymm = findsymmetric(sxy, sztx)   # D = <sxy, sztx> <- TRY THIS
    elseif symmetry.value == :E
        nsymm = findsymmetric(sxyz, sztxz)  # E = <sxyz, sztxz>, EQ1 symmetries
    elseif symmetry.value == :Eprime
        nsymm = findsymmetric(sxytz, sztx) # E' = <sxytz, sztx> (E' is conjugate to E)
    elseif symmetry.value == :F
        nsymm = findsymmetric(sxy, sz, txz) # F = <sxy, sz, txz>, EQ7 symmetries
    elseif symmetry.value == :TW1
        nsymm = findsymmetric(sxytxz) # TW1 = sxy*txz = z traveling
    elseif symmetry.value == :TW2
        nsymm = findsymmetric(sztx) # sz*tx traveling waves in x
    elseif symmetry.value == :sztxz
        nsymm = findsymmetric(sztxz) # sz*txz = x traveling
    else
        throw("Unknown symmetry group")
    end

    return nsymm
end

###########################################
# BASIS SET GENERATION AND MANIPULATION
###########################################
"""
Returns the basis set for a discretization determined by `jkl` tuple. 
`α = 1` and `γ = 2` are parameters for the wavenumber. The symmetry map is as follows:

- A:  <sxyz, txz>
- B:  <sxy, sz>
- C:  <sxytz, sz>
- D:  <sxy, sztx>
- E:  <sxyz, sztxz> EQ1
- E': <sxytz, sztx>
- F:  <sxy, sz, txz> EQ7

Arguments:
- jkl: the basis set to generate
- α: the streamwise wavenumber
- γ: the spanwise wavenumber
- chebyshev: whether to use Chebyshev or Legendre polynomials as the basis
- symmetry: a letter corresponding to the symmetry group to use when generating the functions (see above)

Returns:
- the set of `BasisFunction`s
"""
function generate_psi(jkl::Tuple{Int64,Int64,Int64}, α=1, γ=2, chebyshev=true, symmetry=B)
    J, K, L = jkl
    ijklfull = basisIndexMap(J, K, L)
    Ψfull = makeBasisSet(α, γ, ijklfull, normalize=false, chebyshev=chebyshev)
    nsymm = get_nsymm(ijklfull, symmetry)
    Ψ = Ψfull[nsymm]
    return Ψ
end

"""
Given a basis set Ψ, computes the matrices representing the coefficients of the system of ordinary differential equations for equilibria. 
"""
function eqb_matrices(Ψ)
    Nmodes = length(Ψ)
    y = Polynomials.Polynomial([0; 1], :y)

    # Calculate linear terms
    B = [innerproduct(Ψ[i], Ψ[j]) for i in 1:Nmodes, j in 1:Nmodes]
    A1 = [-innerproduct(Ψ[i], y * xderivative(Ψ[j])) for i in 1:Nmodes, j in 1:Nmodes]
    A2 = [-innerproduct(Ψ[i], vex(Ψ[j])) for i in 1:Nmodes, j in 1:Nmodes]
    A3 = [innerproduct(Ψ[i], laplacian(Ψ[j])) for i in 1:Nmodes, j in 1:Nmodes]


    # Nonlinear terms
    Ndense = fill(0 // 1, Nmodes, Nmodes, Nmodes)

    for j in 1:Nmodes
        for k in 1:Nmodes

            Ψj_dotgrad_Ψk = dotgrad(Ψ[j], Ψ[k])

            for i in 1:Nmodes
                try
                    val = -innerproduct(Ψ[i], Ψj_dotgrad_Ψk) # <- overflows
                    Ndense[i, j, k] = abs(val) > 1e-15 ? val : 0
                catch e
                    Ndense[i, j, k] = typemax(Int64)
                end
            end
        end
    end

    N = SparseBilinear(Ndense)
    Nf = SparseBilinear(N.ijk, Float64.(N.val), Nmodes)

    # Create functions for bifurcation continuation (dependent on R)
    Bf = Float64.(B)
    # Convert A matrices to Float64 *once*
    A1f = Float64.(A1)
    A2f = Float64.(A2)
    A3f = Float64.(A3)

    # Define Af without the Float64.() wrapper
    Af(R) = A1f + A2f + (1 / R) * A3f
    
    return Af, Bf, Nf
end

"""
Backwards compatiable, now renamed to eqb_matrices
"""
function ode_matrices(Ψ)
    return eqb_matrices(Ψ)
end

"""
Given a basis set Ψ, compute the matrices representing the coefficients needed for a traveling wave search. 
"""
function tw_matrices(Ψ)
    Nmodes = length(Ψ)
    A, _, N = eqb_matrices(Ψ)
    Cx = [innerproduct(Ψ[i], xderivative(Ψ[j])) for i in 1:Nmodes, j in 1:Nmodes]
    Cz = [innerproduct(Ψ[i], zderivative(Ψ[j])) for i in 1:Nmodes, j in 1:Nmodes]
    return A, Float64.(Cx), Float64.(Cz), N
end

"""
Given a tuple of (J, K, L) values and optional wavenumbers, Chebyshev vs. Legendre specifier, and symmetry group
calculates and returns the basis set Ψ, the IJKL map, and the matrices representing the system of ODEs A, B, N.
"""
function generate_psi_ijkl_a_b_n(jkl::Tuple{Int64, Int64, Int64}; α = 1, γ = 2, chebyshev=true, symmetry=B)
    J, K, L = jkl
    ijklfull = basisIndexMap(J,K,L)
    Ψfull = makeBasisSet(α, γ, ijklfull, normalize=false, chebyshev=chebyshev)
    nsymm = get_nsymm(ijklfull, symmetry) 
    ijkl = ijklfull[nsymm, :]
    Ψ = Ψfull[nsymm]
    A, B, N = ode_matrices(Ψ)
    return Ψ, ijkl, A, B, N
end

function generate_psi_ijkl(jkl::Tuple{Int64, Int64, Int64}; α = 1, γ = 2, symmetry=B)
    Ψ, ijkl, _, _, _ = generate_psi_ijkl_a_b_n(jkl, α=α, γ=γ, symmetry=symmetry)
    return Ψ, ijkl
end


###########################################
# BASIS SET MAPPINGS
###########################################
"""
Given a vector with ijkl[n] = (i, j, k, l) create a dict N[(i, j, k, l)] = n
"""
function reverse_mapping(ijkl)
    N = length(ijkl[:, 1])
    reverse_map = Dict() 
    for n in 1:N
        i, j, k, l = ijkl[n, :]
        # use a key of the 4-tuple (i, j, k, l) and store the corresponding n
        reverse_map[(i, j, k, l)] = n
    end
    return reverse_map
end

"""
Takes a point `x` in the basis set determined by jkl1 and projects it onto a point in jkl2.
"""
function map_x(x, jkl1, jkl2)
    _, ijkl1 = generate_psi_ijkl(jkl1)
    _, ijkl2 = generate_psi_ijkl(jkl2)
    x_out = zeros(size(ijkl2, 1))
    reverse = reverse_mapping(ijkl2)
    for index in 1:size(ijkl1, 1)
        i, j, k, l = ijkl1[index, :]
        x_out[reverse[(i, j, k, l)]] = x[index]
    end
    return x_out
end

function map_x(x, jkl1, jkl2)
    _, ijkl1 = generate_psi_ijkl(jkl1)
    _, ijkl2 = generate_psi_ijkl(jkl2)
    map_x(x, ijkl1, ijkl2)
end

"""
Takes a point `x` in the basis set determined by ijkl1 and projects it onto a point in ijkl2.
"""
function map_x(x::Vector{Float64}, ijkl1::Matrix{Int64}, ijkl2::Matrix{Int64})
    x_out = zeros(size(ijkl2, 1))
    reverse = reverse_mapping(ijkl2)
    for index in 1:size(ijkl1, 1)
        i, j, k, l = ijkl1[index, :]
        x_out[reverse[(i, j, k, l)]] = x[index]
    end
    return x_out
end

###########################################
# FUZZING GUESSES
###########################################

"""
Takes an initial guess `xproj` and Reynolds Number `R` and fuzzes nearby to find a solution. Returns the first solution found currently, or the original guess if nothing was found.
"""
function fuzz_guess(xproj::Vector{Float64}, R::Float64, f, Df, Nfuzz=100)
    # Define continuation function
    g(x) = f(x, (R=R,))
    Dg(x) = Df(x, (R=R,))
    Nmodes = length(xproj)

    # Hone in on xproj
    x, X = hookstepsolve(g, xproj, verbose=false)

    xnorm = 0.40

    for i in 1:Nfuzz
        xguess = randn(Nmodes)
        xguess = xnorm / norm(xguess) * xguess

        xsoln, X = hookstepsolve(g, Dg, xguess, verbose=false)

        if norm(xsoln) > 1e-02 && norm(g(xsoln)) < 1e-07
            return xsoln
        end
    end
    return xproj # we didn't find anything new
end

"""
Takes a vector `Xsoln` and returns the elements that are unique up to `digits`.
"""
function qualitatively_unique_vectors(Xsoln, digits=3)
    seen_norms = Dict()  # Dictionary to store unique rounded norms and their index
    unique_vectors = []

    for i in eachindex(Xsoln)
        rounded_norm = round(norm(Xsoln[i]), sigdigits=digits)
        if !haskey(seen_norms, rounded_norm)
            seen_norms[rounded_norm] = i  # Store the first occurrence
            push!(unique_vectors, Xsoln[i])
        end
    end

    return unique_vectors
end

"""
Takes an initial guess `xproj` and Reynolds Number `R` and fuzzes nearby to find a solution. Returns the list of found solutions.
"""
function fuzz_guesses(xproj::Vector{Float64}, R::Float64, f, Df, Nfuzz=100)
    # Define continuation function
    g(x) = f(x, (R=R,))
    Dg(x) = Df(x, (R=R,))
    Nmodes = length(xproj)

    # Hone in on xproj
    x, X = hookstepsolve(g, xproj, verbose=false)

    xnorm = 0.40

    Xguess = fill(x, 0)
    Xsoln = fill(x, 0)

    for i in 1:Nfuzz
        xguess = randn(Nmodes)
        xguess = xnorm / norm(xguess) * xguess

        xsoln, X = hookstepsolve(g, Dg, xguess, verbose=false)

        if norm(xsoln) > 1e-02 && norm(g(xsoln)) < 1e-07
            push!(Xguess, xguess)
            push!(Xsoln, xsoln)
        end
    end
    if length(Xsoln) > 1
        return qualitatively_unique_vectors(Xsoln)
    end
    return [xproj] # we didn't find anything new
end

###########################################
# VELOCITY FIELD ANALYSIS
###########################################
"""
Returns the power input given a basis and coefficients.
"""
function power_input(basis, coeffs)
    basis_sum = 0
    for (ψ, xi) in zip(basis, coeffs)
        u = ψ.u[1]
        if u.ejx.waveindex == 0 && u.ekz.waveindex == 0
            ∂u_∂y = yderivative(xi * u)
            basis_sum += ∂u_∂y(0, -1, 0) + ∂u_∂y(0, 1, 0) # x, z values don't matter
        end
    end
    1 + 0.5 * basis_sum
end

"""
Returns the curl of a basis function `f`.
"""
function curl(f)
    ∂fx_∂y = yderivative(f.u[1])
    ∂fx_∂z = zderivative(f.u[1])
    ∂fy_∂x = xderivative(f.u[2])
    ∂fy_∂z = zderivative(f.u[2])
    ∂fz_∂x = xderivative(f.u[3])
    ∂fz_∂y = yderivative(f.u[3])
    return BasisFunction(∂fz_∂y - ∂fy_∂z, -∂fx_∂z + ∂fz_∂x, ∂fy_∂x - ∂fx_∂y)
end

"""
One term in the dissipation calculation.
"""
function Dij(Ψi, Ψj)
    ui_y = yderivative(Ψi.u[1])
    ui_z = zderivative(Ψi.u[1])
    vi_x = xderivative(Ψi.u[2])
    vi_z = zderivative(Ψi.u[2])
    wi_x = xderivative(Ψi.u[3])
    wi_y = yderivative(Ψi.u[3])

    uj_y = yderivative(Ψj.u[1])
    uj_z = zderivative(Ψj.u[1])
    vj_x = xderivative(Ψj.u[2])
    vj_z = zderivative(Ψj.u[2])
    wj_x = xderivative(Ψj.u[3])
    wj_y = yderivative(Ψj.u[3])

    term1 = innerproduct(wi_y, wj_y) - innerproduct(wi_y, vj_z) - innerproduct(vi_z, wi_y) + innerproduct(vi_z, vj_z)
    term2 = innerproduct(ui_z, uj_z) - innerproduct(ui_z, wj_x) - innerproduct(wi_x, uj_z) + innerproduct(ui_y, uj_y)
    term3 = innerproduct(vi_x, vj_x) - innerproduct(vi_x, uj_y) - innerproduct(ui_y, vj_x) + innerproduct(ui_y, uj_y)
    return term1 + term2 + term3
end

"""
Returns the dissipation rate of the velocity field represented by a basis and coefficient vector.
"""
function dissipation(basis, coeffs)
    dissipation_sum = 0
    for (Ψi, xi) in zip(basis, coeffs)
        for (Ψj, xj) in zip(basis, coeffs)
            dissipation_sum += (Dij(Ψi, Ψj) * xi * xj)
        end
    end
    return dissipation_sum
end

###########################################
# BIFURCATION KIT USAGE
###########################################
"""
Uses BifurcationKit to continue a starting guess. 
Solves the problem B Xdot = Ax + N(X) for Xdot = 0 (i.e. an eqb).

Arguments:
- Ψ: the basis set
- A: the pre-generated matrix A 
- B: the pre-generated matrix B
- N: the SparseBilinear terms
- initial_guess: the starting point for the bifurcation
- initial_reynolds: the starting Reynolds Number for the first guess

"""
function generate_branch(
    Ψ::Vector{BasisFunction},
    A, B, N,
    initial_guess::Vector{Float64},
    initial_reynolds::Float64;
    α=1, γ=2,
    low=100.0,
    max=500.0,
    tol=1.0e-12,
    dsmin=1.0e-7,
    max_steps=10000,
    recording_function=power_input
)
    # define ODE functions for continuation
    lu_B = lu(B)
    function f(X, p)
        (; R) = p
        return lu_B \ (A(R) * X + N(X))
    end

    function Df(X, p)
        (; R) = p
        return lu_B \ (A(R) + derivative(N, X))
    end

    x = initial_guess

    # automatic bifurcation diagram computation
    parameters = (R=initial_reynolds,)
    problem = BifurcationProblem(
        f,
        x,
        parameters,
        (@optic _.R);
        record_from_solution=(x, p; k...) -> recording_function(Ψ, x)
    )
    opts = ContinuationPar(
        p_max=max,
        p_min=low,
        n_inversion=20,
        max_steps=max_steps,
        newton_options=NewtonPar(tol=tol),
        dsmin=dsmin
    ) # the continuation parameters
    branch1 = continuation(
        problem,
        PALC(), # the algorithm used for continuation: partial arc length in this case
        opts,
        bothside=true,
    )
    return branch1
end

"""
Wrapper for if we don't know A, B, N yet.
"""
function generate_branch(Ψ::Vector{BasisFunction}, initial_guess::Vector{Float64}, initial_reynolds::Float64; α=1, γ=2, low=100.0, max=500.0, tol=1.0e-9, dsmin=1.0e-4, max_steps=10000, recording_function=power_input)
    A, B, N = ode_matrices(Ψ)
    return generate_branch(
        Ψ,
        A, B, N,
        initial_guess,
        initial_reynolds;
        α=α, γ=γ,
        low=low,
        max=max,
        tol=tol,
        dsmin=dsmin,
        max_steps=max_steps,
        recording_function=recording_function
    )
end

"""
Wrapper for if we don't know Ψ yet, just the (J, K, L) tuple.
"""
function generate_branch(jkl::Tuple{Int64,Int64,Int64}, initial_guess::Vector{Float64}, initial_reynolds::Float64; α=1, γ=2, low=100.0, max=500.0, tol=1.0e-6, dsmin=1.0e-4, max_steps=10000, chebyshev=true, recording_function=power_input)
    Ψ = generate_psi(jkl, α, γ, chebyshev)
    return generate_branch(
        Ψ,
        initial_guess,
        initial_reynolds;
        α=α, γ=γ,
        low=low,
        max=max,
        tol=tol,
        dsmin=dsmin,
        max_steps=max_steps,
        recording_function=recording_function
    )
end

"""
Instead of relying on an initial guess, we generate a random selection of possible starting points using fuzzing.
Then, we calculate bifurcations from each and return them all.
"""
function generate_branches(Ψ::Vector{BasisFunction}, A, B, N, initial_guess::Vector{Float64}, initial_reynolds::Float64; α = 1, γ = 2, low = 100.0, max = 500.0, tol = 1.0e-9, dsmin=1.0e-4, max_steps = 10000, recording_function = power_input)
    # define ODE functions for continuation
    function f(X, p)
        (;R) = p
        return B \ (A(R) * X + N(X))
    end
    
    function Df(X, p)
        (;R) = p
        return B \ ( A(R) + derivative(N, X) )
    end

    xs = fuzz_guesses(initial_guess, initial_reynolds, f, Df, 500)
    branches = []

    opts = ContinuationPar(
        p_max = max, 
        p_min = low,
        n_inversion = 20,
        max_steps = max_steps,
        newton_options=NewtonPar(tol = tol),
        dsmin=dsmin
    ) # the continuation parameters
    parameters = (R = initial_reynolds,)

    problem = BifurcationProblem(
        f, 
        xs[1],
        parameters, 
        (@optic _.R);
        record_from_solution = (x,p; k...) -> recording_function(Ψ, x)
    )
    
    for x in xs
        branch_i = bifurcationdiagram(
            re_make(
                problem, 
                u0 = x, 
            ), 
            PALC(), 
            2,
        	opts,
            bothside = true,
        )
        push!(branches, branch_i)
    end
    return branches
end
     
"""
Wrapper for if we only know (J, K, L).
"""
function generate_branches(jkl::Tuple{Int64, Int64, Int64}, initial_guess::Vector{Float64}, initial_reynolds::Float64; α = 1, γ = 2, low = 100.0, max = 500.0, tol = 1.0e-6, dsmin=1.0e-4, max_steps = 10000, recording_function = power_input)
    Ψ = generate_psi(jkl)
    A, B, N = ode_matrices(Ψ)
    return generate_branches(Ψ, A, B, N, initial_guess, initial_reynolds; α=α, γ=γ, low=low, max=max, tol=tol, dsmin=dsmin, max_steps=max_steps, recording_function=recording_function)
end

"""
Helper to generate the α and γ given box lengths.
"""
function get_α_γ(Lx, Lz)
    α = 2π/Lx
    γ = 2π/Lz
    return α, γ
end


end