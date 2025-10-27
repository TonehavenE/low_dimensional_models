module TW_Newton

export residual, jacobian_residual, save_sigma, newton_hookstep_bordered!, verify, newton_bordered!

using LinearAlgebra
using SparseArrays
using DelimitedFiles
myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')

using Revise
include("BasisFunctions.jl")
include("Hookstep.jl")
include("BifurcationGenerator.jl")

using .BasisFunctions
using .BifurcationGenerator
using .Hookstep

### 
# Fundamentals
###

function residual(x, cx, cz, A, Cx, Cz, N)
    return (cx * Cx + cz * Cz) * x - A * x - N(x)
end

"""
Jacobian of residual with respect to x (J).
"""
function jacobian_residual(x, cx, cz, A, Cx, Cz, N)
    J = cx * Cx + cz * Cz - A - derivative(N, x)
    return J
end

"""
Helper function to save symmetries in format read by Channelflow.
"""
function save_sigma(cx, cz, T, file_name) 
    file = open(file_name, "w") # Open in write mode
    az = cz ≈ 0 ? 0 : round(-cz * T, sigdigits=6)
    ax = cx ≈ 0 ? 0 : round(-cx * T, sigdigits=6)
    write(file, "% 1\n1 1 1 1 $(ax) $(az)")
    close(file)
end

### 
# Hookstep 
###

"""
The function we are trying to optimize
"""
function f_bordered(u, n, A, Cx, Cz, N, keep_cx, keep_cz)
    x = vec(u[1:n])
    cx = u[n+1]
    cz = u[n+2]
    
    R = vec(residual(x, cx, cz, A, Cx, Cz, N))
    
    # Build full residual vector
    m = n + (keep_cx ? 1 : 0) + (keep_cz ? 1 : 0)
    f_full = zeros(m)
    f_full[1:n] = R
    
    idx = n
    if keep_cx
        idx += 1
        f_full[idx] = dot(Cx * x, x)
    end
    if keep_cz
        idx += 1
        f_full[idx] = dot(Cz * x, x)
    end
    
    return f_full
end

"""
Bordered Jacobian matrix
"""
function Df_bordered(u, n, A, Cx, Cz, N, keep_cx, keep_cz)
    x = u[1:n]
    cx = u[n+1]
    cz = u[n+2]
    
    Jx = jacobian_residual(x, cx, cz, A, Cx, Cz, N)
    cxx = Cx * x
    czx = Cz * x
    
    # Build bordered Jacobian
    m_rows = n + (keep_cx ? 1 : 0) + (keep_cz ? 1 : 0)
    m_cols = n + 2  # always have cx, cz
    
    M = zeros(m_rows, m_cols)
    M[1:n, 1:n] = Jx
    M[1:n, n+1] = cxx
    M[1:n, n+2] = czx
    
    row = n
    if keep_cx
        row += 1
        M[row, 1:n] = 2 * cxx'  # derivative of x'*Cx*x w.r.t. x
        M[row, n+1] = 0.0
        M[row, n+2] = 0.0
    end
    if keep_cz
        row += 1
        M[row, 1:n] = 2 * czx'
        M[row, n+1] = 0.0
        M[row, n+2] = 0.0
    end
    
    return M
end

function newton_hookstep_bordered!(x, cx, cz; A, Cx, Cz, N,
                                   δ=0.1, Nnewton=20, Nhook=4, 
                                   tol_phase=1e-12, verbose=true)
    n = length(x)
    u_guess = vcat(x, [cx, cz])
    
    # Determine which constraints to keep
    cxx = Cx * x
    czx = Cz * x
    keep_cx = norm(cxx) > tol_phase
    keep_cz = norm(czx) > tol_phase
    
    # Define closures for f and Df
    f_closure(u) = f_bordered(u, n, A, Cx, Cz, N, keep_cx, keep_cz)
    Df_closure(u) = Df_bordered(u, n, A, Cx, Cz, N, keep_cx, keep_cz)
    
    # Call hookstep solver
    u_final, Xiterates = hookstepsolve(f_closure, Df_closure, u_guess, 
                                       δ=δ, Nnewton=Nnewton, Nhook=Nhook, 
                                       verbose=verbose)
    
    return u_final[1:n], u_final[n+1], u_final[n+2], Xiterates # x, cx, cz, state iterates
end

### 
# Timestepping Verification
### 
"""
    time_evolution_rhs!(dx, x, p, t)

Computes the right-hand-side for the time-dependent system, dx/dt = Af*x + Nf(x).
This is the "lab frame" equation of motion before transforming to the moving frame.
"""
function time_evolution_rhs!(dx, x, p, t)
    # Unpack parameters. Here, 'p' is a tuple containing Af and Nf.
    Af_mat, Nf_func = p
    
    # Calculate the time derivative
    dx .= Af_mat * x + Nf_func(x)
end


"""
    rk4_step!(dx, x, F, p, t, dt)

Performs a single step of the classical 4th-order Runge-Kutta method.
Updates the state vector `x` in-place.
"""
function rk4_step!(dx, x, F, p, t, dt)
    # k1 = F(t, x)
    k1 = similar(x)
    F(k1, x, p, t)

    # k2 = F(t + dt/2, x + dt/2 * k1)
    k2 = similar(x)
    F(k2, x .+ (dt/2) .* k1, p, t + dt/2)

    # k3 = F(t + dt/2, x + dt/2 * k2)
    k3 = similar(x)
    F(k3, x .+ (dt/2) .* k2, p, t + dt/2)

    # k4 = F(t + dt, x + dt * k3)
    k4 = similar(x)
    F(k4, x .+ dt .* k3, p, t + dt)

    # Update x
    x .+= (dt/6).* (k1 .+ 2 .* k2 .+ 2 .* k3.+ k4)
end


"""
    apply_translation(x0, cx, cz, Cxf, Czf, T)

Applies the exact analytical translation to the state vector `x0`.
The translation operator exp(T * (cx*d/dx + cz*d/dz)) is computed
via the matrix exponential of the generator (cx*Cxf + cz*Czf).
"""
function apply_translation(x0, cx, cz, Cxf, Czf, T)
    # Construct the generator of translation in the direction (cx, cz)
    translation_generator = cx * Cxf + cz * Czf
    
    # The translation operator is the exponential of the generator * time
    # This is the matrix equivalent of the Fourier shift theorem.
    translation_operator = exp(T * translation_generator)
    
    return translation_operator * x0
end

"""
Verifies whether x_sol, cx_sol, cz_sol behaves like a traveling wave.
Continues the solution for `T` seconds in `dt` intervals, and then sees 
if it is properly a shift of the initial conditions.
"""
function verify(x_sol, cx_sol, cz_sol, matrices; T = 5.0, dt = 0.001)
    Af, Cxf, Czf, Nf = matrices 
    num_steps = ceil(Int, T / dt)
    T_actual = num_steps * dt # The actual final time
    
    # Time Evolution
    u = copy(x_sol) # Create a mutable copy for the simulation
    du = similar(u)
    # Pack parameters into a tuple to pass to the RHS function
    params = (Af, Nf) 
    
    println("Starting time evolution...")
    for i in 1:num_steps
        current_time = (i-1) * dt
        rk4_step!(du, u, time_evolution_rhs!, params, current_time, dt)
    end
    println("Evolution finished at T = $T_actual")
    x_final = u
    
    # 4. Verification
    # Apply the analytical shift to the *initial* state
    println("Calculating analytically shifted state...")
    x_sol_shifted = apply_translation(x_sol, cx_sol, cz_sol, Cxf, Czf, T_actual)
    
    # 5. Compare and Report Error
    # Calculate the L2 norm of the difference between the simulated final state
    # and the analytically shifted initial state.
    error_norm = norm(x_final - x_sol_shifted) / sqrt(length(x_sol))
    println("\n--- Verification Results ---")
    println("L2 Error between simulated and analytically shifted wave: ", error_norm)
    
    if error_norm < 1e-4 # A reasonable tolerance for numerical error
        println("Result: The solution behaves like a true traveling wave.")
        return true
    else
        println("Result: The solution does NOT behave like a true traveling wave. The profile may be unstable or the solver/integrator may have significant errors.")
        return false
    end
end

# verify(u; T = 5.0, dt = 0.001) = verify(u[1:end-2], u[end-1], u[end]; T=T, dt = dt) # example usage

###
# Rudimentary Newton
### 
function newton_bordered!(x, cx, cz; A, Cx, Cz, N,
                          maxiter=30, tol=1e-12, tol_phase=1e-12, verbose=true)
    n = length(x)
    u = vcat(x, cx, cz)
    for k in 1:maxiter
        # evaluate residual and Jacobian
        R = residual(u[1:n], u[n+1], u[n+2], A, Cx, Cz, N)
        rnorm = norm(R)
        if verbose; println("Newton iter $k: ||R|| = $rnorm"); end
        if rnorm < tol
            power = power_input(Ψ, u[1:n])
            println("Converged to a solution with:\nL2 Norm: $(norm(u[1:n]))\nPower Input: $(power)")
            return u, true
        end
        Jx = jacobian_residual(u[1:n], u[n+1], u[n+2], A, Cx, Cz, N) 
        # Jx should be n×n
        cxx = Cx * u[1:n]
        czx = Cz * u[1:n]

        keep_cx = norm(cxx) > tol_phase
        keep_cz = norm(czx) > tol_phase

        m = n + (keep_cx ? 1 : 0) + (keep_cz ? 1 : 0)
        M = zeros(m,m)
        rhs = zeros(m)
    
        # top-left block
        M[1:n, 1:n] = Jx
        rhs[1:n] = -R
    
        col = n
        if keep_cx
            col += 1
            M[1:n, col] = cxx          # ∂r/∂c_x
            M[col, 1:n] = cxx'         # phase constraint row
            rhs[col] = 0.0
        end
        if keep_cz
            col += 1
            M[1:n, col] = czx
            M[col, 1:n] = czx'
            rhs[col] = 0.0
        end
        
        # solve
        Δ = M \ rhs
        Δx = Δ[1:n]
        k = n
        Δcx = keep_cx ? Δ[k+1] : 0.0
        k += keep_cx ? 1 : 0
        Δcz = keep_cz ? Δ[k+1] : 0.0
        Δu = [Δx; Δcx; Δcz]

        # line search / damping (backtracking) to ensure residual reduces
        alpha = 1.0
        for ls in 1:12
            candidate = u + alpha * Δu
            # candidate = u + alpha * [Δx Δcx Δcz]
            Rnew = residual(candidate[1:n], candidate[n+1], candidate[n+2], A, Cx, Cz, N)
            if norm(Rnew) < (1 - 1e-4*alpha)*rnorm || alpha < 1e-6
                u = candidate
                break
            else
                alpha *= 0.5
            end
        end
    end
    println("Failed to converge")
    return u, false
end

end