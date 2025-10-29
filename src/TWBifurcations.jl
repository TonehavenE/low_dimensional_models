module TWBifurcations

export tw_branch_continuation, plot_tw_bifurcation, plot_tw_phase_diagram

using Plots, DelimitedFiles, LinearAlgebra, BifurcationKit
include("./BasisFunctions.jl")
include("./BifurcationGenerator.jl")
include("./tw_newton.jl")
include("./Hookstep.jl")

using .BasisFunctions
using .BifurcationGenerator
using .TW_Newton
using .Hookstep

myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')

"""
    tw_branch_continuation(Ψ, A, Cx, Cz, N, x0, cx0, cz0, Re0;
                          Re_min=100.0, Re_max=500.0,
                          max_steps=1000, tol=1e-10, dsmin=1e-6,
                          recording_function=power_input)

Continue a traveling wave solution branch in Reynolds number.
Returns a BifurcationKit continuation result.
"""
function tw_branch_continuation(Ψ, A, Cx, Cz, N, x0, cx0, cz0, Re0;
                                Re_min=100.0, Re_max=500.0,
                                max_steps=1000, tol=1e-10, dsmin=1e-6,
                                recording_function=power_input)
    
    n = length(x0)
    
    # Pack state as [x; cz]
    u0 = vcat(x0, cz0) # State vector is length n+1
    
    # --- Define the fixed phase condition vector ---
    # We use the derivative of the initial guess, Du0 = d(x0)/dz.
    # This vector will be passed in as a parameter.
    Du0 = Cz * x0
    
    # Define residual function (size n+1)
    function f_tw(u, p)
        (; Re, x0_param, Du0_param) = p # Unpack parameters
        x = u[1:n]
        cx = 0.0  # Hard-code cx=0
        cz = u[n+1] # cz is the last element
    
        A_Re = A(Re)
        R = residual(x, cx, cz, A_Re, Cx, Cz, N)
    
        f_full = zeros(eltype(u), n + 1)
        f_full[1:n] = R
        
        # --- NEW PHASE CONDITION ---
        # This is dot(x - x0, Du0) = 0
        # It's non-singular and fixes the phase.
        f_full[n+1] = dot(x - x0_param, Du0_param)
        return f_full
    end

    # Set up bifurcation problem
    # Pass the fixed vectors (x0, Du0) as parameters
    params = (Re = Re0, x0_param = x0, Du0_param = Du0)
    
    prob = BifurcationProblem(
        f_tw, u0, params, (@optic _.Re);
        record_from_solution = (u, p; k...) -> (R=p, x=u[1:n], cz=u[n+1], pow=recording_function(Ψ, u[1:n]))
    )
    
    # Continuation options
    opts = ContinuationPar(
        p_max=Re_max,
        p_min=Re_min,
        n_inversion=20,
        max_steps=max_steps,
        newton_options=NewtonPar(tol=tol, max_iterations=20), # Add max_iterations
        dsmin=dsmin
    )
    
    println("Starting continuation from Re = $Re0...")
    branch = continuation(
        prob,
        PALC(), 
        opts,
        bothside=true,
    )
        
    println("Continuation complete. Found $(length(branch.branch)) points.")
    
    return branch
end

"""
    plot_tw_bifurcation(branches::Vector; 
                        quantity_name="Power Input",
                        title="TW Bifurcation Diagram",
                        labels=nothing,
                        save_path=nothing)

Plot bifurcation diagram from multiple traveling wave branches.
"""
function plot_tw_bifurcation(branches::Vector;
                             quantity_name="Power Input",
                             title="TW Bifurcation Diagram",
                             labels=nothing,
                             save_path=nothing,
                             xlims=nothing,
                             ylims=nothing)
    
    plt = plot(
        xlabel = "Reynolds Number",
        ylabel = quantity_name,
        title = title,
        legend = :best,
        grid = true,
        size = (1000, 700),
        dpi = 300
    )
    
    colors = [:blue, :red, :green, :orange, :purple, :brown, :pink, :gray]
    
    for (i, branch) in enumerate(branches)
        label = labels !== nothing ? labels[i] : "Branch $i"
        color = colors[mod1(i, length(colors))]
        
        plot!(plt, branch.param, branch.pow,
              label = label,
              color = color,
              linewidth = 2.5,
              marker = :circle,
              markersize = 3,
              # alpha = 0.8
        )
    end
    
    if xlims !== nothing
        xlims!(plt, xlims...)
    end
    if ylims !== nothing
        ylims!(plt, ylims...)
    end
    
    if save_path !== nothing
        save(plt, save_path)
        println("Saved bifurcation diagram to: $save_path")
    end
    
    return plt
end

"""
Continue a z-traveling wave (cx=0) solution branch in Reynolds number.
"""
function tw_z_branch_continuation(Ψ, A, Cx, Cz, N, x0, cx0, cz0, Re0;
                                  Re_min=100.0, Re_max=500.0,
                                  max_steps=1000, tol=1e-10, dsmin=1e-6,
                                  recording_function=power_input)
    
    n = length(x0)
    
    # Pack state as [x; cz]
    u0 = vcat(x0, cz0) # State vector is length n+1
    
    # --- Define the fixed phase condition vector ---
    Du0_z = Cz * x0
    
    # Define residual function (size n+1)
    function f_tw_z(u, p)
        (; Re, x0_param, Du0_param) = p # Unpack parameters
        x = u[1:n]
        cx = 0.0  # Hard-code cx=0
        cz = u[n+1] # cz is the last element
    
        A_Re = A(Re)
        R = residual(x, cx, cz, A_Re, Cx, Cz, N)
    
        f_full = zeros(eltype(u), n + 1)
        f_full[1:n] = R
        
        # Phase condition: dot(x - x0, Du0_z) = 0
        f_full[n+1] = dot(x - x0_param, Du0_param)
        return f_full
    end

    # Set up bifurcation problem
    params = (Re = Re0, x0_param = x0, Du0_param = Du0_z)
    
    prob = BifurcationProblem(
        f_tw_z, u0, params, (@optic _.Re);
        record_from_solution = (u, p_val; k...) -> (R=p_val, x=u[1:n], cx=0.0, cz=u[n+1], pow=recording_function(Ψ, u[1:n]))
    )
    
    # Continuation options
    opts = ContinuationPar(
        p_max=Re_max,
        p_min=Re_min,
        n_inversion=20,
        max_steps=max_steps,
        newton_options=NewtonPar(tol=tol, max_iterations=20),
        dsmin=dsmin
    )
    
    println("Starting Z-TW continuation from Re = $Re0...")
    branch = continuation(prob, PALC(), opts, bothside=true)
    println("Continuation complete. Found $(length(branch.branch)) points.")
    
    return branch
end


"""
Continue an x-traveling wave (cz=0) solution branch in Reynolds number.
"""
function tw_x_branch_continuation(Ψ, A, Cx, Cz, N, x0, cx0, cz0, Re0;
                                  Re_min=100.0, Re_max=500.0,
                                  max_steps=1000, tol=1e-10, dsmin=1e-6,
                                  recording_function=power_input)
    
    n = length(x0)
    
    # Pack state as [x; cx]
    u0 = vcat(x0, cx0) # State vector is length n+1
    
    # --- Define the fixed phase condition vector ---
    Du0_x = Cx * x0
    
    # Define residual function (size n+1)
    function f_tw_x(u, p)
        (; Re, x0_param, Du0_param) = p
        x = u[1:n]
        cx = u[n+1] # cx is the last element
        cz = 0.0  # Hard-code cz=0
    
        A_Re = A(Re)
        R = residual(x, cx, cz, A_Re, Cx, Cz, N)
    
        f_full = zeros(eltype(u), n + 1)
        f_full[1:n] = R
        
        # Phase condition: dot(x - x0, Du0_x) = 0
        f_full[n+1] = dot(x - x0_param, Du0_param)
        return f_full
    end

    # Set up bifurcation problem
    params = (Re = Re0, x0_param = x0, Du0_param = Du0_x)
    
    prob = BifurcationProblem(
        f_tw_x, u0, params, (@optic _.Re);
        record_from_solution = (u, p_val; k...) -> (R=p_val, x=u[1:n], cx=u[n+1], cz=0.0, pow=recording_function(Ψ, u[1:n]))
    )
    
    # Continuation options
    opts = ContinuationPar(
        p_max=Re_max,
        p_min=Re_min,
        n_inversion=20,
        max_steps=max_steps,
        newton_options=NewtonPar(tol=tol, max_iterations=20),
        dsmin=dsmin
    )
    
    println("Starting X-TW continuation from Re = $Re0...")
    branch = continuation(prob, PALC(), opts, bothside=true)
    println("Continuation complete. Found $(length(branch.branch)) points.")
    
    return branch
end

"""
Continue a general (xz)-traveling wave solution branch in Reynolds number.
"""
function tw_xz_branch_continuation(Ψ, A, Cx, Cz, N, x0, cx0, cz0, Re0;
                                   Re_min=100.0, Re_max=500.0,
                                   max_steps=1000, tol=1e-10, dsmin=1e-6,
                                   recording_function=power_input)
    
    n = length(x0)
    
    # Pack state as [x; cx; cz]
    u0 = vcat(x0, cx0, cz0) # State vector is length n+2
    
    # --- Define fixed phase condition vectors ---
    Du0_x = Cx * x0
    Du0_z = Cz * x0
    
    # Define residual function (size n+2)
    function f_tw_xz(u, p)
        (; Re, x0_param, Du0x_param, Du0z_param) = p
        x = u[1:n]
        cx = u[n+1]
        cz = u[n+2]
    
        A_Re = A(Re)
        R = residual(x, cx, cz, A_Re, Cx, Cz, N)
    
        f_full = zeros(eltype(u), n + 2)
        f_full[1:n] = R
        
        # Two phase conditions
        f_full[n+1] = dot(x - x0_param, Du0x_param) # x-phase
        f_full[n+2] = dot(x - x0_param, Du0z_param) # z-phase
        return f_full
    end

    # Set up bifurcation problem
    params = (Re = Re0, x0_param = x0, Du0x_param = Du0_x, Du0z_param = Du0_z)
    
    prob = BifurcationProblem(
        f_tw_xz, u0, params, (@optic _.Re);
        record_from_solution = (u, p_val; k...) -> (R=p_val, x=u[1:n], cx=u[n+1], cz=u[n+2], pow=recording_function(Ψ, u[1:n]))
    )
    
    # Continuation options
    opts = ContinuationPar(
        p_max=Re_max,
        p_min=Re_min,
        n_inversion=20,
        max_steps=max_steps,
        newton_options=NewtonPar(tol=tol, max_iterations=20),
        dsmin=dsmin
    )
    
    println("Starting XZ-TW continuation from Re = $Re0...")
    branch = continuation(prob, PALC(), opts, bothside=true)
    println("Continuation complete. Found $(length(branch.branch)) points.")
    
    return branch
end

end