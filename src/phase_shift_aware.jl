using DelimitedFiles, LinearAlgebra, FFTW, Plots
include("./BasisFunctions.jl")
include("./BifurcationGenerator.jl")
using .BasisFunctions, .BifurcationGenerator

myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')

"""
    find_optimal_phase_shift(data_dns, data_ode)

Find the optimal phase shift between DNS and ODE data in Fourier space.
Returns the shift amounts and aligned ODE data.
"""
function find_optimal_phase_shift(data_dns, data_ode)
    # Use FFT to find optimal shift
    fft_dns = fft(data_dns)
    fft_ode = fft(data_ode)
    
    # Cross-correlation in Fourier space
    cross_corr = ifft(fft_dns .* conj.(fft_ode))
    
    # Find peak of cross-correlation
    max_idx = argmax(abs.(cross_corr))
    shift = Tuple(max_idx) .- 1
    
    # Apply shift to ODE data
    data_ode_shifted = circshift(data_ode, shift)
    
    # Calculate errors
    error_original = norm(data_dns - data_ode) / norm(data_dns) * 100
    error_shifted = norm(data_dns - data_ode_shifted) / norm(data_dns) * 100
    
    return shift, data_ode_shifted, error_original, error_shifted
end

"""
    detailed_plane_comparison(Ψ, solution, root, plane; Lx=2π, Lz=π, check_phase=true)

Detailed comparison including phase shift detection and visualization.
"""
function detailed_plane_comparison(Ψ, solution, root, plane; 
                                  Lx=2π, Lz=π, check_phase=true, save_path=nothing)
    
    # Define ODE velocity functions
    u_ode(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v_ode(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w_ode(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))
    
    println("\n" * "="^70)
    println("Detailed comparison for $plane plane")
    println("="^70)
    
    if plane == "xy"
        # Read DNS data
        u_dns_file = root * "_u_xy.asc"
        v_dns_file = root * "_v_xy.asc"
        u_dns = myreaddlm(u_dns_file)
        v_dns = myreaddlm(v_dns_file)
        
        ny, nx = size(u_dns)
        xs = range(0, Lx, length=nx)
        ys = [cos(j * π / (ny-1)) for j in 0:ny-1]
        
        # Compute ODE on same grid
        u_ode_vals = [u_ode(xs[i], ys[j], 0.0) for j in 1:ny, i in 1:nx]
        v_ode_vals = [v_ode(xs[i], ys[j], 0.0) for j in 1:ny, i in 1:nx]
        
        println("\nComponent u:")
        println("  DNS: min=$(minimum(u_dns)), max=$(maximum(u_dns)), mean=$(mean(u_dns))")
        println("  ODE: min=$(minimum(u_ode_vals)), max=$(maximum(u_ode_vals)), mean=$(mean(u_ode_vals))")
        error_u = norm(u_dns - u_ode_vals) / norm(u_dns) * 100
        println("  Direct L2 error: $(error_u)%")
        
        if check_phase
            shift_u, u_ode_shifted, err_orig_u, err_shift_u = find_optimal_phase_shift(u_dns, u_ode_vals)
            println("  Optimal shift: $shift_u")
            println("  Error after shift: $(err_shift_u)%")
        end
        
        println("\nComponent v:")
        println("  DNS: min=$(minimum(v_dns)), max=$(maximum(v_dns)), mean=$(mean(v_dns))")
        println("  ODE: min=$(minimum(v_ode_vals)), max=$(maximum(v_ode_vals)), mean=$(mean(v_ode_vals))")
        error_v = norm(v_dns - v_ode_vals) / norm(v_dns) * 100
        println("  Direct L2 error: $(error_v)%")
        
        if check_phase
            shift_v, v_ode_shifted, err_orig_v, err_shift_v = find_optimal_phase_shift(v_dns, v_ode_vals)
            println("  Optimal shift: $shift_v")
            println("  Error after shift: $(err_shift_v)%")
        end
        
    elseif plane == "xz"
        u_dns = myreaddlm(root * "_u_xz.asc")
        w_dns = myreaddlm(root * "_w_xz.asc")
        
        nz, nx = size(u_dns)
        xs = range(0, Lx, length=nx)
        zs = range(0, Lz, length=nz)
        
        u_ode_vals = [u_ode(xs[i], 0.0, zs[k]) for k in 1:nz, i in 1:nx]
        w_ode_vals = [w_ode(xs[i], 0.0, zs[k]) for k in 1:nz, i in 1:nx]
        
        println("\nComponent u:")
        println("  DNS: min=$(minimum(u_dns)), max=$(maximum(u_dns)), mean=$(mean(u_dns))")
        println("  ODE: min=$(minimum(u_ode_vals)), max=$(maximum(u_ode_vals)), mean=$(mean(u_ode_vals))")
        error_u = norm(u_dns - u_ode_vals) / norm(u_dns) * 100
        println("  Direct L2 error: $(error_u)%")
        
        if check_phase
            shift_u, u_ode_shifted, err_orig_u, err_shift_u = find_optimal_phase_shift(u_dns, u_ode_vals)
            println("  Optimal shift: $shift_u")
            println("  Error after shift: $(err_shift_u)%")
        end
        
        println("\nComponent w:")
        println("  DNS: min=$(minimum(w_dns)), max=$(maximum(w_dns)), mean=$(mean(w_dns))")
        println("  ODE: min=$(minimum(w_ode_vals)), max=$(maximum(w_ode_vals)), mean=$(mean(w_ode_vals))")
        error_w = norm(w_dns - w_ode_vals) / norm(w_dns) * 100
        println("  Direct L2 error: $(error_w)%")
        
        if check_phase
            shift_w, w_ode_shifted, err_orig_w, err_shift_w = find_optimal_phase_shift(w_dns, w_ode_vals)
            println("  Optimal shift: $shift_w")
            println("  Error after shift: $(err_shift_w)%")
        end
        
    elseif plane == "yz"
        v_dns = myreaddlm(root * "_v_yz.asc")
        w_dns = myreaddlm(root * "_w_yz.asc")
        
        nz, ny = size(v_dns)
        zs = range(0, Lz, length=nz)
        ys = [cos(j * π / (ny-1)) for j in 0:ny-1]
        
        v_ode_vals = [v_ode(0.0, ys[j], zs[k]) for k in 1:nz, j in 1:ny]
        w_ode_vals = [w_ode(0.0, ys[j], zs[k]) for k in 1:nz, j in 1:ny]
        
        println("\nComponent v:")
        println("  DNS: min=$(minimum(v_dns)), max=$(maximum(v_dns)), mean=$(mean(v_dns))")
        println("  ODE: min=$(minimum(v_ode_vals)), max=$(maximum(v_ode_vals)), mean=$(mean(v_ode_vals))")
        error_v = norm(v_dns - v_ode_vals) / norm(v_dns) * 100
        println("  Direct L2 error: $(error_v)%")
        
        if check_phase
            shift_v, v_ode_shifted, err_orig_v, err_shift_v = find_optimal_phase_shift(v_dns, v_ode_vals)
            println("  Optimal shift: $shift_v")
            println("  Error after shift: $(err_shift_v)%")
        end
        
        println("\nComponent w:")
        println("  DNS: min=$(minimum(w_dns)), max=$(maximum(w_dns)), mean=$(mean(w_dns))")
        println("  ODE: min=$(minimum(w_ode_vals)), max=$(maximum(w_ode_vals)), mean=$(mean(w_ode_vals))")
        error_w = norm(w_dns - w_ode_vals) / norm(w_dns) * 100
        println("  Direct L2 error: $(error_w)%")
        
        if check_phase
            shift_w, w_ode_shifted, err_orig_w, err_shift_w = find_optimal_phase_shift(w_dns, w_ode_vals)
            println("  Optimal shift: $shift_w")
            println("  Error after shift: $(err_shift_w)%")
        end
    end
    
    println("="^70)
end

"""
    check_solution_basics(Ψ, solution)

Basic sanity checks on the ODE solution.
"""
function check_solution_basics(Ψ, solution)
    println("\n" * "="^70)
    println("ODE Solution Basics")
    println("="^70)
    
    println("Number of modes: $(length(Ψ))")
    println("Solution vector length: $(length(solution))")
    println("Solution L2 norm: $(norm(solution))")
    println("Solution min: $(minimum(solution)), max: $(maximum(solution))")
    println("Non-zero components: $(count(x -> abs(x) > 1e-10, solution))")
    
    # Check divergence-free
    println("\nChecking divergence-free condition...")
    for i in 1:min(5, length(Ψ))
        # Sample a few random points
        x, y, z = 2π*rand(), 2*(rand()-0.5), π*rand()
        
        # Compute divergence numerically
        ε = 1e-6
        u_f = sum(Ψ[j].u[1](x, y, z) * solution[j] for j in eachindex(solution))
        v_f = sum(Ψ[j].u[2](x, y, z) * solution[j] for j in eachindex(solution))
        w_f = sum(Ψ[j].u[3](x, y, z) * solution[j] for j in eachindex(solution))
        
        u_b = sum(Ψ[j].u[1](x-ε, y, z) * solution[j] for j in eachindex(solution))
        v_t = sum(Ψ[j].u[2](x, y+ε, z) * solution[j] for j in eachindex(solution))
        v_b = sum(Ψ[j].u[2](x, y-ε, z) * solution[j] for j in eachindex(solution))
        w_f2 = sum(Ψ[j].u[3](x, y, z+ε) * solution[j] for j in eachindex(solution))
        w_b = sum(Ψ[j].u[3](x, y, z-ε) * solution[j] for j in eachindex(solution))
        
        div = (u_f - u_b)/ε + (v_t - v_b)/(2ε) + (w_f2 - w_b)/ε
        
        if i <= 3
            println("  Point $i: div = $(div)")
        end
    end
    
    # Check power input
    power = power_input(Ψ, solution)
    println("\nPower input: $(power)")
    
    # Check dissipation  
    diss = dissipation(Ψ, solution)
    println("Dissipation: $(diss)")
    println("Power - Dissipation: $(power - diss)")
    
    println("="^70)
end

println("Phase-shift aware comparison tools loaded!")
println("\nUsage:")
println("  check_solution_basics(Ψ, solution)")
println("  detailed_plane_comparison(Ψ, solution, root, \"xy\")")
println("  detailed_plane_comparison(Ψ, solution, root, \"xz\")")
println("  detailed_plane_comparison(Ψ, solution, root, \"yz\")")