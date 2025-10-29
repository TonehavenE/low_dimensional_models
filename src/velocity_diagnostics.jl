using DelimitedFiles, LinearAlgebra
myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')

"""
    diagnose_velocity_data(root)

Diagnose the dimensions and structure of velocity field data files.
"""
function diagnose_velocity_data(root)
    println("=== Diagnosing velocity data: $root ===\n")
    
    files = ["_u_xy.asc", "_u_xz.asc", "_u_yz.asc",
             "_v_xy.asc", "_v_xz.asc", "_v_yz.asc",
             "_w_xy.asc", "_w_xz.asc", "_w_yz.asc"]
    
    for file in files
        path = root * file
        if isfile(path)
            data = myreaddlm(path)
            println("$file: size = $(size(data))")
            println("  min = $(minimum(data)), max = $(maximum(data))")
            println("  mean = $(mean(data)), std = $(std(data))")
            println()
        else
            println("$file: NOT FOUND")
        end
    end
    
    println("=== Expected dimensions ===")
    println("For Lx=2π, Lz=π domain:")
    println("  XY plane (z fixed): rows=Ny (y-direction), cols=Nx (x-direction)")
    println("  XZ plane (y fixed): rows=Nz (z-direction), cols=Nx (x-direction)")
    println("  YZ plane (x fixed): rows=Nz (z-direction), cols=Ny (y-direction)")
    println("\nNote: y uses Chebyshev points cos(ny*π/(Ny-1))")
end

"""
    read_velocity_plane(root, component, plane)

Read velocity component from a plane slice.
Returns properly oriented matrix and coordinate ranges.

Arguments:
- root: base path without extension
- component: 'u', 'v', or 'w'
- plane: 'xy', 'xz', or 'yz'

Returns:
- data: 2D array (properly oriented)
- coords: tuple of coordinate ranges (for plotting)
"""
function read_velocity_plane(root, component, plane; Lx=2π, Lz=π)
    file = root * "_$(component)_$(plane).asc"
    data = myreaddlm(file)
    
    if plane == "xy"
        # XY plane: data[j, i] where i=x index, j=y index
        # rows = Ny (y-direction), cols = Nx (x-direction)
        ny, nx = size(data)
        xs = range(0, Lx, length=nx)
        # Chebyshev points: y = cos(j*π/(Ny-1)) for j=0:Ny-1
        ys = [cos(j * π / (ny-1)) for j in 0:ny-1]
        # Note: Chebyshev points go from 1 to -1, may need to reverse
        return data, (xs, ys)
        
    elseif plane == "xz"
        # XZ plane: data[k, i] where i=x index, k=z index  
        # rows = Nz (z-direction), cols = Nx (x-direction)
        nz, nx = size(data)
        xs = range(0, Lx, length=nx)
        zs = range(0, Lz, length=nz)
        return data, (xs, zs)
        
    elseif plane == "yz"
        # YZ plane: data[k, j] where j=y index, k=z index
        # rows = Nz (z-direction), cols = Ny (y-direction)
        nz, ny = size(data)
        zs = range(0, Lz, length=nz)
        ys = [cos(j * π / (ny-1)) for j in 0:ny-1]
        return data, (zs, ys)
        
    else
        error("Unknown plane: $plane. Must be 'xy', 'xz', or 'yz'")
    end
end

"""
    compare_ode_dns_single_plane(Ψ, solution, root, plane;
                                 Lx=2π, Lz=π, num_points=nothing)

Compare ODE and DNS velocity fields on a single plane with proper indexing.
"""
function compare_ode_dns_single_plane(Ψ, solution, root, plane;
                                     Lx=2π, Lz=π, num_points=nothing)
    
    # Define ODE velocity functions
    u_ode(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v_ode(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w_ode(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))
    
    # Read DNS data
    u_dns, coords_u = read_velocity_plane(root, 'u', plane, Lx=Lx, Lz=Lz)
    v_dns, coords_v = read_velocity_plane(root, 'v', plane, Lx=Lx, Lz=Lz)
    w_dns, coords_w = read_velocity_plane(root, 'w', plane, Lx=Lx, Lz=Lz)
    
    if plane == "xy"
        # XY plane at z=0
        xs, ys = coords_u
        nx, ny = length(xs), length(ys)
        
        # Use DNS grid or subsample
        if num_points !== nothing
            xs = range(0, Lx, length=num_points)
            ys = [cos(j * π / (num_points-1)) for j in 0:num_points-1]
            nx, ny = num_points, num_points
        end
        
        # Compute ODE velocities on the same grid
        u_ode_vals = [u_ode(xs[i], ys[j], 0.0) for j in 1:ny, i in 1:nx]
        v_ode_vals = [v_ode(xs[i], ys[j], 0.0) for j in 1:ny, i in 1:nx]
        
        println("XY plane comparison at z=0:")
        println("  DNS u: size=$(size(u_dns)), range=[$(minimum(u_dns)), $(maximum(u_dns))]")
        println("  ODE u: size=$(size(u_ode_vals)), range=[$(minimum(u_ode_vals)), $(maximum(u_ode_vals))]")
        println("  L2 error (u): $(norm(u_dns - u_ode_vals) / norm(u_dns) * 100)%")
        println("  DNS v: size=$(size(v_dns)), range=[$(minimum(v_dns)), $(maximum(v_dns))]")
        println("  ODE v: size=$(size(v_ode_vals)), range=[$(minimum(v_ode_vals)), $(maximum(v_ode_vals))]")
        println("  L2 error (v): $(norm(v_dns - v_ode_vals) / norm(v_dns) * 100)%")
        
    elseif plane == "xz"
        # XZ plane at y=0
        xs, zs = coords_u
        nx, nz = length(xs), length(zs)
        
        if num_points !== nothing
            xs = range(0, Lx, length=num_points)
            zs = range(0, Lz, length=num_points)
            nx, nz = num_points, num_points
        end
        
        # ODE: data[k, i] for z-index k, x-index i
        u_ode_vals = [u_ode(xs[i], 0.0, zs[k]) for k in 1:nz, i in 1:nx]
        w_ode_vals = [w_ode(xs[i], 0.0, zs[k]) for k in 1:nz, i in 1:nx]
        
        println("XZ plane comparison at y=0:")
        println("  DNS u: size=$(size(u_dns)), range=[$(minimum(u_dns)), $(maximum(u_dns))]")
        println("  ODE u: size=$(size(u_ode_vals)), range=[$(minimum(u_ode_vals)), $(maximum(u_ode_vals))]")
        println("  L2 error (u): $(norm(u_dns - u_ode_vals) / norm(u_dns) * 100)%")
        println("  DNS w: size=$(size(w_dns)), range=[$(minimum(w_dns)), $(maximum(w_dns))]")
        println("  ODE w: size=$(size(w_ode_vals)), range=[$(minimum(w_ode_vals)), $(maximum(w_ode_vals))]")
        println("  L2 error (w): $(norm(w_dns - w_ode_vals) / norm(w_dns) * 100)%")
        
    elseif plane == "yz"
        # YZ plane at x=0
        zs, ys = coords_u
        nz, ny = length(zs), length(ys)
        
        if num_points !== nothing
            zs = range(0, Lz, length=num_points)
            ys = [cos(j * π / (num_points-1)) for j in 0:num_points-1]
            nz, ny = num_points, num_points
        end
        
        # ODE: data[k, j] for z-index k, y-index j
        v_ode_vals = [v_ode(0.0, ys[j], zs[k]) for k in 1:nz, j in 1:ny]
        w_ode_vals = [w_ode(0.0, ys[j], zs[k]) for k in 1:nz, j in 1:ny]
        
        println("YZ plane comparison at x=0:")
        println("  DNS v: size=$(size(v_dns)), range=[$(minimum(v_dns)), $(maximum(v_dns))]")
        println("  ODE v: size=$(size(v_ode_vals)), range=[$(minimum(v_ode_vals)), $(maximum(v_ode_vals))]")
        println("  L2 error (v): $(norm(v_dns - v_ode_vals) / norm(v_dns) * 100)%")
        println("  DNS w: size=$(size(w_dns)), range=[$(minimum(w_dns)), $(maximum(w_dns))]")
        println("  ODE w: size=$(size(w_ode_vals)), range=[$(minimum(w_ode_vals)), $(maximum(w_ode_vals))]")
        println("  L2 error (w): $(norm(w_dns - w_ode_vals) / norm(w_dns) * 100)%")
    end
end

println("Diagnostic functions loaded!")
println("\nUsage:")
println("  1. diagnose_velocity_data(\"../data/EQ7Re250-32x49x40\")")
println("  2. compare_ode_dns_single_plane(Ψ, solution, \"../data/EQ7Re250-32x49x40\", \"xy\")")
println("  3. compare_ode_dns_single_plane(Ψ, solution, \"../data/EQ7Re250-32x49x40\", \"xz\")")
println("  4. compare_ode_dns_single_plane(Ψ, solution, \"../data/EQ7Re250-32x49x40\", \"yz\")")