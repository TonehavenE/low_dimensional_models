using Revise, Polynomials, WGLMakie, LinearAlgebra, DelimitedFiles, Statistics
includet("./BasisFunctions.jl")
includet("./BifurcationGenerator.jl")
using .BasisFunctions
using .BifurcationGenerator

myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')

"""
    velocity_fields(Ψ, solution; num_points=20, Lx=2π, Lz=π, save_path=nothing)

Visualize velocity field from ODE solution across three planes: xz, xy, and yz.
Optionally saves the figure if save_path is provided.
"""
function velocity_fields(Ψ, solution; num_points=20, Lx=2π, Lz=π, save_path=nothing)
    # Scaling factors for arrow lengths
    xz_scale = 0.5
    xy_scale = 0.5
    yz_scale = 0.5
    arrow_size = 7
    line_width = 1.0
    cmap = [:navyblue, :aqua, :lime, :orange, :red4]

    # Define velocity field functions
    u(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))

    # Generate grid points
    xs = range(0, Lx, num_points)
    ys = range(-1, 1, num_points)
    zs = range(0, Lz, num_points)

    fig = Figure(size=(900, 1200))

    # XZ plane: mean (u, w) averaged over y
    ax1 = Axis(fig[1, 1], 
        title="Mean (u, w) in xz-plane", 
        xlabel="X", ylabel="Z",
        aspect=DataAspect())
    xz_points = [Point2f(x, z) for x in xs for z in zs]
    uw_arrows = [(
        mean(u(x, y, z) for y in ys) * xz_scale, 
        mean(w(x, y, z) for y in ys) * xz_scale
    ) for x in xs for z in zs]
    arrows!(ax1, xz_points, uw_arrows, linewidth=line_width, arrowsize=arrow_size)

    # XY plane: (u, v) at z = 0
    ax2 = Axis(fig[2, 1], 
        title="(u, v) in xy-plane at z = 0", 
        xlabel="X", ylabel="Y",
        aspect=DataAspect())
    xy_points = [Point2f(x, y) for x in xs for y in ys]
    uv_arrows = [(u(x, y, 0) * xy_scale, v(x, y, 0) * xy_scale) for x in xs for y in ys]
    arrows!(ax2, xy_points, uv_arrows, linewidth=line_width, arrowsize=arrow_size)

    # YZ plane: (v, w) at x = 0, with u heatmap background
    ax3 = Axis(fig[3, 1], 
        title="(v, w) in yz-plane at x = 0", 
        xlabel="Z", ylabel="Y",
        aspect=DataAspect())
    
    u_yz = [u(0, y, z) for z in zs, y in ys]
    heatmap!(ax3, zs, ys, u_yz, colormap=cmap)
    
    zy_points = [Point2f(z, y) for z in zs for y in ys]
    wv_arrows = [(w(0, y, z) * yz_scale, v(0, y, z) * yz_scale) for z in zs for y in ys]
    arrows!(ax3, zy_points, wv_arrows, linewidth=line_width, arrowsize=arrow_size, color=:black)
    
    Colorbar(fig[3, 2], colormap=cmap, limits=(-1, 1), label="Streamwise velocity u")

    if save_path !== nothing
        WGLMakie.save(save_path, fig)
        println("Saved figure to: $save_path")
    end
    
    display(fig)
    return fig
end

"""
    velocity_fields_dns(root; save_path=nothing)

Visualize velocity field from DNS data files.
"""
function velocity_fields_dns(root; save_path=nothing)
    # Load DNS data
    u_xy = myreaddlm(root * "_u_xy.asc")
    u_xz = myreaddlm(root * "_u_xz.asc")
    v_xy = myreaddlm(root * "_v_xy.asc")
    v_xz = myreaddlm(root * "_v_xz.asc")
    w_xy = myreaddlm(root * "_w_xy.asc")
    w_xz = myreaddlm(root * "_w_xz.asc")
    u_yz = myreaddlm(root * "_u_yz.asc")
    v_yz = myreaddlm(root * "_v_yz.asc")
    w_yz = myreaddlm(root * "_w_yz.asc")

    scale = 0.5
    arrow_size = 7
    line_width = 1.0
    cmap = [:navyblue, :aqua, :lime, :orange, :red4]
    
    fig = Figure(size=(900, 1200))

    # XZ plane
    ax1 = Axis(fig[1, 1], title="DNS (u, w) in xz-plane", xlabel="X", ylabel="Z", aspect=DataAspect())
    nz, nx = size(u_xz)
    xs = range(0, 2π, length=nx)
    zs = range(0, π, length=nz)
    xz_points = [Point2f(x, z) for x in xs, z in zs] |> vec
    uw_arrows = [(u_xz[j,i]*scale, w_xz[j,i]*scale) for i in 1:nx, j in 1:nz] |> vec
    arrows!(ax1, xz_points, uw_arrows, linewidth=line_width, arrowsize=arrow_size)

    # XY plane
    ax2 = Axis(fig[2, 1], title="DNS (u, v) in xy-plane at z = 0", xlabel="X", ylabel="Y", aspect=DataAspect())
    ny, nx = size(u_xy)
    xs = range(0, 2π, length=nx)
    ys = range(-1, 1, length=ny)
    xy_points = [Point2f(x, y) for y in ys, x in xs] |> vec
    uv_arrows = [(u_xy[j,i]*scale, v_xy[j,i]*scale) for j in 1:ny, i in 1:nx] |> vec
    arrows!(ax2, xy_points, uv_arrows, linewidth=line_width, arrowsize=arrow_size)

    # YZ plane
    ax3 = Axis(fig[3, 1], title="DNS (v, w) in yz-plane at x = 0", xlabel="Z", ylabel="Y", aspect=DataAspect())
    nz, ny = size(v_yz)
    zs = range(0, π, length=nz)
    ys = range(-1, 1, length=ny)
    heatmap!(ax3, zs, ys, u_yz, colormap=cmap)
    zy_points = [Point2f(z, y) for z in zs, y in ys] |> vec
    wv_arrows = [(w_yz[j,i]*scale, v_yz[j,i]*scale) for j in 1:nz, i in 1:ny] |> vec
    arrows!(ax3, zy_points, wv_arrows, linewidth=line_width, arrowsize=arrow_size, color=:black)
    Colorbar(fig[3, 2], colormap=cmap, limits=(-1, 1), label="Streamwise velocity u")

    if save_path !== nothing
        WGLMakie.save(save_path, fig)
        println("Saved figure to: $save_path")
    end
    
    display(fig)
    return fig
end

"""
    velocity_fields_comparison(Ψ, solution, root; save_dir=nothing)

Side-by-side comparison of ODE and DNS velocity fields.
Saves three separate figures (xz, xy, yz) if save_dir is provided.
"""
function velocity_fields_comparison(Ψ, solution, root; save_dir=nothing)
    scale = 0.5
    arrow_size = 7
    line_width = 1.0
    cmap = [:navyblue, :aqua, :lime, :orange, :red4]

    # Define ODE velocity functions
    u(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))

    # Load DNS data
    u_xy = myreaddlm(root * "_u_xy.asc")
    u_xz = myreaddlm(root * "_u_xz.asc")
    v_xy = myreaddlm(root * "_v_xy.asc")
    v_xz = myreaddlm(root * "_v_xz.asc")
    w_xy = myreaddlm(root * "_w_xy.asc")
    w_xz = myreaddlm(root * "_w_xz.asc")
    u_yz = myreaddlm(root * "_u_yz.asc")
    v_yz = myreaddlm(root * "_v_yz.asc")
    w_yz = myreaddlm(root * "_w_yz.asc")

    # === XZ PLANE ===
    fig_xz = Figure(size=(1920, 1080))
    ax11 = Axis(fig_xz[1, 1], title="ODE (u, w) in xz-plane", xlabel="X", ylabel="Z", aspect=DataAspect())
    ax12 = Axis(fig_xz[1, 2], title="DNS (u, w) in xz-plane", xlabel="X", ylabel="Z", aspect=DataAspect())
    
    nz, nx = size(u_xz)
    xs = range(0, 2π, length=nx)
    zs = range(0, π, length=nz)
    xz_points = [Point2f(x, z) for x in xs, z in zs] |> vec
    
    uw_ode = [(u(x,0,z)*scale, w(x,0,z)*scale) for x in xs, z in zs] |> vec
    arrows!(ax11, xz_points, uw_ode, linewidth=line_width, arrowsize=arrow_size)
    
    uw_dns = [(u_xz[j,i]*scale, w_xz[j,i]*scale) for i in 1:nx, j in 1:nz] |> vec
    arrows!(ax12, xz_points, uw_dns, linewidth=line_width, arrowsize=arrow_size)

    # === XY PLANE ===
    fig_xy = Figure(size=(1920, 1080))
    ax21 = Axis(fig_xy[1, 1], title="ODE (u, v) in xy-plane", xlabel="X", ylabel="Y", aspect=DataAspect())
    ax22 = Axis(fig_xy[1, 2], title="DNS (u, v) in xy-plane", xlabel="X", ylabel="Y", aspect=DataAspect())

    ny, nx = size(u_xy)
    xs = range(0, 2π, length=nx)
    ys = range(-1, 1, length=ny)
    xy_points = [Point2f(x, y) for y in ys, x in xs] |> vec

    uv_ode = [(u(x,y,0)*scale, v(x,y,0)*scale) for y in ys, x in xs] |> vec    
    arrows!(ax21, xy_points, uv_ode, linewidth=line_width, arrowsize=arrow_size)

    uv_dns = [(u_xy[j,i]*scale, v_xy[j,i]*scale) for j in 1:ny, i in 1:nx] |> vec
    arrows!(ax22, xy_points, uv_dns, linewidth=line_width, arrowsize=arrow_size)

    # === YZ PLANE ===
    fig_yz = Figure(size=(1920, 1080))
    ax31 = Axis(fig_yz[1, 1], title="ODE (v, w) in yz-plane", xlabel="Z", ylabel="Y", aspect=DataAspect())
    ax32 = Axis(fig_yz[1, 2], title="DNS (v, w) in yz-plane", xlabel="Z", ylabel="Y", aspect=DataAspect())

    nz, ny = size(v_yz)
    zs = range(0, π, length=nz)
    ys = range(-1, 1, length=ny)

    u_yz_ode = [u(0, y, z) for z in zs, y in ys]
    heatmap!(ax31, zs, ys, u_yz_ode, colormap=cmap)
    
    zy_points = [Point2f(z, y) for z in zs, y in ys] |> vec
    wv_ode = [(w(0,y,z)*scale, v(0,y,z)*scale) for z in zs, y in ys] |> vec
    arrows!(ax31, zy_points, wv_ode, linewidth=line_width, arrowsize=arrow_size, color=:black)

    heatmap!(ax32, zs, ys, u_yz, colormap=cmap)
    wv_dns = [(w_yz[j,i]*scale, v_yz[j,i]*scale) for j in 1:nz, i in 1:ny] |> vec
    arrows!(ax32, zy_points, wv_dns, linewidth=line_width, arrowsize=arrow_size, color=:black)
    
    Colorbar(fig_yz[1, 3], colormap=cmap, limits=(-1, 1), label="Streamwise velocity u")

    # Save if directory provided
    if save_dir !== nothing
        mkpath(save_dir)
        base_name = basename(root)
        WGLMakie.save(joinpath(save_dir, "$(base_name)_xz_comparison.png"), fig_xz)
        WGLMakie.save(joinpath(save_dir, "$(base_name)_xy_comparison.png"), fig_xy)
        WGLMakie.save(joinpath(save_dir, "$(base_name)_yz_comparison.png"), fig_yz)
        println("Saved comparison figures to: $save_dir")
    end

    display(fig_xz)
    display(fig_xy)
    display(fig_yz)
    
    return (fig_xz, fig_xy, fig_yz)
end

println("Velocity field visualization functions loaded!")
println("\nAvailable functions:")
println("  - velocity_fields(Ψ, solution; save_path=...)")
println("  - velocity_fields_dns(root; save_path=...)")
println("  - velocity_fields_comparison(Ψ, solution, root; save_dir=...)")
