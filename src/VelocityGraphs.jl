using Revise, Polynomials, BifurcationKit, Plots, DelimitedFiles, Statistics, WGLMakie, LinearAlgebra, Base.Threads
includet("./BasisFunctions.jl")
includet("./Hookstep.jl")
includet("./BifurcationGenerator.jl")
using .BasisFunctions
using .Hookstep
using .BifurcationGenerator
myreaddlm(filename) = readdlm(filename, comments=true, comment_char='%')
plot! = Plots.plot!
plot = Plots.plot

function velocity_fields(Ψ, solution, num_points = 20, Lx = 2π, Lz = π)
    xz_scale = 0.5
    xy_scale = 0.5
    yz_scale = 0.5
    arrow = 7
    lw = 1.0

    # Define velocity functions
    u(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))

    # Generate grid points
    xs = range(0, Lx, num_points)
    ys = range(-1, 1, num_points)
    zs = range(0, Lz, num_points)

    fig = Figure(size=(900, 1200))

    # xz plane: (u, w) at y = 0
    ax1 = Axis(fig[1, 1], title="Mean (u, w) vector field in the xz-plane", xlabel="X", ylabel="Z")
    xz_points = [Point2f(x, z) for x in xs for z in zs]
    # uw_arrows = [(u(x, 0, z) * xz_scale, w(x, 0, z) * xz_scale) for x in xs for z in zs]
    uw_arrows = [(
        mean(u(x, y, z) for y in ys) * xz_scale, 
        mean(w(x, y, z) for y in ys) * xz_scale
    ) for x in xs for z in zs]
    arrows!(ax1, xz_points, uw_arrows, linewidth=lw, arrowsize=arrow)

    # xy plane: (u, v) at z = 0
    ax2 = Axis(fig[2, 1], title="u, v vector field in the xy-plane with z = 0 (left of the box)", xlabel="X", ylabel="Y")
    xy_points = [Point2f(x, y) for x in xs for y in ys]
    uv_arrows = [(u(x, y, 0) * xy_scale, v(x, y, 0) * xy_scale) for x in xs for y in ys]
    arrows!(ax2, xy_points, uv_arrows, linewidth=lw, arrowsize=arrow)

    # yz plane: (v, w) at x = 0, colored by u
    ax3 = Axis(fig[3, 1], title="v, w vector field in the yz-plane with x = 0 (front of the box)", xlabel="Z", ylabel="Y")
    
    # Compute u-values for background heatmap
    u_yz = [u(0, y, z) for z in zs, y in ys]

    cmap = [:navyblue, :aqua, :lime, :orange, :red4]
    # cmap = :roma

    # Plot heatmap of u(0, y, z) in the background
    WGLMakie.heatmap!(ax3, zs, ys, u_yz, colormap=cmap)

    # Overlay v, w quiver plot
    zy_points = [Point2f(z, y) for z in zs for y in ys]
    wv_arrows = [(w(0, y, z) * yz_scale, v(0, y, z) * yz_scale) for y in ys for z in zs]
    arrows!(ax3, zy_points, wv_arrows, linewidth=lw, arrowsize=arrow, color=:black)

    # Add colorbar for reference
    Colorbar(fig[3, 2], colormap=cmap, limits = (-1, 1), label="Streamwise velocity u")

    display(fig)
end

function velocity_fields_read(root="../data/EQ7Re250-32x49x40")
    # File paths
    u_xy_file = root*"_u_xy.asc"
    u_xz_file = root*"_u_xz.asc"
    v_xy_file = root*"_v_xy.asc"
    v_xz_file = root*"_v_xz.asc"
    w_xy_file = root*"_w_xy.asc"
    w_xz_file = root*"_w_xz.asc"
    u_yz_file = root*"_u_yz.asc"
    v_yz_file = root*"_v_yz.asc"
    w_yz_file = root*"_w_yz.asc"

    # Read the data from the files
    u_xy = myreaddlm(u_xy_file)
    u_xz = myreaddlm(u_xz_file)
    v_xy = myreaddlm(v_xy_file)
    v_xz = myreaddlm(v_xz_file)
    w_xy = myreaddlm(w_xy_file)
    w_xz = myreaddlm(w_xz_file)
    u_yz = myreaddlm(u_yz_file)
    v_yz = myreaddlm(v_yz_file)
    w_yz = myreaddlm(w_yz_file)

    scale = 0.5
    
    fig = Figure(size=(900, 1200))

    # xz plane: (u, w) at y = 0
    ax1 = Axis(fig[1, 1], title="(u, w) vector field in the xz-plane", xlabel="X", ylabel="Z")
    nz, nx = size(u_xz)
    xs = range(0, 2π, length=nx)
    zs = range(0,   π, length=nz)
    
    xz_points = [Point2f(x, z) for x in xs, z in zs]  |> vec
    uw_arrows  = [(u_xz[j,i]*scale, w_xz[j,i]*scale) for i in 1:nx, j in 1:nz] |> vec
    arrows!(ax1, xz_points, uw_arrows, linewidth=1.0, arrowsize=7)

    # xy plane: (u, v) at z = 0
    ax2 = Axis(fig[2, 1], title="u, v vector field in the xy-plane with z = 0", xlabel="X", ylabel="Y")
    ny, nx = size(u_xy)
    xs = range(0, 2π, length=nx)
    ys = range(-1, 1, length=ny)

    # build points & arrows in the *same* order
    xy_points = [Point2f(x, y) for y in ys, x in xs] |> vec
    uv_arrows = [(u_xy[j,i]*scale, v_xy[j,i]*scale)
                 for j in 1:ny, i in 1:nx] |> vec
    arrows!(ax2, xy_points, uv_arrows, linewidth=1.0, arrowsize=7)

    # yz plane: (v, w) at x = 0
    ax3 = Axis(fig[3, 1], title="v, w vector field in the yz-plane with x = 0", xlabel="Z", ylabel="Y")

    nz, ny = size(v_yz)
    zs = range(0,   π, length=nz)
    ys = range(-1,  1, length=ny)
    
    # Compute u-values for background heatmap (v and w in yz-plane)
    # u_yz_data = u_yz[1, :]  # Assuming data is along z-axis

    cmap = [:navyblue, :aqua, :lime, :orange, :red4]
    # Plot heatmap of u(0, y, z) in the background
    WGLMakie.heatmap!(ax3, zs, ys, u_yz, colormap=cmap)

    # Overlay v, w quiver plot

    # build points & arrows in the *same* order
    zy_points = [Point2f(z, y) for z in zs, y in ys] |> vec
    wv_arrows = [(w_yz[j,i]*scale, v_yz[j,i]*scale)
                 for j in 1:nz, i in 1:ny] |> vec
    arrows!(ax3, zy_points, wv_arrows, linewidth=1.0, arrowsize=7, color=:black)

    # Add colorbar for reference
    Colorbar(fig[3, 2], colormap=cmap, limits = (-1, 1), label="Streamwise velocity u")

    display(fig)
end

function velocity_fields_comparison(Ψ, solution, root="../data/EQ7Re250-32x49x40")
    xz_scale = 0.5
    xy_scale = 0.5
    yz_scale = 0.5
    scale = 0.5
    arrow = 7
    lw = 1.0

    # Define velocity functions
    u(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))

    # File paths
    u_xy_file = root*"_u_xy.asc"
    u_xz_file = root*"_u_xz.asc"
    v_xy_file = root*"_v_xy.asc"
    v_xz_file = root*"_v_xz.asc"
    w_xy_file = root*"_w_xy.asc"
    w_xz_file = root*"_w_xz.asc"
    u_yz_file = root*"_u_yz.asc"
    v_yz_file = root*"_v_yz.asc"
    w_yz_file = root*"_w_yz.asc"

    # Read the data from the files
    u_xy = myreaddlm(u_xy_file)
    u_xz = myreaddlm(u_xz_file)
    v_xy = myreaddlm(v_xy_file)
    v_xz = myreaddlm(v_xz_file)
    w_xy = myreaddlm(w_xy_file)
    w_xz = myreaddlm(w_xz_file)
    u_yz = myreaddlm(u_yz_file)
    v_yz = myreaddlm(v_yz_file)
    w_yz = myreaddlm(w_yz_file)

    fig = Figure(size=(900, 1200))

    # xz plane: (u, w) at y = 0
    # ax11 = Axis(fig[1, 1], title="ODE (u, w) in the xz-plane", xlabel="X", ylabel="Z")
    # ax12 = Axis(fig[1, 2], title="DNS (u, w) in the xz-plane", xlabel="X", ylabel="Z")
    ax11 = Axis(fig[1,1],
        title  = "ODE (u,w) in xz-plane",
        xlabel = "X", ylabel = "Z",
        aspect = DataAspect(),
    )
    ax12 = Axis(fig[1,2],
        title  = "DNS (u,w) in xz-plane",
        xlabel = "X", ylabel = "Z",
        aspect = DataAspect(),
    )
    
    nz, nx = size(u_xz)
    xs = range(0, 2π, length=nx)
    zs = range(0,   π, length=nz)
    
    xz_points = [Point2f(x, z) for x in xs, z in zs]  |> vec
    
    uw_arrows = [(u(x, 0, z) * xz_scale, w(x, 0, z) * xz_scale) for x in xs for z in zs]
    arrows!(ax11, xz_points, uw_arrows, linewidth=lw, arrowsize=arrow)

    uw_arrows  = [(u_xz[j,i]*scale, w_xz[j,i]*scale) for i in 1:nx, j in 1:nz] |> vec
    arrows!(ax12, xz_points, uw_arrows, linewidth=lw, arrowsize=arrow)

    # xy plane: (u, v) at z = 0
    # ax21 = Axis(fig[2, 1], title="ODE (u, v) in the xy-plane with z = 0", xlabel="X", ylabel="Y")
    # ax22 = Axis(fig[2, 2], title="DNS (u, v) in the xy-plane with z = 0", xlabel="X", ylabel="Y")
    ax21 = Axis(fig[2,1],
        title  = "ODE (u,v) in xy-plane",
        xlabel = "X", ylabel = "Y",
        aspect = DataAspect(),
    )
    ax22 = Axis(fig[2,2],
        title  = "DNS (u,v) in xy-plane",
        xlabel = "X", ylabel = "Y",
        aspect = DataAspect(),
    )

    ny, nx = size(u_xy)
    xs = range(0, 2π, length=nx)
    ys = range(-1, 1, length=ny)

    # build points & arrows in the *same* order
    xy_points = [Point2f(x, y) for y in ys, x in xs] |> vec
    uv_arrows = [(u(x, y, 0) * xy_scale, v(x, y, 0) * xy_scale) for x in xs for y in ys]
    
    arrows!(ax21, xy_points, uv_arrows, linewidth=lw, arrowsize=arrow)
    uv_arrows = [(u_xy[j,i]*scale, v_xy[j,i]*scale)
                 for j in 1:ny, i in 1:nx] |> vec
    arrows!(ax22, xy_points, uv_arrows, linewidth=lw, arrowsize=arrow)
    

    # yz plane: (v, w) at x = 0, colored by u
    # ax31 = Axis(fig[3, 1], title="ODE (v, w) in the yz-plane with x = 0", xlabel="Z", ylabel="Y")
    # ax32 = Axis(fig[3, 2], title="DNS (v, w) in the yz-plane with x = 0", xlabel="Z", ylabel="Y")
    ax31 = Axis(fig[3,1],
        title  = "ODE (v,w) in yz-plane",
        xlabel = "Z", ylabel = "Y",
        aspect = DataAspect(),
    )
    ax32 = Axis(fig[3,2],
        title  = "DNS (v,w) in yz-plane",
        xlabel = "Z", ylabel = "Y",
        aspect = DataAspect(),
    )

    nz, ny = size(v_yz)
    zs = range(0,   π, length=nz)
    ys = range(-1,  1, length=ny)
    # Compute u-values for background heatmap
    u_yz_calculated = [u(0, y, z) for z in zs, y in ys]

    cmap = [:navyblue, :aqua, :lime, :orange, :red4]
    # cmap = :roma

    # Plot heatmap of u(0, y, z) in the background
    WGLMakie.heatmap!(ax31, zs, ys, u_yz_calculated, colormap=cmap)

    # Overlay v, w quiver plot
    # Overlay v, w quiver plot
    

    # build points & arrows in the *same* order
    zy_points = [Point2f(z, y) for z in zs, y in ys] |> vec
    wv_arrows = [(w(0, y, z) * yz_scale, v(0, y, z) * yz_scale) for y in ys for z in zs]
    arrows!(ax31, zy_points, wv_arrows, linewidth=lw, arrowsize=arrow, color=:black)
    
    # yz plane: (v, w) at x = 0
    
    
    # Compute u-values for background heatmap (v and w in yz-plane)
    WGLMakie.heatmap!(ax32, zs, ys, u_yz, colormap=cmap)
    # Overlay v, w quiver plot
    wv_arrows = [(w_yz[j,i]*scale, v_yz[j,i]*scale)
                 for j in 1:nz, i in 1:ny] |> vec
    arrows!(ax32, zy_points, wv_arrows, linewidth=1.0, arrowsize=7, color=:black)
    
    # Add colorbar for reference
    Colorbar(fig[3, 3], colormap=cmap, limits = (-1, 1), label="Streamwise velocity u")
    resize_to_layout!(fig)
    display(fig)
end

"""
Plots the ODE solution defined by Psi and solution on the left, and the DNS solution found at `root` on the right.
"""
function velocity_fields_comparison_save(Ψ, solution, root="../data/EQ7Re250-32x49x40")
    # Parameters
    xz_scale = 0.5
    xy_scale = 0.5
    yz_scale = 0.5
    scale = 0.5
    arrow = 7
    lw = 1.0
    cmap = [:navyblue, :aqua, :lime, :orange, :red4]

    # Define velocity functions
    u(x, y, z) = sum(Ψ[i].u[1](x, y, z) * solution[i] for i in eachindex(solution))
    v(x, y, z) = sum(Ψ[i].u[2](x, y, z) * solution[i] for i in eachindex(solution))
    w(x, y, z) = sum(Ψ[i].u[3](x, y, z) * solution[i] for i in eachindex(solution))

    # File paths
    u_xy_file = root*"_u_xy.asc"
    u_xz_file = root*"_u_xz.asc"
    v_xy_file = root*"_v_xy.asc"
    v_xz_file = root*"_v_xz.asc"
    w_xy_file = root*"_w_xy.asc"
    w_xz_file = root*"_w_xz.asc"
    u_yz_file = root*"_u_yz.asc"
    v_yz_file = root*"_v_yz.asc"
    w_yz_file = root*"_w_yz.asc"

    # Read the data from the files
    u_xy = myreaddlm(u_xy_file)
    u_xz = myreaddlm(u_xz_file)
    v_xy = myreaddlm(v_xy_file)
    v_xz = myreaddlm(v_xz_file)
    w_xy = myreaddlm(w_xy_file)
    w_xz = myreaddlm(w_xz_file)
    u_yz = myreaddlm(u_yz_file)
    v_yz = myreaddlm(v_yz_file)
    w_yz = myreaddlm(w_yz_file)

    # xz plane: (u, w) at y = 0
    # define plotting axes
    fig_xz = Figure(size=(1920, 1080))
    ax11 = Axis(fig_xz[1, 1],
        title="ODE (u, w) in xz-plane",
        xlabel="X", ylabel="Z",
        aspect=DataAspect())
    ax12 = Axis(fig_xz[1, 2],
        title="DNS (u, w) in xz-plane",
        xlabel="X", ylabel="Z",
        aspect=DataAspect())
    
    # define point ranges
    nz, nx = size(u_xz)
    xs = range(0, 2π, length=nx)
    zs = range(0,   π, length=nz)
    xz_points = [Point2f(x, z) for x in xs, z in zs]  |> vec

    # ODE
    uw_ode  = [(u(x,0,z)*scale, w(x,0,z)*scale) for x in xs,   z in zs] |> vec
    arrows!(ax11, xz_points, uw_ode, linewidth=lw, arrowsize=arrow)
    # DNS
    uw_arrows  = [(u_xz[j,i]*scale, w_xz[j,i]*scale) for i in 1:nx, j in 1:nz] |> vec
    arrows!(ax12, xz_points, uw_arrows, linewidth=lw, arrowsize=arrow)

    WGLMakie.save("./images/velocity/"*basename(root)*"xz_comparison.png", fig_xz)


    # xy plane: (u, v) at z = 0
    fig_xy = Figure(size=(1920, 1080))
    ax21 = Axis(fig_xy[1, 1],
        title="ODE (u, v) in xy-plane",
        xlabel="X", ylabel="Y",
        aspect=DataAspect())
    ax22 = Axis(fig_xy[1, 2],
        title="DNS (u, v) in xy-plane",
        xlabel="X", ylabel="Y",
        aspect=DataAspect())

    ny, nx = size(u_xy)
    xs = range(0, 2π, length=nx)
    ys = range(-1, 1, length=ny)
    xy_points = [Point2f(x, y) for y in ys, x in xs] |> vec

    # ODE
    uv_ode = [(u(x,y,0)*xy_scale, v(x,y,0)*xy_scale) for y in ys, x in xs] |> vec    
    arrows!(ax21, xy_points, uv_ode, linewidth=lw, arrowsize=arrow)

    # DNS
    uv_arrows = [(u_xy[j,i]*scale, v_xy[j,i]*scale)
                 for j in 1:ny, i in 1:nx] |> vec
    arrows!(ax22, xy_points, uv_arrows, linewidth=lw, arrowsize=arrow)

    WGLMakie.save("./images/velocity/"*basename(root)*"xy_comparison.png", fig_xy)

    # yz plane: (v, w) at x = 0, colored by u
    fig_yz = Figure(size=(1920, 1080), layout=(1,3))
    ax31 = Axis(fig_yz[1, 1],
        title="ODE (v, w) in yz-plane",
        xlabel="Z", ylabel="Y",
        aspect=DataAspect())
    ax32 = Axis(fig_yz[1, 2],
        title="DNS (v, w) in yz-plane",
        xlabel="Z", ylabel="Y",
        aspect=DataAspect())

    nz, ny = size(v_yz)
    zs = range(0,   π, length=nz)
    ys = range(-1,  1, length=ny)

    # ODE
    # Compute u-values for background heatmap
    u_yz_calculated = [u(0, y, z) for z in zs, y in ys]
    
    # Plot heatmap of u(0, y, z) in the background
    WGLMakie.heatmap!(ax31, zs, ys, u_yz_calculated, colormap=cmap)
    zy_points = [Point2f(z, y) for z in zs, y in ys] |> vec
    wv_ode = [(w(0,y,z)*yz_scale, v(0,y,z)*yz_scale) for z in zs, y in ys] |> vec
    arrows!(ax31, zy_points, wv_ode, linewidth=lw, arrowsize=arrow, color=:black)

    # DNS
    # Compute u-values for background heatmap (v and w in yz-plane)
    WGLMakie.heatmap!(ax32, zs, ys, u_yz, colormap=cmap)
    wv_arrows = [(w_yz[j,i]*scale, v_yz[j,i]*scale)
                 for j in 1:nz, i in 1:ny] |> vec
    arrows!(ax32, zy_points, wv_arrows, linewidth=1.0, arrowsize=7, color=:black)
    
    # Add colorbar for reference
    Colorbar(fig_yz[1, 3], colormap=cmap, limits = (-1, 1), label="Streamwise velocity u")

    WGLMakie.save("./images/velocity/"*basename(root)*"yz_comparison.png", fig_yz)

    println("Saved: xz_comparison.png, xy_comparison.png, yz_comparison.png")    
end

