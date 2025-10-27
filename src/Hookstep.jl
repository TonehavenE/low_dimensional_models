module Hookstep

export hookstepsolve 

using LinearAlgebra

"""
Solve Df * Δx = -f robustly, handling rectangular matrices
"""
function robust_solve(Df, f; λ=1e-8, verbose=false)
    m, n = size(Df)
    
    if verbose
        println("Df size: $m × $n")
        println("rank(Df): $(rank(Df))")
        if m >= n
            DtD = Df' * Df
            println("rank(Df'Df): $(rank(DtD))")
            println("cond(Df'Df): $(cond(DtD))")
        end
    end
    
    if m == n
        # Square system - try direct solve with fallback to regularization
        try
            return -Df \ f
        catch e
            if isa(e, SingularException) || isa(e, LinearAlgebra.LAPACKException)
                # Regularize and retry
                return -(Df' * Df + λ * I) \ (Df' * f)
            else
                rethrow(e)
            end
        end
    elseif m > n
        # Overdetermined: regularized least-squares
        # Solve (Df'*Df + λI) Δx = -Df'*f
        return -(Df' * Df + λ * I) \ (Df' * f)
    else
        # Underdetermined: minimum norm solution with regularization
        return -Df' * ((Df * Df' + λ * I) \ f)
    end
end

"""
hookstep(fx, Dfx, Δx, δ; Nmusearch=10, verbose=true, δtol = 0.01) :

    return hookstep Δx that minimizes 1/2 ||fx + Df Δx||^2 subject to ||Δx|| = δ
"""
function hookstep(fx, Dfx, δ, Δx_newt; δtol = 1e-04, Nmusearch=10, verbose=false)
    
    norm_Δx = norm(Δx_newt)
    if (norm_Δx <= δ)
        verbose && println("Newton step is within trust region, so returning hookstep Δx == Newton step")
        return Δx_newt
    end

    # start Newton iteration to solve norm(Δx(μ)) - δ == 0, with guess μ=0, Δx(μ) = Δx_newt
    μ = 0
    Δx = copy(Δx_newt)
    H = Dfx'*Dfx 
    
    # Add regularization to handle rank-deficient H
    m, n = size(Dfx)
    if m < n
        # Underdetermined: add Tikhonov regularization
        μ = 1e-6  # Start with small regularization
    end
    
    verbose && println("Starting search for hookstep Δx(μ) of radius norm(Δx(μ)) = δ = $δ")
    verbose && println("μ = $(μ), norm(Δx(μ)) = $(norm_Δx)")
    
    for m_iter=1:Nmusearch 
        verbose && println("\nhookstep μ search $m_iter")
           
        ϕ = norm_Δx - δ           
        
        # Use regularized solve
        Hμ = H + μ*I
        try
            Hsolve = Hμ\Δx
            ϕprime = -(1/norm_Δx) * dot(Δx, Hsolve)
            
            verbose && println("  ϕ  = $ϕ")
            verbose && println("  ϕ' = $ϕprime")               
            
            μ = μ - (norm_Δx/δ)*(ϕ/ϕprime)
            μ = max(μ, 1e-12)  # Keep μ positive
            
        catch e
            if isa(e, SingularException)
                verbose && println("  Singular matrix, increasing μ")
                μ = max(10*μ, 1e-6)
            else
                rethrow(e)
            end
        end
        
        # Δx = -(H + μ*I)\(Dfx'*fx)
        Δx = robust_solve(H + μ*I, Dfx'*fx)
        
        norm_Δx = norm(Δx)
        verbose && println("μ = $(μ), norm(Δx(μ)) = $(norm_Δx)")
                     
        if abs(norm_Δx - δ)/δ <= δtol
            verbose && println("Found hookstep Δx with norm(Δx) = $(norm_Δx) ≈ $(δ) = δ.")
            return Δx
        end
    end
    verbose && println("Stopping with hookstep Δx with norm(Δx) = $(norm_Δx) for $(δ) = δ.")
    return Δx
end

# function hookstep(fx, Dfx, δ, Δx_newt; δtol = 1e-04, Nmusearch=10, verbose=false)
    
#     norm_Δx = norm(Δx_newt)
#     if (norm_Δx <= δ)
#         verbose && println("Newton step is within trust region, so returning hookstep Δx == Newton step")
#         return Δx_newt
#     end

#     # start Newton iteration to solve norm(Δx(μ)) - δ == 0, with guess μ=0, Δx(μ) = Δx_newt
#     μ = 0
#     Δx = copy(Δx_newt)
#     H = Dfx'*Dfx 
    
#     verbose && println("Starting search for hookstep Δx(μ) of radius norm(Δx(μ)) = δ = $δ")
#     verbose && println("μ = $(μ), norm(Δx(μ)) = $(norm_Δx)")
    
#     # calculate hookstep Δx of radius δ using Newton iteration over μ on
#     # equation norm(Δx(μ))- δ == 0. Read the algorithm notes above.
    
#     for m=1:Nmusearch 
#         verbose && println("\nhookstep μ search $m")
#         verbose && println("  Δx = $Δx")  
#         ϕ = norm_Δx - δ           
        
#         Hsolve = (H + μ*I)\Δx
#         verbose && println("        H = $(H)")
#         verbose && println("        μ = $μ")
#         verbose && println("  H + μ I = $(H + μ*I)")
#         verbose && println("  Hsolve = $Hsolve")
           
#         ϕ = norm_Δx - δ           
#         #ϕprime = -(1/norm_Δx) * Δx' * ((H + μ*I)\Δx)
#         ϕprime = -(1/norm_Δx) * dot(Δx, (H + μ*I)\Δx)
#         verbose && println("  ϕ  = $ϕ")
#         verbose && println("  ϕ' = $ϕprime")               
        
#         μ = μ - (norm_Δx/δ)*(ϕ/ϕprime)
#         verbose && println(" new μ = $μ")
        
#         Δx = -(H + μ*I)\(Dfx'*fx)
        
#         verbose && println("Dfx'*fx = $(Dfx'*fx)")
#         verbose && println("  Δx = $Δx")
#         norm_Δx = norm(Δx)
#         verbose && println("μ = $(μ), norm(Δx(μ)) = $(norm_Δx)")
                     
#         if abs(norm_Δx - δ)/δ <= δtol
#             verbose && println("Found hookstep Δx with norm(Δx) = $(norm_Δx) ≈ $(δ) = δ.")
#             return Δx
#         end
#     end
#     verbose && println("Stopping with hookstep Δx with norm(Δx) = $(norm_Δx) for $(δ) = δ.")
#     return Δx
# end


function Df_finitediff(f,x; eps=1e-06)
    fx = f(x)
    M = length(fx)
    N = length(x)
    
    Df = zeros(M,N)
    for j=1:N
        dxj = zeros(N)
        dxj[j] = eps
        fx_dxj = f(x + dxj)
        dfdxj = (fx_dxj - fx)/eps
        for i=1:M
            Df[i,j] = dfdxj[i]
        end
    end
    Df
end

function hookstepsolve(f, xguess;  δ=0.1, Nnewton=20, Nhook=4, Nmusearch=6, verbose=true, hookstep_verbose=false)
    hookstepsolve(f, x -> Df_finitediff(f,x), xguess, δ=δ, Nnewton=Nnewton, Nhook=Nhook, Nmusearch=Nmusearch, verbose=verbose, hookstep_verbose=hookstep_verbose)
end


function hookstepsolve(f, Df, xguess; δ=0.1, Nnewton=20, Nhook=4, Nmusearch=6, verbose=true, hookstep_verbose=false)

    ftol = 1e-08
    xtol = 1e-08
    δmax = 1
    δmin = 0.001
    
    #norm2(x) = x'*x
    norm2(x) = dot(x,x)
    
    # x, rx change once per newton step, are constant throughout hookstep calculations
    x = xguess           
    Xiterates = zeros(Nnewton+1, length(x))
    Xiterates[1,:] = x
    
    for n = 1:Nnewton
        verbose && println("\nNewton step $n :")
        verbose && println("x = $x")
        
        # start Newton step from best computed values over all previous computations
        fx = f(x)
        rx = 1/2*norm2(fx)
        
        if norm(fx) < ftol
            verbose && println("Stopping and exiting search since norm(f(x)) = $(norm(fx)) < $ftol = ftol.")
            return x, Xiterates[1:n,:]
        end
        
        # compute Newton step Δx
        Dfx = Df(x)   
        # Δx = -Dfx\fx
        Δx = robust_solve(Dfx, fx)  # Instead of -Dfx\fx
        norm_Δx = norm(Δx)
        DfΔx_newt= Dfx*Δx
        
        verbose && println("Δx newton = $Δx")
        verbose && println("norm(Δx newton) = $(norm(Δx))")
       
        # verify that residual is decreasing in direction of Newton step (it oughta be!)
        #@show  fx
        #@show typeof(fx)
        #@show  fx'
        #@show typeof(fx')
        #@show (Dfx*Δx)
        #@show typeof(Dfx*Δx)
        #@show  fx'*(Dfx*Δx)
        #@show  typeof(fx'*(Dfx*Δx))
        
        #@show  dot(fx, Dfx*Δx)
        if  dot(fx, Dfx*Δx) >= 0 
            verbose && println("Residual is increasing in direction of Newton step, indicating that the")
            verbose && println("solution of the Newton-step equation is inaccurate. Exiting search and")
            verbose && println("returning current value of x")
            Xiterates[n+1,:] = x + Δx
            return x, Xiterates[1:n+1,:]             
        end
             
        if norm(Δx) < xtol
            verbose && println("Stopping because norm(Δx) = $(norm(Δx)) < $(xtol) = xtol")
            Xiterates[n+1, :] = x+Δx
            return x+Δx, Xiterates[1:n+1,:]
        end
                
        Δx_newt = Δx           # Store Newton step and its norm for use in hookstep calculations       
        norm_Δx_newt = norm_Δx 
        x_hook = x + Δx        # Declare x_hook, modify it iteratively in hookstep loop.

        for h in 1:Nhook
            verbose && println("\nHookstep $(h): finding hookstep Δx s.t. |Δx| = δ = $δ")
            Δx = hookstep(fx, Dfx, δ, Δx_newt, Nmusearch=Nmusearch, verbose=hookstep_verbose)
            DfΔx = Dfx*Δx
            #δ = norm(Δx)  # hookstep function returns Δx with norm(Δx) ≈ δ, revise δ to make this exact
            
            if norm(Δx) < xtol
                verbose && println("Stopping search because norm(Δx) = $(norm(Δx)) < $(xtol) = xtol")
                Xiterates[n+1, :] = x_hook
                return x + Δx, Xiterates[1:n+1,:]
            end
                        
            newton_step_within_delta = norm_Δx_newt <= δ ? true : false
            
            # Compute actual (squared) residual of hookstep and linear & quadratic estimates based purely
            # on Δx and evaluations of f(x) and Df(x) at current Newton step. These derive from 
            # r(x + Δx) = 1/2 ||f(x+Δx)||^2
            #           ≈ 1/2 ||f(x) + Df(x) Δx||^2                  (this estimate in quadratic in Δx)
            #           ≈ 1/2 (f(x) + Df(x) Δx)ᵀ (f(x) + Df(x) Δx)
            #           ≈ 1/2 (f(x)ᵀ f(x) + 2 fᵀ Df(x) Δx + (Df(x) Δx)ᵀ (Df(x) Δx))
            #           ≈ r(x) + fᵀ Df(x) Δx + 1/2 (Df(x) Δx)ᵀ (Df(x) Δx) 
            #  
            # r(x + Δx) ≈ r(x) + fᵀ Df(x) Δx  (dropping quadratic terms give estimate linear in Δx)
            x_hook = x + Δx
            r_hook = 1/2*norm2(f(x+Δx))             # actual residual of hookstep, x + Δx
            #r_linear = 1/2*(norm2(fx) + fx'*DfΔx)   # estimate of residual that is linear in Δx
            r_linear = 1/2*(norm2(fx) + dot(fx,DfΔx))   # estimate of residual that is linear in Δx
            r_quadratic = 1/2*norm2(fx + DfΔx)      # estimate of residual that is quadratic in Δx
        
            # Differences in actual, linear, and quadratic estimates with sign set so that 
            # positive Δr == good, and bigger Δr == better. 
            Δr_hook = -(r_hook - rx)
            #Δr_linear = -1/2*(fx'*DfΔx)              # == -(r_linear - rx) without doing the subtraction
            Δr_linear = -1/2*dot(fx,DfΔx)             # == -(r_linear - rx) without doing the subtraction
            Δr_quadratic = -(r_quadratic - rx)
                        
            verbose && println("\nTrust region adjustment for $(h): |Δx| = $δ, r(xₙ+Δx) = $(r_hook), compared to r(xₙ) = $(rx)")
            # revise trust region radius and do or don't recompute hookstep based on 
            # comparisons between actual change in residual and linear & quadratic models
            # note that diff between r_quadratic and r_linear is positive definite, so 0 < Δr_quadratic < Δr_linear
            
            if Δr_hook > Δr_linear                # actual is better than linear estimate (quadratic helps!)
                verbose && print("negative curvature, ")
                if newton_step_within_delta
                    verbose && println("but newton step is within trust region")
                    verbose && println("so don't increase δ, and go to next Newton step")
                    break
                else
                    verbose && println("and newton step is outside trust region")
                    verbose && println("so increase δ = $δ - > 3δ/2 = $(3δ/2) and recompute hookstep")
                    δ = 3δ/2           
                    continue
                end
            elseif Δr_hook < 0.01 * Δr_quadratic
                verbose && println("poor improvement, decreasing δ  = $δ -> δ/2 = $(δ/2) and recomputing hookstep") 
                δ = δ/2                            # actual is vastly worse than quadratic estimate
                continue                           # reduce trust region  and recompute hookstep
            elseif Δr_hook < 0.10 * Δr_quadratic
                verbose && println("marginal improvement, decreasing δ  = $δ -> δ/2 = $(δ/2) and continuing to next Newtown step")
                δ = δ/2                            # actual is worse than quadratic estimate
                break                              # reduce trust region and go to next Newton step
            elseif 0.8*Δr_quadratic < Δr_hook < 2* Δr_quadratic   
                # actual is close to quadratic estimate                
                # revise trust region based on quadratic model of residual and recompute
                rprime = Δr_linear/δ
                rprime2 = 2*(r_hook - rx - rprime*δ)/δ^2 
                δnew = -rprime/rprime2
                verbose && println("accurate improvement, considering δ  = $δ -> δnew = $(δnew) from quadratic model")
                                      
                if newton_step_within_delta
                    verbose && println("but Newton step is within trust region, so don't change δ, and go to next Newton step")
                elseif δnew < 2δ/3
                    verbose && println("too much decrease, changing δ = $δ -> 2δ/3 = $(2δ/3) and continuing to next Newtown step")
                    δ = 2δ/3
                elseif δnew > 4δ3    
                    verbose && println("too much increase, changing δ = $δ -> 4δ/3 = $(4δ3) and continuing to next Newtown step")
                    δ = 4δ/3
                else
                    verbose && println("not too much change, changing δ = $δ -> δnew = $δnew and continuing to next Newtown step")
                    δ = δnew
                end
                break
            else 
                verbose && println("good improvement, keeping δ = $(δ) and continuing to next Newtown step")
                break                              # hookstep is decent enough, don't adjust, go to next Newton               
            end
                    
        end
        
        verbose && println("Finished with hookstep computations. Resetting x to x_hook")
        verbose && println("prior x == $x")
        x = x_hook
        verbose && println("  new x == $x == x_hook")
        Xiterates[n+1,:] = x
    end
    
    println("Stopping because we reached maximum # of Newton iterations")
    return x, Xiterates
end


end # module Hookstep
