using ForwardDiff
using LinearAlgebra

function f(x)
    return [x[1]^2 + x[2]^2,x[1]^2-3x[2]+2]
end

function newton(fcn, guess, tol, iterations)
    x = guess
    iter = 0

    while iter < iterations
        val = fcn(x)
        jacob = ForwardDiff.jacobian(fcn,x)

        delta = -inv(jacob) * val
        x += delta

        if norm(delta) < tol
            break
        end

    iter += 1;
    end

    return x
end

newton(f,[2,1],1e-18,100)

