using ForwardDiff
using LinearAlgebra

function f(x,y)
    return [x^4+3x+3y^2+4x-3]
end

function jacobian(x,y)
    return ForwardDiff.jacobian(f,[x,y])
end

function newton(fcn,jcbn, guess, tol, iterations)
    x = guess
    y = guess
    iter = 0

    while iter < iterations
        val = fcn(x,y)
        jacob = jcbn(x,y)

        deltax, deltay = -jacob \ val

        x += deltax
        y += deltay

        if norm([deltax,deltay]) < tol
            break
        end

    iter += 1;
    end

    return x,y
end

result = newton(f,jacobian,0.01,1e-18,100)