using ForwardDiff
using LinearAlgebra

function f(x)
    return [-x^5+3x-3]
end

function jacobian(x)
    return ForwardDiff.derivative(f,x)
end

function newton(fcn,jcbn, guess, tol, iterations)
    x = guess
    iter = 0

    while iter < iterations
        val = fcn(x)
        jacob = jcbn(x)

        deltax= -jacob \ val

        x += deltax

        if norm([deltax]) < tol
            break
        end

    iter += 1;
    end

    return x
end

result = newton(f,jacobian,0.01,1e-18,100)