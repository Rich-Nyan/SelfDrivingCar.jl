using ForwardDiff
using LinearAlgebra

function g(x)
    return [x[1]^2+x[2]^2]
end
function grad_descent(fcn, guess, learning_rate, tol, iterations)
    x = guess
    iter = 0

    while iter < iterations
        grad = ForwardDiff.jacobian(fcn,x)

        x -= learning_rate * grad[1,:]

        if norm(grad) < tol
            break
        end

    iter += 1;
    end

    return x
end

grad_descent(g,[2,1],0.0001,1e-18,100000)