using Optim
using LinearAlgebra
using ForwardDiff
using Plots
# Starts at time stamp 0

# x
# [1,5,9,...,4T-3]: x 
# [2,6,10,...,4T-2]: y 
# [3,7,11,...,4T-1]: theta 
# [4,8,12,...,4T]: velocity 

# u
# [1,3,5,...,2T-1]: theta
# [2,4,6,...,2T]: velocity

# Dynamics
# jacob : gets the derivative wrt x axis, y axis, angle, and velocity
# x : holds x,y,θ,v at that specific timestamp (state vector)
# u : theta and velocity

# Values
T = 100
dt = 0.1
initial_pose = [0,0,0,0]
final_pose = [1,0,0,0]

z_guess = ones(6 * T)
λ_guess = ones(4 * T + 4)

iterations = [1,2,3,4,5,10]
iter = length(iterations)

function dynamics!(jacob, x, u)
    jacob[1] = cos(x[3]) * x[4]
    jacob[2] = sin(x[3]) * x[4]
    jacob[3] = u[1]
    jacob[4] = u[2]
end

function objective_function(z)
    return norm(z[4*T+1:end])
end
# Constraints h(z)
# h(z)
# T vectors of length 4 (x)
# T vectors of length 2 (u)
function dynamic_feasibility(z)
    x = Vector{eltype(z)}(undef,4 * T)
    u = Vector{eltype(z)}(undef,2 * T)
    h = Vector{eltype(z)}(undef, 4 * T + 4)
    
    x = z[1:4*T]
    u = z[4*T+1:end]
    # Initial Pose 
    h[1] = x[1] - (initial_pose[1] + dt * (cos(initial_pose[3]) * initial_pose[4]))
    h[2] = x[2] - (initial_pose[2] + dt * (sin(initial_pose[3]) * initial_pose[4]))
    h[3] = x[3] - (initial_pose[3] + dt * u[1])
    h[4] = x[4] - (initial_pose[4] + dt * u[2])
    # Intermediate Poses
    for i in 1:T-1
        cos_theta = cos(x[4*i-1])
        sin_theta = sin(x[4*i-1])

        h[4*i+1] = x[4*i+1] - (x[4*i-3] + dt * (cos_theta * x[4*i]))
        h[4*i+2] = x[4*i+2] - (x[4*i-2] + dt * (sin_theta * x[4*i]))
        h[4*i+3] = x[4*i+3] - (x[4*i-1] + dt * u[2*i+1])
        h[4*i+4] = x[4*i+4] - (x[4*i] + dt * u[2*i+2])
    end
    # Final Pose
    h[4*T+1] = x[4*T-3] - final_pose[1]
    h[4*T+2] = x[4*T-2] - final_pose[2]
    h[4*T+3] = x[4*T-1] - final_pose[3]
    h[4*T+4] = x[4*T] - final_pose[4]
    return h
end

# Lagrange Multiplier
function lagrangian(z, h_z)
    f_z = objective_function(z)
    return f_z + dot(λ_guess, h_z)
end

# Newton
function newton(fcn, guess, tol, iterations)
    x = guess
    iter = 0

    while iter < iterations
        val = fcn(x)
        jacob = ForwardDiff.jacobian(fcn,x)

        delta = -pinv(jacob) * val
        x += delta

        if norm(delta) < tol
            break
        end

    iter += 1;
    end

    return x
end

#Optimize Function
function optimizer()

    function objective(z)
        h_z = dynamic_feasibility(z)
        lagrange_gradient = ForwardDiff.gradient(z -> lagrangian(z, h_z), z)
        return vcat(lagrange_gradient,h_z)
    end
    states = zeros(6,T,4)
    for i in 1:iter
        result = newton(objective, z_guess, 1e-6, iterations[i])
        for j in 1:T
            states[i,j,1] = result[4*j-3]
            states[i,j,2] = result[4*j-2]
            states[i,j,3] = result[4*j-1]
            states[i,j,4] = result[4*j]
        end
    end

    open("trajectory/test2.txt", "w") do io
        for i in 1:iter
            for j in 1:T
                println(io, states[i, j, 1], ",", states[i, j, 2], ",", states[i, j, 3], ",", states[i, j, 4])
            end
        end
    end

    return states
end

# Setup
optimizer()


