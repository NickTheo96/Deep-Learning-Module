import numpy as np
import matplotlib.pyplot as plt  # it is more convenient to type plt than matplotlib.pyplot every time
import math     # these commands allow us to use functions in the numpy and math modules


x_data = np.array([x for x in range(0, 10)])
y_data = 1.1 + 0.6 * x_data + np.random.randn(x_data.size)

plt.plot(x_data, y_data, '.')
plt.show()


def calculate_J(x, y, m, c):
    """
    x, y are ndarrays of the same length
    m and c are floating point numbers giving gradient and intercept of a line
    """
    yhat = m * x + c
    errs = (yhat - y)**2
    J = np.mean(errs)
    return J


m_values = np.linspace(0, 1, 200)
c_values = np.linspace(0, 2, 200)

J_grid = np.zeros([m_values.size, c_values.size])

for m_index in range(0, m_values.size):
    for c_index in range(0, c_values.size):
        J_grid[m_index, c_index] = calculate_J(x_data, y_data, m_values[m_index], c_values[c_index])

plt.imshow(J_grid.transpose(), origin='lower', extent=[0, 1, 0, 2])
plt.colorbar()
plt.show()

plt.imshow(1. / J_grid.transpose(), origin='lower', extent=[0, 1, 0, 2])
plt.colorbar()
plt.show()


def J_gradient(x, y, m, c):
    """
    x, y are ndarrays of the same length
    m and c are floating point numbers giving gradient and intercept of a line

    Returns the gradient J with respect to
    """
    yhat = m * x + c
    c_grads = 2 * (yhat - y)
    m_grads = 2 * (yhat - y) * x
    # now take the means of these, since J is the mean of the square error
    c_grad = np.mean(c_grads)
    m_grad = np.mean(m_grads)
    return m_grad, c_grad


J_gradient(x_data, y_data, 0, 0)
J_gradient(x_data, y_data, 0.6, 1.1)

m = 0.0
c = 0.0

m_path = [m]
c_path = [c]

n_iterations = 200
learning_rate = 0.034

for n in range(1, n_iterations):
    m_grad, c_grad = J_gradient(x_data, y_data, m, c)
    m = m - learning_rate * m_grad
    c = c - learning_rate * c_grad
    m_path.append(m)
    c_path.append(c)

# at this point you may want to inspect m_path and c_path

plt.imshow(1. / J_grid.transpose(), origin='lower',extent=[0,1,0,2])
plt.colorbar()
plt.plot(m_path, c_path, 'r')
plt.plot(m_path, c_path, 'r.')
plt.show()