import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab
import math

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(-10, 10, 100)
plt.plot(x, (0.25 * mlab.normpdf(x, 0, 1)) + (0.75 * mlab.normpdf(x, 4, 1)))
plt.show()
