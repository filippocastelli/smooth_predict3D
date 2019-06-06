import matplotlib.pylab as plt
import numpy as np
from scipy.optimize import leastsq

x = np.linspace(start = 0,
                stop = 4,
                num = 100)

def gaussian(x, x0, k):
    
    std = np.sqrt( x0**2 / (8*k))
    a = 2 / (1 - np.exp(-k))
    c = 2 - a
    y = a*np.exp(-((x-(x0/2))**2)/(2*std**2)) + c
    
    return y
    
def model(t, coeffs):
    return gaussian(t, 4, coeffs[0])

def residuals(coeffs, y, t):
    return y - model(t,coeffs)

def spline(t):
    r = []
    y0 = lambda x: x**2
    y1 = lambda x: (x-4)**2
    y2 = lambda x: -(x-2)**2 + 2
    
    for x in t:
        if x < 1:
            r.append(y0(x))
        elif x >= 1 and x <= 3:
            r.append(y2(x))
        elif x > 3:
            r.append(y1(x))
            
    return np.array(r)
                
# y3 = 2.407*np.exp(-((x-2)**2)/(1.5**2)) - 0.407
gaus = gaussian(x = x, x0 = 4, k = 1)

splinex = spline(x)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_ylim([0,2])
# ax.plot(x, splinex)


x0 = np.array([1])
results, flag = leastsq(residuals, x0, args = (splinex,x))

gaus2 = gaussian(x, 4, results[0])

ax.plot(x,gaus2)

x2 = np.linspace(0,100, num = 100)

gaus3 = gaussian(x2, 100, results[0])

plt.plot(x2, gaus3)
