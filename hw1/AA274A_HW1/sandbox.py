from __future__ import division # Python 2.7
import math

import numpy as np
from matplotlib import ticker as tck
from matplotlib import pyplot as plt


def wrapToPi(a):
    if isinstance(a, list):    # backwards compatibility for lists (distinct from np.array)
        return [(x + np.pi) % (2*np.pi) - np.pi for x in a]
    return (a + np.pi) % (2*np.pi) - np.pi


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    # From: https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
    # Retrieved: September 2020
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a
    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den==1:
            if num==0:
                return r'$0$'
            if num==1:
                return r'$%s$'%latex
            elif num==-1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num==1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num==-1:
                return r'$\frac{-%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)
    return _multiple_formatter



if __name__ == '__main__':

    theta = np.linspace(0, 2*np.pi, 360)
    x = np.cos(theta)
    y = np.sin(theta)
    tan_theta = y/x

    atan1 = np.arctan(y/x)
    atan2 = np.arctan2(y, x)

    subplots = []
    for i in range(1,5):
        subplot = plt.subplot(2,2,i)
        subplots.append(subplot)

    plt.subplot(2,2,1)
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.subplot(2,2,2)
    plt.plot(theta, tan_theta)
    plt.xlabel('theta')
    plt.ylabel('tan(theta)')

    plt.subplot(2,2,3)
    plt.plot(theta, atan1)
    plt.xlabel('theta')
    plt.ylabel('np.arctan(y/x)')

    plt.subplot(2,2,4)
    plt.plot(theta, atan2)
    plt.xlabel('theta')
    plt.ylabel('np.arctan2(y,x)')


    for subplot in subplots[1:]:
        subplot.grid(True)
        subplot.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        subplot.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
        subplot.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    for subplot in subplots[2:]:
        subplot.set_ylim([-np.pi -0.5, np.pi + 0.5])
        subplot.yaxis.set_major_locator(plt.MultipleLocator(np.pi / 2))
        subplot.yaxis.set_minor_locator(plt.MultipleLocator(np.pi / 4))
        subplot.yaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))

    plt.subplots_adjust(wspace=0.4)
    plt.show()