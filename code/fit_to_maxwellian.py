import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import scipy
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--sim', help='simulation name', required=True)
args = parser.parse_args()

filename = '../data/summary_velocities_{}.dat'.format(args.sim)
all_data = np.loadtxt(filename)
data = all_data[:,0] # this is the peculiar velocity


def cumul_distro(x,a):
    y = scipy.special.erf(x/(a*np.sqrt(2)))
    y = y - np.sqrt(2.0/np.pi) * x * np.exp(-0.5*(x/a)**2)/a
    return y


x_data = np.sort(data)
n = len(x_data)
y_data = np.linspace(1/n, 1.0, n)

popt, pcov = curve_fit(cumul_distro, x_data, y_data, bounds=[0,500])

print(popt[0], np.sqrt(pcov))
y_fit = cumul_distro(x_data, popt[0])

figname = 'plot_fit_{}.jpg'.format(args.sim)
plt.figure()
plt.plot(x_data, y_fit)
plt.plot(x_data, y_data)
plt.savefig(figname)