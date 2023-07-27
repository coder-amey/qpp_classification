import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

import numpy as np

# from mpl_toolkits.axes_grid1 import make_axes_locatable

from waveletFunctions import wave_signif, wavelet


# CONFIG
DT = 1
MOTHER = 'MORLET'
PADDING = 1  # pad the time series with zeroes (recommended)
DJ = 28.0
LAG1 = 0.72  # lag-1 autocorrelation for red noise background

# READ THE DATA & DERIVED PARAMS
detrended_signal = np.loadtxt('detrended.txt')
n = len(detrended_signal)
time = np.arange(len(detrended_signal)) * DT  # construct time array
xlim = ([0, n * DT])  # plotting range
variance = np.std(detrended_signal, ddof=1) ** 2
print("variance = ", variance)

# TRANSFORMATION
# Wavelet transform:
wave, period, scale, coi = wavelet(detrended_signal, DT, pad=PADDING, mother=MOTHER)
power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
global_ws = (np.sum(power, axis=1) / n)  # time-average over all times

# Significance levels:
signif = wave_signif(([variance]), dt=DT, sigtest=0, scale=scale,
    lag1=LAG1, mother=MOTHER)
# expand signif --> (J+1)x(N) array
sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
sig95 = power / sig95  # where ratio > 1, power is significant

# Global wavelet spectrum & significance levels:
dof = n - scale  # the -scale corrects for padding at edges
global_signif = wave_signif(variance, dt=DT, scale=scale, sigtest=1,
    lag1=LAG1, dof=dof, mother=MOTHER)

#PLOTTING
fig = plt.figure(figsize=(9, 10))
gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                    wspace=0, hspace=0)

coi_power = np.copy(power)
for t_step, cap in enumerate(coi):
    coi_power[np.argwhere(period > cap), t_step] = 0
    
print(f"Max power levels: {np.max(coi_power)}")

# Plot the signal 
# Contour plot of wavelet power spectrum
pltx = plt.subplot(gs[0, 0:3])
levels = [0, 0.5, 1, 2, 4, 999]
# *** or use 'contour'
CS = plt.contourf(time, period, coi_power, len(levels))
im = plt.contourf(CS, levels=levels,
    colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (seconds)')
plt.title('c) Wavelet Power Spectrum')
plt.xlim(xlim[:])
# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
# plt.contour(time, period, sig95, [-99, 1], colors='k')
# cone-of-influence, anything "above" is dubious
plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
    edgecolor="#00000040", hatch='x')
plt.plot(time, coi, 'k')
# format y-scale
pltx.set_yscale('log')
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
pltx.ticklabel_format(axis='y', style='plain')

# Plot the detrended signal
plt.subplot(gs[1, 0:3])
plt.plot(time, detrended_signal, 'k')
plt.xlim(xlim[:])
plt.xlabel('Time (seconds)')
plt.ylabel('Flux')
plt.title('b) Detrended flare')


# Contour plot of wavelet power spectrum
plt3 = plt.subplot(gs[2, 0:3])
levels = [0, 0.5, 1, 2, 4, 999]
# *** or use 'contour'
CS = plt.contourf(time, period, power, len(levels))
im = plt.contourf(CS, levels=levels,
    colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency (seconds)')
plt.title('c) Wavelet Power Spectrum')
plt.xlim(xlim[:])
# 95# significance contour, levels at -99 (fake) and 1 (95# signif)
plt.contour(time, period, sig95, [-99, 1], colors='k')
# cone-of-influence, anything "above" is dubious
plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
    edgecolor="#00000040", hatch='x')
plt.plot(time, coi, 'k')
# format y-scale
plt3.set_yscale('log')
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt3.ticklabel_format(axis='y', style='plain')

# --- Plot global wavelet spectrum
plt4 = plt.subplot(gs[2, -1])
plt.plot(global_ws, period)
plt.plot(global_signif, period, '--')
plt.xlabel('Power')
plt.title('d) Global Wavelet Spectrum')
plt.xlim([0, 1.25 * np.max(global_ws)])
# format y-scale
plt4.set_yscale('log')
plt.ylim([np.min(period), np.max(period)])
ax = plt.gca().yaxis
ax.set_major_formatter(ticker.ScalarFormatter())
plt4.ticklabel_format(axis='y', style='plain')

plt.show()