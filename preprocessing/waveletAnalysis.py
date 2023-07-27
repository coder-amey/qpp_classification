# External imports
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np

# Local imports
from waveletFunctions import wavelet

def wavelet_transform_fn(flux_array):
    # Find and clip peak
    # Alter dt
    # Try without padding
    # Wavelet transform:
    wave, period, scale, coi = wavelet(flux_array, dt=1, pad=1, mother='MORLET')
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
    return wave, power, global_ws, period

def plot_flare(flux_array):
    # --- Plot time series
    # [!] As-is
    wave, power, global_ws, period = wavelet_transform_fn(flux_array)
    time = np.arange(len(flux_array)) * dt + 1871.0  # construct time array
    xlim = ([1870, 2000])  # plotting range
    fig = plt.figure(figsize=(9, 10))
    gs = GridSpec(3, 4, hspace=0.4, wspace=0.75)
    plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                        wspace=0, hspace=0)
    plt.subplot(gs[0, 0:3])
    plt.plot(time, flux_array, 'k')
    plt.xlim(xlim[:])
    plt.xlabel('Time')
    plt.ylabel('Flux')
    plt.title('Flare')

    # --- Contour plot wavelet power spectrum
    # plt3 = plt.subplot(3, 1, 2)
    plt3 = plt.subplot(gs[1, 0:3])
    levels = [0, 0.5, 1, 2, 4, 999]
    # *** or use 'contour'
    CS = plt.contourf(time, period, power, len(levels))
    im = plt.contourf(CS, levels=levels,
        colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
    plt.xlabel('Time')
    plt.ylabel('Period')
    plt.title('Wavelet Power Spectrum')
    plt.xlim(xlim[:])
    
    # --- Plot global wavelet spectrum
    plt4 = plt.subplot(gs[1, -1])
    plt.plot(global_ws, period)
    plt.xlabel('Power')
    plt.title('Global Wavelet Spectrum')
    plt.xlim([0, 1.25 * np.max(global_ws)])
    # format y-scale
    plt4.set_yscale('log', base=2, subs=None)
    plt.ylim([np.min(period), np.max(period)])
    ax = plt.gca().yaxis
    ax.set_major_formatter(ticker.ScalarFormatter())
    plt4.ticklabel_format(axis='y', style='plain')
    plt4.invert_yaxis()

    plt.show()
