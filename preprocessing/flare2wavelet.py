# External imports
import matplotlib.pylab as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import pandas as pd
from scipy.interpolate import splrep, splev

# Local imports
from wavelet_transform.waveletFunctions import wave_signif, wavelet

#TODO: secondary smoothing; old_data x 3; new_data -> (noisy, clean) x 3. predict

# CONFIG
DATA_PATH = "/dcs/large/u2288122/Workspace/qpp_classification/consolidated_data"
S = 40
DT = 1
MOTHER = 'MORLET'
PADDING = 1 # pad the time series with zeroes (recommended)
BUFFER = S # Buffer-signal on the left of peak
DJ = 28.0
LAG1 = 0.72  # lag-1 autocorrelation for red noise background


def running_average_filter(signal, window_size=S):
    assert(np.ndim(signal) == 1)
    n = signal.shape[0]
    assert(n > window_size)
    
    smoothed_signal = np.convolve( \
        np.pad(signal, (window_size//2, window_size-(window_size//2)-1), mode='edge'), \
            np.ones((window_size,))/window_size, mode='valid')
    return smoothed_signal


def flare2wavelet(flare, plot=False):
    # READ THE DATA & DERIVE PARAMS
    n = len(flare)
    time = np.arange(n) * DT  # construct time array
    xlim = ([0, n * DT])  # plotting range
    ylim = ([1, 1000])

    # Find and clip at the flare peak
    flare_peak = np.argmax(flare)
    if flare_peak < BUFFER:
        flare_start = 0
    else:
        flare_start = flare_peak - BUFFER
    
    trend = running_average_filter(np.array(flare[flare_start:]))
    detrended_flare = np.array(flare[flare_start:]) - trend

    detrended_flare = running_average_filter(detrended_flare, window_size=5)

    # Pad the signals with zeros up to original dimensions
    detrended_flare = np.pad(detrended_flare, (n - detrended_flare.shape[0], 0))

    variance = np.std(detrended_flare, ddof=1) ** 2
    # print("variance = ", variance)

    # TRANSFORMATION
    # Wavelet transform:
    wave, period, scale, coi = wavelet(detrended_flare, DT, pad=PADDING, mother=MOTHER)
    power = (np.abs(wave)) ** 2  # compute wavelet power spectrum
    global_ws = (np.sum(power, axis=1) / n)  # time-average over all times
    coi_power = np.copy(power)
    for t_step, cap in enumerate(coi):
        coi_power[np.argwhere(period > cap), t_step] = 0        
    max_power = np.max(power)
    # print(f"Max power levels: {max_power}")

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
    if plot:
        fig = plt.figure(figsize=(9, 12))
        gs = GridSpec(4, 4, hspace=0.4, wspace=0.75)
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.9, top=0.95,
                        wspace=0, hspace=0)
        # Plot the flare
        plt1 = plt.subplot(gs[0, 0:2])
        plt1.plot(time, flare, 'k')
        plt.xlim(xlim[:])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Flux')
        plt.title('a) Raw flare')

        # Plot the flare along with the trend
        plt2 = plt.subplot(gs[0, 2:])
        plt2.plot(time, flare, 'k')
        plt2.plot(time[flare_start::], trend, 'r--')
        plt.xlim(xlim[:])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Flux')
        plt.title('Flare with trendline (clipped)')

        
        # Plot the detrended flare
        plt3 = plt.subplot(gs[1, 0:3])
        plt3.plot(time, detrended_flare, 'k')
        plt.xlim(xlim[:])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Flux')
        plt.title('b) Detrended flare')
        

        # Contour plot of wavelet power spectrum
        plt4 = plt.subplot(gs[2, 0:3])
        if max_power <= 2:
            levels = [0, 0.5, 1, 2]
        else:
            levels = [0, 0.5, 1, max_power / 2, max_power]
        # *** or use 'contour'
        CS = plt4.contourf(time, period, power, len(levels))
        im = plt4.contourf(CS, levels=levels,
            colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
        plt.xlim(xlim[:])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Period (seconds)')
        plt.title('c) Wavelet Power Spectrum')
        
        # 95# significance contour, levels at -99 (fake) and 1 (95# signif)
        plt4.contour(time, period, sig95, [-99, 1], colors='k')
        # cone-of-influence, anything "above" is dubious
        plt4.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
            edgecolor="#00000040", hatch='x')
        plt4.plot(time, coi, 'k')
        # format y-scale
        plt4.set_yscale('log')
        plt.ylim([np.min(period), np.max(period)])
        plt.ylim(bottom=1)
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt.ticklabel_format(axis='y', style='plain')

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
        plt.ylim(bottom=1)
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt.ticklabel_format(axis='y', style='plain')


        # Contour plot of wavelet coi_power spectrum
        plt5 = plt.subplot(gs[3, 0:3])
        if max_power <= 2:
            levels = [0, 0.5, 1, 2]
        else:
            levels = [0, 0.5, 1, max_power / 2, max_power]
        # *** or use 'contour'
        CS = plt.contourf(time, period, coi_power, len(levels))
        im = plt.contourf(CS, levels=levels,
            colors=['white', 'bisque', 'orange', 'orangered', 'darkred'])
        plt.xlabel('Time (seconds)')
        plt.ylabel('Period (seconds)')
        plt.title('e) Wavelet Power Spectrum (COI)')
        plt.xlim(xlim[:])
        # cone-of-influence, anything "above" is dubious
        plt.fill_between(time, coi * 0 + period[-1], coi, facecolor="none",
            edgecolor="#00000040", hatch='x')
        plt.plot(time, coi, 'k')
        # format y-scale
        plt5.set_yscale('log')
        plt.ylim([np.min(period), np.max(period)])
        plt.ylim(bottom=1)
        ax = plt.gca().yaxis
        ax.set_major_formatter(ticker.ScalarFormatter())
        plt5.ticklabel_format(axis='y', style='plain')
        plt.show()
    
    return power.T, coi_power.T
    


if __name__ == "__main__":
    flares_dataset = pd.read_pickle(os.path.join(DATA_PATH, 'flares.pkl'))
    # print(flares_dataset.head)
    # [4, 15, 16, 36, 47, 53, 64, 78, 83, 96, 105, 110, 138, 160, 167, 195, 198, 210]
    # [3, 5, 16, 20, 27, 31, 32, 36, 40, 45, 47, 51, 53, 78, 83, 94, 96, 105, 127, 138, 146, 147, 158, 176, 195, 196, 215]
    for i in [3, 5, 16, 20, 27]:
        print(f"Ground truth: {flares_dataset.iloc[i].y}")
        _, coip = flare2wavelet(flares_dataset.iloc[i].X, plot=True)
        # print(coip.shape)
        # plt.plot(coip, 'x')
        # plt.show()


# NOTES
"""
flare
-> get trend(flare)
-> detrended flare = flare - trend
    detrended
    -> wavelet_transform(detrended_flare)
    -> power, coi
        power / coi_power [num_scales x t_steps]     scales: discrete frequency bands [non-integrals]
        [!] We transpose these values: power / coi_power [t_steps x num_scales]
"""