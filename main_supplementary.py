
## Complementary material  

# Library imports
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc
import das4whales as dw
import cv2
import os

plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelpad'] = 2


def plot_gabor_theorical():
    """
    Plot the theoretical Gabor filter with annotations to explain the parameters
    """

    # Define Gabor filter parameters (dummy values)
    sigma = 1.0      # Spatial extent
    fc = 1.5         # Carrier frequency
    gamma = 0.5      # Aspect ratio
    theta = np.pi/4  # Orientation
    amplitude = 1.0

    # Generate grid for the Gabor filter
    x_max, y_max = 3 * sigma, 3 * sigma
    x = np.linspace(-x_max, x_max, 400)
    y = np.linspace(-y_max, y_max, 400)
    x, y = np.meshgrid(x, y)

    # Rotate coordinates
    x_prime = x * np.cos(theta) + y * np.sin(theta)
    y_prime = -x * np.sin(theta) + y * np.cos(theta)

    # Create Gabor filter
    ellipse = np.exp(-(x_prime**2 + gamma**2 * y_prime**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * fc * x_prime)
    gabor = amplitude * ellipse * sinusoid

    # Plot the Gabor filter and annotate parameters
    fig, ax = plt.subplots(figsize=(12, 10))
    c = ax.imshow(gabor, extent=[-3, 3, -3, 3], cmap='RdBu_r', origin='lower', aspect='auto')
    plt.colorbar(c, ax=ax, label='Gain [ ]')

    # Add the ellipse outline
    ellipse_patch = Ellipse((0, 0), width=2*sigma, height=2*sigma/gamma, angle=np.degrees(theta),
                            edgecolor='black', facecolor='none', lw=4, linestyle='--')
    ax.add_patch(ellipse_patch)

    # Annotate parameters
    # ax.arrow(0, 0, sigma * np.cos(theta), sigma * np.sin(theta), color='black',
    #          width=0.05, length_includes_head=True, label='Orientation (θ)')

    # Plot the arc for angle
    arc_radius = np.cos(theta) * sigma
    arc = Arc((0, 0), width=2*arc_radius, height=2*arc_radius, angle=0,
            theta1=0, theta2=np.degrees(theta), color='black', lw=2)
    ax.add_patch(arc)

    # Add horizontal and tilted lines for reference
    line_length = 1.5
    ax.plot([0, line_length], [0, 0], color='black', lw=3, linestyle='--', label='Horizontal Reference')
    ax.plot([0, line_length * np.cos(theta)], [0, line_length * np.sin(theta)],
            color='black', lw=3, linestyle='--', label='Tilted Reference')

    # Add the angle label
    ax.annotate('$\\theta$', (arc_radius * np.cos(theta/2), arc_radius * np.sin(theta/2)),
                textcoords="offset points", xytext=(15, -15), ha='center', color='black')

    # ax.annotate('$\\theta$', (0.5 * sigma * np.cos(theta), 0.5 * sigma * np.sin(theta)),
    #             textcoords="offset points", xytext=(-10, 10), ha='center', color='black')
    # ax.annotate('$f_c$', (0.5, 0), textcoords="offset points", xytext=(5, -15), color='black')
    ax.annotate('$\\gamma$', (-sigma / gamma, sigma / gamma), textcoords="offset points", xytext=(0, -80), color='black')

    # Add double arrows
    translat = 1.8
    ax.annotate('', xy=(-sigma * np.cos(theta) + translat, -sigma * np.sin(theta) - translat), xytext=(sigma * np.cos(theta) +translat, sigma * np.sin(theta) -translat), arrowprops=dict(arrowstyle='<->', color='black', lw=4))
    ax.annotate('$2\\sigma$', (translat, -translat), textcoords="offset points", xytext=(30, -30), ha='center', color='black')

    translat = 0.7
    ax.annotate('', xy=(-1/(2*fc) * np.cos(theta) + translat, -1/(2*fc) * np.sin(theta) - translat), xytext=(1/(2*fc) * np.cos(theta) + translat, 1/(2*fc) * np.sin(theta)- translat), arrowprops=dict(arrowstyle='<->', color='black', lw=4))
    ax.annotate('1/$f_c$', (translat, -translat), textcoords="offset points", xytext=(30, -30), ha='center', color='black')

    # Adjust plot limits and labels
    ax.set_xlim(-x_max, x_max)
    ax.set_ylim(-y_max, y_max)
    ax.set_xlabel('Index / $\\sigma$ [ ]')
    ax.set_ylabel('Index / $\\sigma$ [ ]')
    plt.grid(False)
    plt.show()

    return

def main(urls, selected_channels_m):
    # Plot the theoretical Gabor filter
    plot_gabor_theorical()

    # Download some DAS data
    url = urls[0]
    filepath, filename = dw.data_handle.dl_file(url)

    # Read HDF5 files and access metadata
    # Get the acquisition parameters for the data folder
    metadata = dw.data_handle.get_acquisition_parameters(filepath, interrogator='optasense')
    fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

    print(f'Sampling frequency: {metadata["fs"]} Hz')
    print(f'Channel spacing: {metadata["dx"]} m')
    print(f'Gauge length: {metadata["GL"]} m')
    print(f'File duration: {metadata["ns"] / metadata["fs"]} s')
    print(f'Cable max distance: {metadata["nx"] * metadata["dx"]/1e3:.1f} km')
    print(f'Number of channels: {metadata["nx"]}')
    print(f'Number of time samples: {metadata["ns"]}')


    selected_channels = dw.data_handle.get_selected_channels(selected_channels_m, metadata['dx'])
    tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
    cable_name = os.path.split(os.path.split(filepath)[0])[1]

    # Create the f-k filter
    fk_params = {   # Parameters for the signal
    'c_min': 1400.,
    'c_max': 3300.,
    'fmin': 14.,
    'fmax': 30.
    }

    fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params, display_filter=False)

    # Apply the f-k filter to the data, returns spatio-temporal strain matrix
    trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)

    # Delete the raw data to free memory
    del tr

    # Matched filter
    HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)
    HF_note = np.hanning(len(HF_note)) * HF_note

    nmf_m_HF = dw.detect.calc_nmf_correlogram(trf_fk, HF_note)

    SNR_hf = dw.dsp.snr_tr_array(nmf_m_HF)

    del trf_fk, nmf_m_HF

    # Create the Gabor filters for envelope clustering
    # Detection speed:
    c0 = 1500 # m/s
    theta_c0 = dw.improcess.angle_fromspeed(c0, fs, dx, selected_channels)

    ksize = 100  # Kernel size 
    sigma = 4 # Standard deviation of the Gaussian envelope
    theta = np.pi/2 + np.deg2rad(theta_c0) # Orientation angle (in radians)
    lambd = 20 #theta_c0 * np.pi / 180  # Wavelength of the sinusoidal factor
    gamma = 0.15 # Spatial aspect ratio (controls the ellipticity of the filter)

    print(f'Gabor filter parameters: theta={theta}, lambd={lambd}, sigma={sigma}, ksize={ksize}, gamma={gamma}')

    # Create the Gabor filter
    gabor_filtup = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_64F)
    gabor_filtdown = np.flipud(gabor_filtup)

    # Create custom Gabor filter with hyperbolic carrier
    HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)
    print(len(HF_note))

    # plt.figure(figsize=(8, 8))
    # # plt.subplot(121)
    # plt.imshow(gabor_filtup, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
    # plt.xlabel('Time indices')
    # plt.ylabel('Distance indices')

    # # Print one line of the filter (middle line)
    # plt.figure(figsize=(8, 8))
    # plt.plot(gabor_filtup[ksize//2, :])
    # plt.plot(HF_note)
    # plt.xlabel('Time indices')
    # plt.ylabel('Filter amplitude')

    im = SNR_hf.copy()
    im[im < 0] = 0
    image = dw.improcess.scale_pixels(im) * 255
    imagebin = dw.improcess.binning(image, 1/10, 1/10)

    fimage = cv2.filter2D(imagebin, cv2.CV_64F, gabor_filtup) + cv2.filter2D(imagebin, cv2.CV_64F, gabor_filtdown)
    fimage = dw.improcess.scale_pixels(fimage)

    # Print the image
    plt.figure(figsize=(12, 10))
    plt.imshow(fimage, cmap='viridis', aspect='auto', origin='lower', vmax=1)
    plt.xlabel('Binned time indices [ ]')
    plt.ylabel('Binned distance indices [ ]')
    plt.title(f'$\\sigma=${sigma}, $\\lambda=${lambd}, $\\gamma=${gamma}')
    plt.colorbar(label='Normalized amplitude [ ]')
    plt.tight_layout()

    from matplotlib.colors import ListedColormap

    ## Threshold study
    # Threshold the image, for different threshold values
    thresholds = [0.2, 0.6, 0.4]
    for threshold in thresholds:
        mask = fimage > threshold

        mask_sparse = dw.improcess.binning(mask, 10, 10)
        # Zero padd the mask to be the same size as the original trace
        diff = np.maximum(np.array(im.shape) - np.array(mask_sparse.shape), 0)
        mask_sparse_pad = np.pad(mask_sparse, ((0, diff[0]), (0, diff[1])), mode='edge')

        plt.figure(figsize=(12, 10))
        bin_cmap = ListedColormap(['black', 'white'])
        pmask = plt.imshow(mask_sparse_pad, cmap=bin_cmap, aspect='auto', origin='lower')
        plt.xlabel('Time indices [ ]')
        plt.ylabel('Distance indices [ ]')
        plt.colorbar(pmask, ticks=[0, 1])
        plt.title(f'Threshold={threshold}')
        plt.tight_layout()

            # Apply the mask to the original trace
        masked_tr = dw.improcess.apply_smooth_mask(im, mask_sparse_pad)
        dw.plot.snr_matrix(masked_tr, time, dist, 20)

    ## Sigma study
    sigmas = [2, 4, 6]

    for sigma in sigmas:
        gabor_filtup = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_64F)
        gabor_filtdown = np.flipud(gabor_filtup)

        fimage = cv2.filter2D(imagebin, cv2.CV_64F, gabor_filtup) + cv2.filter2D(imagebin, cv2.CV_64F, gabor_filtdown)
        fimage = dw.improcess.scale_pixels(fimage)

        # Plot filtered image 
        plt.figure(figsize=(12, 10))
        plt.imshow(fimage, cmap='viridis', aspect='auto', origin='lower', vmax=1)
        plt.xlabel('Binned time indices [ ]')
        plt.ylabel('Binned distance indices [ ]')
        plt.title(f'$\\sigma=${sigma}, $\\lambda=${lambd}, $\\gamma=${gamma}')
        plt.colorbar(label='Normalized amplitude [ ]')
        plt.tight_layout()

        # Threshold the image, for threshold = 0.4
        threshold = 0.4
        mask = fimage > threshold

        mask_sparse = dw.improcess.binning(mask, 10, 10)
        # Zero padd the mask to be the same size as the original trace
        diff = np.maximum(np.array(im.shape) - np.array(mask_sparse.shape), 0)
        mask_sparse_pad = np.pad(mask_sparse, ((0, diff[0]), (0, diff[1])), mode='edge')

        plt.figure(figsize=(12, 10))
        bin_cmap = ListedColormap(['black', 'white'])
        pmask = plt.imshow(mask_sparse_pad, cmap=bin_cmap, aspect='auto', origin='lower')
        plt.xlabel('Time indices [ ]')
        plt.ylabel('Distance indices [ ]')
        plt.colorbar(pmask, ticks=[0, 1])
        plt.title(f'Threshold={threshold}')
        plt.tight_layout()
        plt.show()




    return

if __name__ == '__main__':
    # The dataset of this example is constituted of 60s time series along the north and south cables
    url = ['http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
            'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
            'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5']

    selected_channels_m_north = [12000, 66000, 5]  # list of values in meters corresponding to the starting,
                                                    # ending and step wanted channels along the FO Cable
                                                    # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                                    # in meters
    print('test')
    main(url, selected_channels_m_north)


