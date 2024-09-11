
# Libraries import
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import das4whales as dw
import cv2
import gc

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelpad'] = 20


def main(urls, selected_channels_m):
        # North cable plots
        if len(urls) == 1: 
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


                # ### Select the desired channels and channel interval

                selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                                selected_channels_m]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                                        # channels along the FO Cable
                                                        # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                                        # numbers

                print('Begin channel #:', selected_channels[0], 
                ', End channel #: ',selected_channels[1], 
                ', step: ',selected_channels[2], 
                'equivalent to ',selected_channels[2]*dx,' m')


                ### Load raw DAS data
                
                # Loads the data using the pre-defined selected channels. 

                tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)
        # South cable plots
        else:
                # Download the DAS data
                filepaths = []
                filenames = []
                for url in urls:
                        print(url)
                        filepath, filename = dw.data_handle.dl_file(url)
                        filepaths.append(filepath)
                        filenames.append(filename)

                metadata = dw.data_handle.get_acquisition_parameters(filepaths[0], interrogator='optasense')
                fs, dx, nx, ns, gauge_length, scale_factor = metadata["fs"], metadata["dx"], metadata["nx"], metadata["ns"], metadata["GL"], metadata["scale_factor"]

                selected_channels = dw.data_handle.get_selected_channels(selected_channels_m, dx)

                timestamp = '2021-11-04 02:00:02.025000'
                duration = 60
                selected_channels = dw.data_handle.get_selected_channels(selected_channels_m, dx)

                # Load the data
                tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_mtpl_das_data(filepaths, selected_channels, metadata, timestamp, duration)

        # Create the f-k filters
        fk_params = {   # Parameters for the signal
        'c_min': 1400.,
        'c_max': 3300.,
        'fmin': 14.,
        'fmax': 30.
        }

        fk_params_n = {   # Parameters for the noise
        'c_min': 1400.,
        'c_max': 3300.,
        'fmin': 32.,
        'fmax': 48.
        }


        fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params, display_filter=False)
        fk_filter_noise = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params_n, display_filter=False)

        # Apply the f-k filter to the data, returns spatio-temporal strain matrix
        trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)
        noise = dw.dsp.fk_filter_sparsefilt(tr, fk_filter_noise, tapering=True)

        SNR_noise = dw.dsp.snr_tr_array(noise)
        dw.plot.snr_matrix(SNR_noise, time, dist, 20, fileBeginTimeUTC, title='Noise field')

        noise = dw.dsp.normalize_std(noise)
        window_size = 100
        noise = dw.dsp.moving_average_matrix(abs(sp.hilbert(noise, axis=1)), window_size)

        # Delete the raw data to free memory
        del tr

        # Create the matched filters for detection
        HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)
        HF_note = np.hanning(len(HF_note)) * HF_note

        LF_note = dw.detect.gen_hyperbolic_chirp(14.7, 21.8, 0.78, fs)
        LF_note = np.hanning(len(LF_note)) * LF_note

        # Apply the matched filter to the data 
        nmf_m_HF = dw.detect.calc_nmf_correlogram(trf_fk, HF_note)
        nmf_m_LF = dw.detect.calc_nmf_correlogram(trf_fk, LF_note)

        # Normalize the matched filtered traces
        nmf_m_HF = dw.dsp.normalize_std(nmf_m_HF)
        nmf_m_LF = dw.dsp.normalize_std(nmf_m_LF)

        # Free memory
        del trf_fk
        gc.collect()

        # Plot the SNR of the matched filter
        SNR_hf = 20 * np.log10(abs(sp.hilbert(nmf_m_HF, axis=1)) / abs(sp.hilbert(noise, axis=1)))
        SNR_lf = 20 * np.log10(abs(sp.hilbert(nmf_m_LF, axis=1)) / abs(sp.hilbert(noise, axis=1)))

        SNR_hf = cv2.GaussianBlur(SNR_hf, (9, 73), 0)
        SNR_lf = cv2.GaussianBlur(SNR_lf, (9, 73), 0)

        # Free memory
        del nmf_m_HF, nmf_m_LF
        gc.collect()

        dw.plot.snr_matrix(SNR_hf, time, dist, 20, fileBeginTimeUTC, title='mf detect: HF')
        dw.plot.snr_matrix(SNR_lf, time, dist, 20, fileBeginTimeUTC, title ='mf detect: LF')

        # Create the Gabor filters for envelope clustering
        # Detection speed:
        c0 = 1500 # m/s
        theta_c0 = dw.improcess.angle_fromspeed(c0, fs, dx, selected_channels)

        gabfilt_up, gabfilt_down = dw.improcess.gabor_filt_design(theta_c0, plot=True)

        # Smooth image:
        images = [SNR_hf, SNR_lf]
        labels = ['HF', 'LF']
        for i,im in enumerate(images):
            im[im < 0] = 0
            image = dw.improcess.scale_pixels(im) * 255
            imagebin = dw.improcess.binning(image, 1/10, 1/10)

            fimage = cv2.filter2D(imagebin, cv2.CV_64F, gabfilt_up) + cv2.filter2D(imagebin, cv2.CV_64F, gabfilt_down)

            fimage = dw.improcess.scale_pixels(fimage)

            # Threshold the image
            threshold = 0.4
            mask = fimage > threshold

            mask_sparse = dw.improcess.binning(mask, 10, 10)
            # Zero padd the mask to be the same size as the original trace
            diff = np.maximum(np.array(im.shape) - np.array(mask_sparse.shape), 0)
            mask_sparse_pad = np.pad(mask_sparse, ((0, diff[0]), (0, diff[1])), mode='edge')

            # Apply the mask to the original trace
            masked_tr = dw.improcess.apply_smooth_mask(im, mask_sparse_pad)
            dw.plot.snr_matrix(masked_tr, time, dist, 20, fileBeginTimeUTC, title=f'mf detect: {labels[i]}')

        return      


if __name__ == '__main__':

        # The dataset of this example is constituted of 60s time series along the north and south cables
        url_north = ['http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5']
        
        selected_channels_m_north = [12000, 66000, 5]  # list of values in meters corresponding to the starting,
                                                        # ending and step wanted channels along the FO Cable
                                                        # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                                        # in meters

        url_south = [
        'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T015914Z.h5',
        'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'\
        'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T020014Z.h5'
        ]         
        
        selected_channels_m_south = [12000, 95000, 5]

        main(url_north, selected_channels_m_north)
        gc.collect()
        main(url_south, selected_channels_m_south)