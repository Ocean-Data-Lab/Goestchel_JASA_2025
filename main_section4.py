
# Load and exploit bathymetric data from OOI RCA cables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import das4whales as dw

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

        # Create the f-k filter 
        fk_params = {
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

        if len(urls) == 1:  
            # Plot the spectrogram of the filtered data for a high SNR channel
            # Get the indexes of the maximal value of the data:
            xi_m, tj_m = np.unravel_index(np.argmax(trf_fk, axis=None), trf_fk.shape)
            print(xi_m * dx * selected_channels[2] + selected_channels_m[0], 'm')

            # Spectrogram and strain plot of High SNR channels
            p,tt,ff = dw.dsp.get_spectrogram(trf_fk[xi_m,:], fs, nfft=256, overlap_pct=0.95)
            dw.plot.plot_spectrogram(p, tt,ff, f_min = 10, f_max = 35, v_min=-45)
            dw.plot.plot_3calls(trf_fk[xi_m], time, 6.,27.6, 48.5)

            # Create the matched filters for plots
            tpl_paper = dw.detect.gen_template_fincall(time, fs, fmin = 15., fmax = 30., duration = 2.25, window=False)

            HF_note = dw.detect.gen_template_fincall(time, fs, fmin = 17.8, fmax = 28.8, duration = 0.68)
            LF_note = dw.detect.gen_template_fincall(time, fs, fmin = 14.7, fmax = 21.8, duration = 0.78)
            template = HF_note
            dw.plot.design_mf(trf_fk[xi_m], HF_note, LF_note, 6.17, 28., time, fs)

        # Create the matched filters for detection
        HF_note = dw.detect.gen_hyperbolic_chirp(17.8, 28.8, 0.68, fs)
        HF_note = np.hanning(len(HF_note)) * HF_note

        LF_note = dw.detect.gen_hyperbolic_chirp(14.7, 21.8, 0.78, fs)
        LF_note = np.hanning(len(LF_note)) * LF_note

        # Apply the matched filter to the data 
        nmf_m_HF = dw.detect.calc_nmf_correlogram(trf_fk, HF_note)
        nmf_m_LF = dw.detect.calc_nmf_correlogram(trf_fk, LF_note)

        # Plot the SNR of the matched filter
        SNR_hf = dw.dsp.snr_tr_array(nmf_m_HF)
        SNR_lf = dw.dsp.snr_tr_array(nmf_m_LF)

        dw.plot.snr_matrix(SNR_hf, time, dist, 20, fileBeginTimeUTC, title='mf detect: HF')
        dw.plot.snr_matrix(SNR_lf, time, dist, 20, fileBeginTimeUTC, title ='mf detect: LF')
        return      


if __name__ == '__main__':

        # The dataset of this example is constituted of 60s time series along the north and south cables
        url_north = ['http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5']
        
        selected_channels_m_north = [12000, 66000, 10]  # list of values in meters corresponding to the starting,
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
        
        selected_channels_m_south = [12000, 95000, 10]

        main(url_north, selected_channels_m_north)
        main(url_south, selected_channels_m_south)