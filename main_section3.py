
# Libraries import
import pandas as pd
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import das4whales as dw

plt.rcParams['font.size'] = 24
plt.rcParams['axes.labelpad'] = 24

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
                trf = dw.dsp.bp_filt(tr, fs, 14, 28)
                cable_name = 'North' 
                dw.plot.plot_tx(sp.hilbert(trf, axis=1), time, dist, f'Bandpass 14-28 Hz, {cable_name}', v_max=0.4)
        # South cable plots
        else:
                # Download the DAS data
                filepaths = []
                filenames = []
                cable_name = 'South'
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
        'c_max': 5000.,
        'fmin': 14.,
        'fmax': 28.
        }

        fk_filter = dw.dsp.hybrid_ninf_gs_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, fk_params=fk_params, display_filter=True)

        # Print the compression ratio given by the sparse matrix usage
        dw.tools.disp_comprate(fk_filter)

        # Apply the f-k filter to the data, returns spatio-temporal strain matrix
        trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)
        dw.plot.plot_fk_domain(tr, fs, dx, selected_channels, fig_size=(12, 10), v_min=0, v_max=0.000025, fk_params=fk_params, ax_lims=[12, 30, 0, 0.025])

        # Delete the raw data to free memory
        del tr

        dw.plot.plot_tx(sp.hilbert(trf_fk, axis=1), time, dist, f'Hybrid 14-28 Hz, 1400-5000 m.s$^{-1}$, {cable_name}', v_max=0.4, fig_size=(12, 10))

        # Plot the SNR
        SNR = dw.dsp.snr_tr_array(trf_fk)
        dw.plot.snr_matrix(SNR, time, dist, 20, "$\\overset{\\sim}{\\text{SNR}}$ estimation, "+f"{cable_name}")

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
