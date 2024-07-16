
# Script to generate the figures of the main text of the paper

# Import the necessary libraries

import das4whales as dw
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import pandas as pd

# matplotlib parameters
plt.rcParams['font.size'] = 20

def main(url_north, url_south):
        ### --- MAPS for Section 1 --- ###

        # Import the cable location
        df_north = pd.read_csv('data/north_DAS_multicoord.csv')
        df_south = pd.read_csv('data/south_DAS_multicoord.csv')

        # Import the bathymetry data
        bathy, xlon, ylat = dw.map.load_bathymetry('data/GMRT_OOI_RCA_Cables.grd')

        # Plot the cables over the bathymetry in 2D
        dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)

        # %matplotlib widget
        dw.map.plot_cables3D(df_north, df_south, bathy, xlon, ylat)


        # # Convert the coordinates to UTM
        # utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
        # utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

        # # Change the reference point to the last point
        # x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
        # xf, yf = utm_xf - utm_xf, utm_yf - utm_y0

        # # Convert the coordinates of the cables to UTM
        # utm_north = dw.map.latlon_to_utm(df_north['lon'], df_north['lat'])
        # utm_south = dw.map.latlon_to_utm(df_south['lon'], df_south['lat'])

        # # Change the reference point to the last point for the cables
        # coord_north = np.vstack((utm_xf - utm_north[0], utm_north[1] - utm_y0))
        # coord_south = np.vstack((utm_xf - utm_south[0], utm_south[1] - utm_y0))

        # # Create vectors of coordinates
        # utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
        # utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
        # x = np.linspace(x0, xf, len(xlon))
        # y = np.linspace(y0, yf, len(ylat))

        # # Put everything in the pandas dataframes
        # df_north['utm_x'] = utm_north[0]
        # df_north['utm_y'] = utm_north[1]
        # df_north['x'] = coord_north[0]
        # df_north['y'] = coord_north[1]
        # df_south['utm_x'] = utm_south[0]
        # df_south['utm_y'] = utm_south[1]
        # df_south['x'] = coord_south[0]
        # df_south['y'] = coord_south[1]


        # # Save the dataframes
        # df_north.to_csv('data/north_DAS_multicoord.csv', index=False)
        # df_south.to_csv('data/south_DAS_multicoord.csv', index=False)


        # dw.map.plot_cables2D(df_north['x'], coord_south, bathy, x, y)
        # dw.map.plot_cables2D(utm_north, utm_south, bathy, utm_x, utm_y)


        dw.map.plot_cables3D_m(df_north, df_south, bathy, x, y)


        plt.close('all')

        # Download some DAS data
        filepath = dw.data_handle.dl_file(url_south)

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


        selected_channels_m = [20000, 86000, 10]  # list of values in meters corresponding to the starting,
                                                # ending and step wanted channels along the FO Cable
                                                # selected_channels_m = [ChannelStart_m, ChannelStop_m, ChannelStep_m]
                                                # in meters

        selected_channels = [int(selected_channels_m // dx) for selected_channels_m in
                        selected_channels_m]  # list of values in channel number (spatial sample) corresponding to the starting, ending and step wanted
                                                # channels along the FO Cable
                                                # selected_channels = [ChannelStart, ChannelStop, ChannelStep] in channel
                                                # numbers

        print('Begin channel #:', selected_channels[0], 
        ', End channel #: ',selected_channels[1], 
        ', step: ',selected_channels[2], 
        'equivalent to ',selected_channels[2]*dx,' m')


        # ### Load raw DAS data
        # 
        # Loads the data using the pre-defined slected channels. 

        tr, time, dist, fileBeginTimeUTC = dw.data_handle.load_das_data(filepath, selected_channels, metadata)

        # Create the f-k filter 
        # includes band-pass filter trf = sp.sosfiltfilt(sos_bpfilter, tr, axis=1) 
        fmin = 14
        fmax = 30
        fk_filter = dw.dsp.hybrid_ninf_filter_design((tr.shape[0],tr.shape[1]), selected_channels, dx, fs, 
                                        cs_min=1350, cp_min=1450, cp_max=2000, cs_max=2450, fmin=fmin, fmax=fmax, display_filter=False)

        # Print the compression ratio given by the sparse matrix usage
        dw.tools.disp_comprate(fk_filter)

        # Apply the f-k filter to the data, returns spatio-temporal strain matrix
        trf_fk = dw.dsp.fk_filter_sparsefilt(tr, fk_filter, tapering=True)
        # Delete the raw data to free memory
        del tr

        dw.plot.plot_tx(trf_fk, time, dist, fileBeginTimeUTC, v_max=0.3)

        # Generate a possible set of arrival times
        # The whale call is at 28.5 km along the cable, with an offset of 0.5 km, at a depth of 30m below the surface 
        # Fin whales vocalize at depths of 15-100m
        t0, pos = 6.47, np.array([45000, 1000, 300])

        # Set the speed of sound
        c0 = 1500

        # Least squares minimization solution
        Nbiter = 10

        ## --- LOCALIZATION --- ### 
        # Set every cable positions 
        X, Y, Z = dist, np.zeros_like(dist), np.zeros_like(dist)
        cable_pos = np.array([X, Y, Z]).T

        # Compute the arrival times
        Ti = dw.loc.calc_arrival_times(t0, cable_pos, pos, c0)

        # Solve the least squares problem 
        n = dw.loc.solve_lq(Ti, cable_pos, c0, fix_z=True)

        print(t0, pos)

        # Plot the results
        th_arrtimes = dw.loc.calc_arrival_times(t0, cable_pos, pos, 1500)
        predic_arrtimes = dw.loc.calc_arrival_times(n[-1], cable_pos, n[:3], 1500)

        plt.figure()

        plt.subplot(121)
        plt.plot(th_arrtimes, dist/1e3, ls='-', lw=3, color='tab:orange', label='Theoretical')
        plt.plot(predic_arrtimes, dist/1e3, ls=':', lw=3, color='tab:blue', label='Predicted (n$_f$)')
        plt.title('Theoretical arrival times')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance [km]')
        plt.ylim([dist[0]/1e3, dist[-1]/1e3])
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(th_arrtimes - predic_arrtimes, dist/1e3, ls=':', lw=3, color='tab:red')
        plt.title('Misfit')
        plt.xlabel('Time (s)')
        plt.ylim([dist[0]/1e3, dist[-1]/1e3])
        plt.grid()

        # Plot the whale position
        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        plt.plot(n[0]/1e3, n[1]/1e3, 'o', label='Predicted Whale position')
        plt.plot(pos[0]/1e3, pos[1]/1e3, 'o', label='True whale position')
        plt.plot(dist/1e3, np.zeros_like(dist), 'k', label='Cable position')
        plt.xlabel('Distance [km]')
        plt.ylabel('Range [km]')
        plt.legend()
        plt.grid()

        plt.subplot(122)
        plt.plot(n[0]/1e3, n[2]/1e3, 'o', label='Predicted Whale depth')
        plt.plot(pos[0]/1e3, pos[2]/1e3 , 'o', label='True whale depth')
        plt.plot(dist/1e3, np.zeros_like(dist), 'k', label='Cable position')
        plt.ylabel('Depth [km]')
        plt.xlabel('Distance [km]')
        plt.grid()
        plt.legend()

        plt.show()



        # Uncertainty estimation
        # Compute the variance of the residuals
        depth_flag = False

        sig_e = dw.loc.cal_variance_residuals(th_arrtimes, predic_arrtimes, fix_z=depth_flag)
        print(f'Variance of the residuals of the arrival times prediction: {sig_e:.2e} s^2')

        # Compute the uncertainties on the whale position
        sig_n = dw.loc.calc_uncertainty_position(cable_pos, n, c0, sig_e, fix_z=depth_flag)

        print(f'Uncertainty on the whale position: dx = {sig_n[0]} m, dy = {sig_n[1]} m, dz = {sig_n[2]} m')

        # localization in the UTM coordinates
        # Least squares minimization solution
        # Generate a possible set of arrival times
        t0, pos = 6.47, np.array([55000, 12000, -60])

        # Set the speed of sound
        c0 = 1500

        # Least squares minimization solution
        Nbiter = 10

        # Set every cable positions 
        X, Y, Z = coord_north[0], coord_north[1], np.zeros_like(coord_north[0])
        cable_pos = np.array([X, Y, Z]).T

        # Compute the arrival times
        Ti = dw.loc.calc_arrival_times(t0, cable_pos, pos, c0)

        # Solve the least squares problem 
        n = dw.loc.solve_lq(Ti, cable_pos, c0, fix_z=True)

        print(t0, pos)


        # Add noise to the arrival times
        Ti_noisy = Ti + np.random.normal(0, 1, Ti.shape)  # 0.1 s of noise
        # Calc x distance from the cable positions
        # actual_dist = np.sqrt((pos[0] - coord_north[0])**2 + (pos[1] - coord_north[1])**2)
        actual_dist = np.sqrt(coord_north[0]**2 + coord_north[1] **2)
        # Plot the noisy arrival times

        plt.figure()
        plt.plot(Ti_noisy, actual_dist/1e3, ls=':', lw=3, color='tab:blue', label='Theoretical + noise')
        plt.plot(Ti, actual_dist/1e3, ls='-', lw=1, color='tab:orange', label='Theoretical')
        plt.title('Theoretical arrival times')
        plt.xlabel('Time (s)')
        plt.ylabel('Distance [km]')
        plt.ylim([actual_dist[0]/1e3, actual_dist[-1]/1e3])
        plt.legend()
        plt.grid()


        # Solve the least squares problem 
        n_noisy = dw.loc.solve_lq(Ti_noisy, cable_pos, c0, fix_z=True)


        # Plot the localization in the UTM coordinates
        # Chose a colormap to be sure that values above 0 are white, and values below 0 are blue
        colors_undersea = plt.cm.Blues_r(np.linspace(0, 0.5, 100)) # blue colors for under the sea
        colors_land = np.array([[1, 1, 1, 1]] * 40)  # white for above zero

        # Combine the color maps
        all_colors = np.vstack((colors_undersea, colors_land))
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)
        extent = [x[0], x[-1], y[0], y[-1]]

        # Set the light source
        ls = LightSource(azdeg=350, altdeg=45)

        plt.figure(figsize=(14, 7))
        ax = plt.gca()
        # Plot the bathymetry relief in background
        rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay')
        plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower')
        # Plot the cable location in 2D
        ax.plot(coord_north[0], coord_north[1], 'tab:red', label='North cable')
        ax.plot(coord_south[0], coord_south[1], 'tab:orange', label='South cable')
        plt.plot(pos[0], pos[1], 'o', color='tab:blue', label='True whale pos')
        plt.plot(n[0], n[1], 'x', color='tab:red', label='Predicted whale pos' )
        plt.plot(n_noisy[0], n_noisy[1], '.', color='tab:green', label='Predicted whale pos, noisy times' )
        # Draw isoline at 0
        ax.contour(bathy, levels=[0], colors='k', extent=extent)
        # Use a proxy artist for the color bar
        im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower')
        # Calculate width of image over height
        im_ratio = bathy.shape[1] / bathy.shape[0]
        plt.colorbar(im, ax=ax, label='Depth [m]', aspect=60, pad=0.1, orientation='horizontal', fraction=0.015 * im_ratio)
        im.remove()
        # Set the labels
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        ax.yaxis.tick_right()
        plt.legend(loc='upper center')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':

        # The dataset of this example is constituted of 60s time series along the north and south cables
        url_north = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/NorthCable/TransmitFiber/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-03T15_06_51-0700/'\
                'North-C1-LR-P1kHz-GL50m-Sp2m-FS200Hz_2021-11-04T020002Z.h5'

        url_south = 'http://piweb.ooirsn.uw.edu/das/data/Optasense/SouthCable/TransmitFiber/'\
                'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-01T16_09_15-0700/'\
                'South-C1-LR-95km-P1kHz-GL50m-SP2m-FS200Hz_2021-11-04T020014Z.h5'

        main(url_north, url_south)