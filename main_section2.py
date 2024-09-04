
# Load and exploit bathymetric data from OOI RCA cables

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import das4whales as dw

plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelpad'] = 20


def main():
    # Plot cable positions
    # Import the cable location
    df_north = pd.read_csv('data/north_DAS_multicoord.csv')
    df_south = pd.read_csv('data/south_DAS_multicoord.csv')

    # Import the bathymetry data
    bathy, xlon, ylat = dw.map.load_bathymetry('data/GMRT_OOI_RCA_Cables.grd')

    # Plot the cables geometry in lat/lon coordinates
    dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)
    dw.map.plot_cables3D(df_north, df_south, bathy, xlon, ylat)

    utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
    utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

    # Change the reference point to the last point
    x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
    xf, yf = utm_xf - utm_xf, utm_yf - utm_y0

    # # Create vectors of coordinates
    utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
    utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
    x = np.linspace(x0, xf, len(xlon))
    y = np.linspace(y0, yf, len(ylat))

    # Plot the cables geometry in local coordinates
    dw.map.plot_cables2D_m(df_north, df_south, bathy, x, y)
    dw.map.plot_cables3D_m(df_north, df_south, bathy, x, y)


if __name__ == '__main__':
    main()