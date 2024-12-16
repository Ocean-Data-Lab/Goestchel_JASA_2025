import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Arc

plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelpad'] = 2

# Define Gabor filter parameters
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
