import numpy as np
from scipy.fft import fft2, fftshift, fftfreq
import numba as nb

def separate_each_lattice(arr, at, n):
  if arr.ndim > 1:
    time_steps, *_ = arr.shape
    arr1 = arr[:,:at]
    arr2 = arr[:, at:]

    arr1 = arr1.reshape((time_steps, n,n))
    arr2 = arr2.reshape((time_steps, n,n))

    return arr1, arr2

  arr1 = arr[:at]
  arr2 = arr[at:]

  arr1 = arr1.reshape((n,n))
  arr2 = arr2.reshape((n,n))

  return arr1, arr2

def calculate_psi_k(psi):
  psi_k = fft2(psi, norm="ortho")
  psi_k = fftshift(psi_k)

  nx, ny = psi.shape

  freq_x = fftfreq(nx, d=1/(2*np.pi))
  freq_y = fftfreq(ny, d=1/(2*np.pi))

  freq_x = fftshift(freq_x)
  freq_y = fftshift(freq_y)

  fx, fy = np.meshgrid(freq_x, freq_y)

  return fx, fy, psi_k

# Valley coordinates
K1 = 4 * np.pi / (3 * np.sqrt(3)) * np.array([1, 0])
K2 = -K1

K3 = 2 * np.pi / 3 * np.array([1/np.sqrt(3), 1])
K6 = - K3

K4 = 2 * np.pi / 3 * np.array([1/np.sqrt(3), -1])
K5 = -K4

KK = [K1, K2, K3, K4, K5, K6]

def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

def point_in_triangle(pt, v1, v2, v3):
    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0
    return ((b1 == b2) and (b2 == b3))

def point_in_hexagon_origin(pt, hexagon_vertices):
    center = [0, 0]  # Center is at the origin
    for i in range(len(hexagon_vertices)):
        v1 = hexagon_vertices[i]
        v2 = hexagon_vertices[(i+1) % len(hexagon_vertices)]
        if point_in_triangle(pt, v1, v2, center):
            return True
    return False

def points_graphene_FBZ(Kx, Ky):
    points_mesh = np.vstack([Kx.ravel(), Ky.ravel()]).T
    fbz_vertices = np.array([K2, K5, K3, K1, K4, K6, K2])
    # Filter points inside the hexagon from the meshgrid
    points_inside_mesh = np.array([pt for pt in points_mesh if point_in_hexagon_origin(pt, fbz_vertices)]).T

    return points_inside_mesh

def extract_parameters(sys):
    onsite_energy_a, onsite_energy_b = sys.lattice.onsite_energy['A'], sys.lattice.onsite_energy['B']
    omega_0 = sys.omega_0
    if sys.lattice.eletric_field_params:
        E0 = sys.lattice.eletric_field_params.modulus
    else:
        E0 = 0
    g = sys.g

    return (f'Onsite energy (A,B)=({onsite_energy_a}, {onsite_energy_b}), $\\omega_0={omega_0}$, g = {g}, ' + 
            '$|\\vec{E}| =' + f"{E0}$")

def create_params_string_to_filename(sys):
    onsite_energy_a, onsite_energy_b = sys.lattice.onsite_energy['A'], sys.lattice.onsite_energy['B']
    omega_0 = sys.omega_0
    if sys.lattice.eletric_field_params:
        E0 = sys.lattice.eletric_field_params.modulus
        direction = sys.lattice.eletric_field_params.direction
    else:
        E0 = 0
    g = sys.g

    return f'onsite_energy_(A,B)=({onsite_energy_a},{onsite_energy_b})_omega_0={omega_0}_g={g}_E={E0}_{direction}'


def draw_vectors(ax, vectors: list[np.ndarray], axis_fraction: tuple[float, float], vector_configs: list[dict] = None) -> None:
    """
    Draw multiple vectors on a specified axis, starting from the same position defined as a fraction of the axis limits,
    and label the vectors near their tips, center, or custom position using configuration dictionaries.

    Parameters:
    ax (matplotlib.axes.Axes): The axis on which to draw the vectors.
    vectors (list of np.ndarray): The list of vectors to be drawn.
    axis_fraction (tuple[float, float]): The fraction of the axis limits where the vectors start.
    vector_configs (list of dict, optional): List of dictionaries with configurations for each vector.
        Keys in the dictionary can include:
            - 'label': str, the label for the vector
            - 'color': str, the color of the vector
            - 'ha': str, horizontal alignment of the label
            - 'va': str, vertical alignment of the label
            - 'width': float, width of the vector line
            - 'label_offset': tuple[float, float], offset to translate the label (x_offset, y_offset)
            - 'label_position': str, can be 'tip', 'center', or 'custom'
    """
    # Calculate the start position based on the axis fraction
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()

    start_position = np.array([x_lim[0] + axis_fraction[0] * (x_lim[1] - x_lim[0]), 
                               y_lim[0] + axis_fraction[1] * (y_lim[1] - y_lim[0])])
    
    # Plot each vector with its label, color, width, and alignment
    for i, v in enumerate(vectors):
        # Calculate the end position of the vector
        end_position = start_position + v

        # Get configuration for the current vector
        config = vector_configs[i] if vector_configs and i < len(vector_configs) else {}

        # Set the color and width for the vector; use defaults if not provided
        color = config.get('color', 'r')
        width = config.get('width', 0.005)  # Default width is 1

        # Plot the vector
        ax.quiver(start_position[0], start_position[1], 
                  v[0], v[1], 
                  angles='xy', scale_units='xy', scale=1, color=color, width=width)

        # Add the label based on the specified position (tip, center, or custom)
        label = config.get('label')
        if label:
            ha = config.get('ha', 'left')
            va = config.get('va', 'top')
            fontsize = config.get('fontsize', 12)
            label_position = config.get('label_position', 'tip')  # Default is 'tip'
            label_offset = config.get('label_offset', (0, 0))  # Default offset is (0, 0)

            if label_position == 'center':
                # Find the midpoint of the vector
                midpoint = (start_position + end_position) / 2
                label_x = midpoint[0] + label_offset[0]
                label_y = midpoint[1] + label_offset[1]
            elif label_position == 'custom':
                # Custom label position calculation can be implemented here if needed
                label_x = start_position[0] + label_offset[0]
                label_y = start_position[1] + label_offset[1]
            else:  # 'tip' or default
                label_x = end_position[0] + label_offset[0]
                label_y = end_position[1] + label_offset[1]

            ax.text(label_x, label_y, label, fontsize=fontsize, ha=ha, va=va, color=color)

def draw_basis_vectors(ax, vectors, scale = None, pos = (0.9, 0.05)):
    if not scale:
        RATIO = 0.075
        xlim = ax.get_xlim()
        x_axis_size = xlim[1] - xlim[0]

        scale = RATIO * x_axis_size
    unit_vectors = [vectors[0] * scale, vectors[1] * scale]

    

    vector_configs = [
        {
            'label': r"$\vec{a}_1$",
            'color': "white",
            'ha': "center",
            'va': "bottom",
            'width': 0.007,  # Set width for the vector
            # 'label_offset': 0.3  # Specific label offset for this vector
        },
        {
            'label': r"$\vec{a}_2$",
            'color': "white",
            'ha': "center",
            'va': "bottom",
            'width': 0.007,  # Set width for the vector
            # 'label_offset': 0.1  # Specific label offset for this vector
        }
    ]
    draw_vectors(ax, unit_vectors, pos, vector_configs)

def annotate_electric_field_text(ax, sys, pos = (0.05, 0.05)):
    if not sys.lattice.eletric_field_params or sys.lattice.eletric_field_params.modulus == 0.:
        return
    
    if sys.lattice.eletric_field_params.direction == "x":
        ax.annotate(r"$\vec{E} = \epsilon_0 \hat{x}$",
                xy =pos, xycoords="axes fraction", fontsize=15, color="white")
    if sys.lattice.eletric_field_params.direction == "y":
        ax.annotate(r"$\vec{E} = \epsilon_0 \vec{a}_2$",
                xy = pos, xycoords="axes fraction", fontsize=15, color="white")

def draw_electric_field(ax, vectors, sys, arrow_size=15, pos = (0.11, 0.05)):
    if not sys.lattice.eletric_field_params or sys.lattice.eletric_field_params.modulus == 0.:
        return

    a1 = 0.5 * np.array([np.sqrt(3.), 3.])
    a2 = 0.5 * np.array([-np.sqrt(3.), 3.])

    if sys.lattice.eletric_field_params.direction == "x":
        E_vec = [arrow_size * a1]
        E_config = [
            {
                "label": r"$\vec{E}$", 
                "color": "white",
                "ha": "center",
                "va": "center",
                "label_position": "center",
                "label_offset": (-8, 2),
                "width": 0.01,
                "fontsize": 20
            }
        ]
    if sys.lattice.eletric_field_params.direction == "y":
        E_vec = [arrow_size * a2]
        E_config = [
            {
                "label": r"$\vec{E}$", 
                "color": "white",
                "ha": "center",
                "va": "center",
                "label_position": "center",
                "label_offset": (8, 2),
                "width": 0.01,
                "fontsize": 20
            }
        ]
    draw_vectors(ax, E_vec, pos, E_config)

def get_real_lattice_sites_position(Nx, Ny):
    a1 = 0.5 * np.array([np.sqrt(3.), 3.])
    a2 = 0.5 * np.array([-np.sqrt(3.), 3.])

    delta3 = (0, -1)

    M = np.array([a1, a2]).T

    coeff =  np.stack([Nx.ravel(), Ny.ravel()])
    na = M @ coeff

    nb = np.empty_like(na)

    nb[0] = na[0] + delta3[0]
    nb[1] = na[1] + delta3[1]

    return na, nb

def plot_graphene(Nx, Ny, Cna, Cnb, ax, s=5, cmap="hot", vmin=None, vmax=None, logscale=None):
    na, nb = get_real_lattice_sites_position(Nx, Ny)

    if logscale:
        Cna = np.log10(Cna)
        Cnb = np.log10(Cnb)

    im = ax.scatter(na[0], na[1] ,c=Cna, s=s, label="A", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.scatter(nb[0], nb[1], c=Cnb, s=s, label="B", cmap=cmap, vmin=vmin, vmax=vmax)

    return im
