import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import expm_multiply
from collections import namedtuple

from utils import separate_each_lattice, calculate_psi_k, extract_parameters

Leg = namedtuple("Leg", ("site", "pos", "phase"), defaults=(None,None,0))
Eletric_field = namedtuple("EletricField", ("modulus", "direction", "theta"), defaults=(None, "x", 0.))
System = namedtuple("System", ("lattice","omega_0", "g", "legs"))
System2 = namedtuple("System", ("lattice","omega_0", "g", "legs", "absorb"))

def add_atom_to_hamiltonian(sys):

  H = sys.lattice.hamiltonian

  #Adding the atom to the system
  new_dimension = sys.lattice.num_sites() + 1

  H.resize((new_dimension, new_dimension))
  H = H.astype(np.cdouble)
  H[-1,-1] = sys.omega_0

  for leg in sys.legs:
    idx = sys.lattice.coord_map[leg.site][leg.pos]

    H[-1, idx] = sys.g * np.exp(1j * leg.phase)
    H[idx, -1] = np.conjugate(sys.g * np.exp(1j * leg.phase))

  return H

def time_evolution(H, t, full=False, num_points=50):
  _, dimension = H.shape

  psi = np.zeros(dimension, dtype=np.cdouble)
  psi[-1] = 1. # Atom starts on excited state
  if not full:
    psi = expm_multiply(-1j * H * t, psi)
    return psi
  
  psi = expm_multiply(-1j * H, psi, start=0., stop=t, num=num_points)

  return psi

def time_evolution_lattice(H,t, lattice):
  _, dimension = H.shape

  psi = np.zeros(dimension, dtype=np.cdouble)
  psi[-1] = 1. # Atom starts on excited state
  psi = expm_multiply(-1j * H * t, psi)

  psi_a, psi_b = separate_each_lattice(psi[:-1], at=lattice.la, n=lattice.nmax)

  return psi_a, psi_b


def time_evolution_atom(H, t_i, t_f, num_points):
  _, dimension = H.shape

  t = np.linspace(t_i, t_f, num_points)

  psi = np.zeros(dimension, dtype=np.cdouble)
  psi[-1] = 1. # Atom starts on excited state
  psi = expm_multiply(-1j * H, psi, start=t_i, stop=t_f, num=num_points)

  psi_atom = psi[:,-1]

  return t, psi_atom


def plot_photon_absorption(sys, t_i, t_f, num_points, save=False, name="photon_absorption.pdf"):
  H = sys.lattice.hamiltonian

  #Adding the atom to the system
  new_dimension = sys.lattice.num_sites() + 2

  H.resize((new_dimension, new_dimension))
  H = H.astype(np.cdouble)

  H[-1,-1] = sys.omega_0

  H[-2,-2] = sys.omega_0

  for leg in sys.legs:
    idx = sys.lattice.coord_map[leg.site][leg.pos]

    H[-1, idx] = sys.g * np.exp(1j * leg.phase)
    H[idx, -1] = np.conjugate(sys.g * np.exp(1j * leg.phase))

  for leg in sys.absorb:
    idx = sys.lattice.coord_map[leg.site][leg.pos]

    H[-2, idx] = sys.g * np.exp(1j * leg.phase)
    H[idx, -2] = np.conjugate(H[-2, idx])

  t = np.linspace(t_i, t_f, num_points)

  psi = np.zeros(new_dimension, dtype=np.cdouble)
  psi[-1] = 1. # Atom starts on excited state
  psi = expm_multiply(-1j * H, psi, start=t_i, stop=t_f, num=num_points)

  psi_emitter = psi[:,-1]
  psi_absorb = psi[:,-2]

  plt.xlim(t.min(), t.max())
  plt.ylim(0,1)

  plt.plot(t, np.abs(psi_emitter)**2, label="Emitter")
  plt.plot(t, np.abs(psi_absorb)**2, label="Absorber")

  plt.xlabel("$tJ$")
  plt.ylabel("$|C_e(t)|^2$")

  plt.legend()
  plt.show()

  if save:
    plt.savefig(name)

def plot_lattices_prob_distribution(sys, t, k_space=False, atom_pos=False, cmap="hot",
                                    cn_squared=True, save=False, name="prob_distribution.pdf",
                                    rasterized=False,
                                    vminmax=(None, None),
                                    figsize=None,
                                    plot_parameters=None):

  vmin, vmax = vminmax

  if plot_parameters:
    parameters = extract_parameters(sys)

  H = add_atom_to_hamiltonian(sys)
  psi_a, psi_b = time_evolution_lattice(H,t, sys.lattice)
  psi_lat = np.array([psi_a, psi_b])

  if cn_squared:
    Cn = np.abs(psi_lat) ** 2
    cn_label = '$|C_n(t)|^2$'
  else:
    Cn = np.abs(psi_lat)
    cn_label = '$|C_n(t)|$'

  x_max, y_max, x_min, y_min = sys.lattice.edge_points

  x = np.arange(x_min, x_max + 1)
  y = np.arange(y_min, y_max + 1)

  X, Y = np.meshgrid(x,y)

  titles = ["A lattice","B lattice"]

  site_type = ["A", "B"]

  if k_space:
    fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=True, figsize=(10, 10))

    for i, psi in enumerate(psi_lat):
      fx, fy, psi_k = calculate_psi_k(psi)
      Ck = np.abs(psi_k) ** 2

      axs[0,i].set_title(titles[i])
      axs[0,i].set_xlabel("$n_1$")
      axs[0,i].set_ylabel("$n_2$")
      axs[0,i].set_xlim(x_min, x_max)
      axs[0,i].set_ylim(y_min, y_max)
      axs[0,i].set_aspect("equal")

      axs[1,i].set_xlabel("$k_x$ (units of $\pi$)")
      axs[1,i].set_ylabel("$k_y$ (units of $\pi$)")

      axs[1,i].set_aspect("equal")

      im = axs[0,i].pcolormesh(X, Y, Cn[i], cmap=cmap, shading='gouraud')

      if atom_pos:
        for leg in sys.legs:
          if site_type[i] == leg.site:
            axs[0,i].scatter(*leg.pos, c="white")

      cbar = fig.colorbar(im, ax=axs[0,i])
      cbar.set_label(cn_label)

      im = axs[1,i].pcolormesh(fx / np.pi, fy / np.pi, Ck, 
                               vmin=vmin,
                               vmax=vmax,
                               cmap=cmap,
                               shading='gouraud', 
                               rasterized=rasterized)

      cbar = fig.colorbar(im, ax=axs[1,i])
      cbar.set_label('$|C_k(t)|^2$')
  else:
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, constrained_layout=True, figsize=figsize)
    for i, ax in enumerate(axs):
      ax.set_title(titles[i])
      ax.set_xlabel("$n_1$")

    axs[0].set_ylabel("$n_2$")

    for i, cn_lat in enumerate(Cn):
      axs[i].set_xlim(x_min, x_max)
      axs[i].set_ylim(y_min, y_max)
      axs[i].set_aspect("equal")

      im = axs[i].pcolormesh(X, Y, cn_lat, 
                             cmap=cmap,
                             vmin=vmin,
                             vmax=vmax,
                             shading='gouraud', 
                             rasterized=rasterized)
      im_ratio = cn_lat.shape[0]/cn_lat.shape[1]
      cbar = fig.colorbar(im, ax=axs[i], fraction=0.05*im_ratio)
      cbar.set_label(cn_label)

    if plot_parameters:
      fig.suptitle(parameters + f' t = {t}')

  if save:
    fig.savefig(name, bbox_inches="tight", dpi=200)

  plt.show()

def plot_time_evolution_atom(sys, t_i, t_f, num_points, save=False, name="time"):

  H = add_atom_to_hamiltonian(sys)
  t, psi_atom = time_evolution_atom(H, t_i, t_f, num_points)

  plt.xlim(t.min(), t.max())
  plt.ylim(0,1.1)


  plt.plot(t, np.abs(psi_atom)**2)

  plt.xlabel("$tJ$")
  plt.ylabel("$|C_e(t)|^2$")

  if save:
    plt.savefig(name, bbox_inches="tight", dpi=200)

  plt.show()


def plot_wave_func(sys, t, k_space=False, atom_pos=False, save=False, name="wave_func_2d.pdf"):

  H = add_atom_to_hamiltonian(sys)
  psi_a, psi_b = time_evolution_lattice(H,t, sys.lattice)
  psi_lat = np.array([psi_a, psi_b])

  Cn = np.abs(psi_lat) ** 2

  x_max, y_max, x_min, y_min = sys.lattice.edge_points

  x = np.arange(x_min, x_max + 1)
  y = np.arange(y_min, y_max + 1)

  X, Y = np.meshgrid(x,y)

  titles = ["A lattice","B lattice"]

  site_type = ["A", "B"]

  fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True, constrained_layout=True)
  for i, ax in enumerate(axs):
    ax.set_title(titles[i])
    ax.set_xlabel("$n_x$")

  for i, cn_lat in enumerate(Cn):
    im = axs[i].plot(x, cn_lat[:, int(cn_lat.shape[0] / 2)])
    # im_ratio = cn_lat.shape[0]/cn_lat.shape[1]
    # cbar = fig.colorbar(im, ax=axs[i], fraction=0.05*im_ratio)
    # cbar.set_label('$|C_n(t)|^2$')

  plt.show()

  if save:
    plt.savefig(name, bbox_inches="tight", dpi=200)