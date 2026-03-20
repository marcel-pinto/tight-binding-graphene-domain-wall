import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import bmat, diags


class Graphene:

  def __init__(self, nmax, onsite_energy=(0,0), J = 1., J_prime = 0, anisotropy_J=0, pbc=None, eps_onsite = 0, seed=42):
    self.nmax = nmax
    self.mid_point = np.array([nmax - 1, nmax-1], dtype=int) / 2
    self.la = self.lb = nmax ** 2
    self.onsite_energy = {"A": onsite_energy[0], "B": onsite_energy[1]}
    self.J = J
    self.J_prime = J_prime
    self.anisotropy_J= anisotropy_J

    self.eps_onsite = eps_onsite
    
    if self.eps_onsite:
      np.random.seed(seed)
    

    if pbc not in ["x", "y", "xy", None]:
        raise Exception("The PBC conditions can be only 'x' , 'y' or 'xy'." )

    self.pbc = pbc
    
    self.coord_map = self._create_coordinates_map(shape=(nmax, nmax))

  def _generate_off_diag_block(self):
    ks = [0, 1, self.nmax]

    diagonals = [np.full(self.la - k, -self.J) for k in ks]
    if self.anisotropy_J:
       diagonals[1][:] = -self.anisotropy_J

    diagonals[1][self.nmax - 1 :: self.nmax] = 0.

    non_pbc_part = diags(diagonals, ks)

    if not self.pbc:
      return non_pbc_part

    if self.pbc == "x":
      offset = -self.nmax + 1
      extra_diagonals = np.zeros((self.nmax ** 2) - (self.nmax - 1))
      extra_diagonals[::self.nmax] = -self.J

      other_diagonals = diags(extra_diagonals, offset, shape=(self.la, self.la))

      return non_pbc_part + other_diagonals

    if self.pbc == "y":
      offset = -self.la + self.nmax
      extra_diagonals = np.full(self.nmax, -self.J)
      other_diagonals = diags(extra_diagonals, offset, shape=(self.la, self.la))

      return non_pbc_part + other_diagonals

    if self.pbc == "xy":
      offset = [-self.la + self.nmax, -self.nmax + 1]

      # Y part
      y_part = np.full(self.nmax, -self.J)

      # X part
      x_part = np.zeros((self.nmax ** 2) - (self.nmax - 1))
      x_part[::self.nmax] = -self.J
      extra_diagonals = [
        y_part,
        x_part
      ]

      other_diagonals = diags(extra_diagonals, offset, shape=(self.la, self.la))

      return non_pbc_part + other_diagonals

  def _generate_main_diag_block(self, site):
    energy = self.onsite_energy[site]
    n = self.num_sites(site)
    onsite_energies_diag = np.full(n, energy)
    
    if self.eps_onsite:
      noise_onsite_energies = (np.random.rand(n) - 0.5)* self.eps_onsite
      onsite_energies_diag += noise_onsite_energies
  
    if not self.J_prime:
      return diags(onsite_energies_diag)
    
    if not self.pbc:
      ks = [1, self.nmax - 1, self.nmax]
      k = [self._build_diag_for_NNN(ki, n) for ki in ks]

      diag = [onsite_energies_diag] + k * 2
      offsets = [0] + ks + [-ki for ki in ks]

      return diags(diag, offsets=offsets)

    if self.pbc == "x":
      ks = [1, self.nmax - 1, self.nmax, 2 * self.nmax - 1]
      k = [self._build_diag_for_NNN(ki, n) for ki in ks]

      diag = [onsite_energies_diag] + k * 2
      offsets = [0] + ks + [-ki for ki in ks]

      return diags(diag, offsets=offsets)

    if self.pbc == "y":
      ks = [1, self.nmax - 1, self.nmax, self.nmax * (self.nmax - 1), self.nmax * (self.nmax - 1) + 1]
      k = [self._build_diag_for_NNN(ki, n) for ki in ks]
      
      diag = [onsite_energies_diag] + k * 2
      offsets = [0] + ks + [-ki for ki in ks]
      
      return diags(diag, offsets=offsets)
    
    if self.pbc =="xy":
      ks = [1, self.nmax - 1, self.nmax, 2 * self.nmax - 1]
      k = [self._build_diag_for_NNN(ki, n) for ki in ks]
      
      diag = [onsite_energies_diag] + k * 2
      offsets = [0] + ks + [-ki for ki in ks]
      return diags(diag, offsets=offsets)
      

  def _build_diag_for_NNN(self, ki, n):
    k = np.full(n - ki, self.J_prime)

    if ki == 1:
      k[self.nmax-1::self.nmax] = 0
      return k

    if ki == self.nmax - 1:
      k[::self.nmax] = 0
      return k
    
    if ki == 2 * self.nmax - 1:
      k[:] = 0
      k[::self.nmax] = self.J_prime
      return k
    
    return k

      
  def num_sites(self, kind="all") -> int:
    match kind.upper():
      case "A":
        return self.la
      case "B":
        return self.lb
      case "ALL":
        return self.la + self.lb
      case _:
        return None

  @property
  def hamiltonian(self):

    Haa  = self._generate_main_diag_block(site="A")
    Hbb  = self._generate_main_diag_block(site="B")
    Hab  = self._generate_off_diag_block()

    return bmat([
        [Haa,   Hab],
        [Hab.T, Hbb]
      ]).todok()


  @staticmethod
  def _create_coordinates_map(shape):
    x, y = shape
    total_size = x * y

    a = np.arange(total_size).reshape(shape)

    b = total_size + a

    avg_x = int(x / 2)
    avg_y = int(y / 2)

    return {
        "A": {(i - avg_x, j - avg_y) : a[j,i] for i in range(x) for j in range(y)},
        "B": {(i - avg_x, j - avg_y) : b[j,i] for i in range(x) for j in range(y)},
      }

  @property
  def edge_points(self):
    x_max, y_max = max(self.coord_map["A"].keys())
    x_min, y_min = min(self.coord_map["A"].keys())

    return x_max, y_max, x_min, y_min


  @property
  def graph(self):
    adj_matrix = self.hamiltonian.todense() / (-self.J)

    return nx.from_numpy_array(adj_matrix)


  def _graphene_layout(self, a=1):
    a_positions = self.coord_map["A"].keys()
    b_positions = self.coord_map["B"].keys()

    a1 = 0.5 * np.array([np.sqrt(3.), 3.])
    a2 = 0.5 * np.array([-np.sqrt(3.), 3.])

    delta = a * np.array([0,-1.])

    Ma = np.array([a1, a2]).T

    a_pos_plot = [Ma @ np.array([n1,n2]) for n1, n2 in a_positions]
    b_pos_plot = [(Ma @ np.array([n1,n2])) + delta for n1, n2 in b_positions]

    a_nodes_pos = {node: position for node, position in enumerate(a_pos_plot)}
    b_nodes_pos = {node: position for node, position in enumerate(b_pos_plot, start=self.la)}

    return a_nodes_pos | b_nodes_pos


  def plot(self, with_labels=False, labels_type='number', theta_rot=False, color_by_weight=False):
      """
      Plot the graphene lattice with edges colored according to their weights in the adjacency matrix.
      
      Parameters:
      -----------
      with_labels : bool
          Whether to display labels for the nodes
      labels_type : str
          Type of labels to display ('number' for node indices)
      theta_rot : float or False
          Rotation angle (in radians) for the entire lattice
      color_by_weight : bool
          If True, edges are colored according to their weight in the adjacency matrix
      """
      G = self.graph

      node_size = 60

      figsize = (15,9) if not color_by_weight else (17,9)

      fig, ax = plt.subplots(figsize=figsize)

      pos = self._graphene_layout()

      if theta_rot:
          M = np.array([
              [np.cos(theta_rot), - np.sin(theta_rot)],
              [np.sin(theta_rot), np.cos(theta_rot)]
          ])
          pos = {node: np.dot(M, p) for node, p in pos.items()}

      a_nodes = range(self.la)
      b_nodes = range(self.la, self.la + self.lb)

      if color_by_weight:
          # Get the adjacency matrix
          adj_matrix = np.abs(self.hamiltonian.todense())
          
          edge_min = np.min(adj_matrix[adj_matrix> 0])
          edge_max = np.max(adj_matrix)
          # Create edge colors based on weights
          edge_colors = []
          edge_widths = []
          edges_for_drawing = []
          
          # Only consider edges with non-zero weight
          for u, v in G.edges():
              weight = abs(adj_matrix[u, v])
              if weight > 0:
                  edges_for_drawing.append((u, v))
                  # Normalize weight for coloring - we expect weights close to 1.0
                  edge_colors.append(weight)
                  edge_widths.append(1.0 + weight)
          
          # Draw edges with color mapping
          edges = nx.draw_networkx_edges(
              G, 
              pos=pos, 
              edgelist=edges_for_drawing,
              width=edge_widths,
              edge_color=edge_colors,
              edge_cmap=plt.cm.viridis,  # You can choose a different colormap
              edge_vmin=edge_min,
              edge_vmax=edge_max,  # Adjust this range based on your expected weights
              ax=ax
          )
          
          # Add a colorbar
          sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=edge_min, vmax=edge_max))
          sm.set_array([])
          cbar = plt.colorbar(sm, ax=ax)
          cbar.set_label('Edge Weight')
      else:
          # Draw all edges with the same color
          nx.draw_networkx_edges(G, pos=pos, ax=ax)

      # Draw nodes
      nx.draw_networkx_nodes(G, node_size=node_size, pos=pos, nodelist=a_nodes, ax=ax).set_edgecolor('black')
      nx.draw_networkx_nodes(G, node_size=node_size, node_color="darkorange", pos=pos, nodelist=b_nodes, ax=ax).set_edgecolor('black')

      offset = np.array([0.,0.2])

      if with_labels:
          if labels_type == 'number':
              labels_pos = {node: position + offset for node, position in pos.items()}
              nx.draw_networkx_labels(G, pos=labels_pos)

      plt.box(False)
      plt.show()

      return fig, ax


  def get_edge_points(self):
    if self.pbc == "xy":
        return {"A": [], "B": []}
    if self.pbc == "y":
        return {
          "A": [
            [n * self.nmax for n in range(self.nmax)],
            [n * self.nmax + (self.nmax - 1) for n in range(self.nmax)]
            ],
        "B": [
            [n * self.nmax + self.la for n in range(self.nmax)],
            [n * self.nmax + (self.nmax - 1) + self.la for n in range(self.nmax)]
            ]
        }
    if self.pbc == "x":
      return {
          "A": [
            [n  for n in range(self.nmax)],
            [n + (self.nmax - 1) * self.nmax for n in range(self.nmax)]
            ],
        "B": [
            [n + self.la for n in range(self.nmax)],
            [n + (self.nmax - 1) * self.nmax + self.la for n in range(self.nmax)]
            ]
        }


if __name__ == "__main__":
  CMAP = 'hot'
  lattice=Graphene(nmax=4, onsite_energy=(0.0, 0.0), pbc="y", J=-1, J_prime=1, eps_onsite=1)
  Haa = lattice._generate_main_diag_block("A").todense()
  Hbb = lattice._generate_main_diag_block("B").todense()
  Hab = lattice._generate_off_diag_block().todense()

  fig, ax = plt.subplots(2,2, sharex=True, sharey=True)
  im = ax[0,0].matshow(Haa, cmap=CMAP)
  plt.colorbar(im)
  im = ax[1,1].matshow(Hbb, cmap=CMAP)
  plt.colorbar(im)
  im = ax[0,1].matshow(Hab, cmap=CMAP)
  plt.colorbar(im)
  im = ax[1,0].matshow(Hab.T, cmap=CMAP)
  plt.colorbar(im)
  
  plt.show()


  def plot(self, with_labels=False, labels_type='number', theta_rot=False, color_by_weight=True):
    """
    Plot the graphene lattice with edges colored according to their weights in the adjacency matrix.
    
    Parameters:
    -----------
    with_labels : bool
        Whether to display labels for the nodes
    labels_type : str
        Type of labels to display ('number' for node indices)
    theta_rot : float or False
        Rotation angle (in radians) for the entire lattice
    color_by_weight : bool
        If True, edges are colored according to their weight in the adjacency matrix
    """
    G = self.graph

    node_size = 60
    fig, ax = plt.subplots(figsize=(15,9))

    pos = self._graphene_layout()

    if theta_rot:
        M = np.array([
            [np.cos(theta_rot), - np.sin(theta_rot)],
            [np.sin(theta_rot), np.cos(theta_rot)]
        ])
        pos = {node: np.dot(M, p) for node, p in pos.items()}

    a_nodes = range(self.la)
    b_nodes = range(self.la, self.la + self.lb)

    if color_by_weight:
        # Get the adjacency matrix
        adj_matrix = np.abs(self.hamiltonian.todense())
        
        # Create edge colors based on weights
        edge_colors = []
        edge_widths = []
        edges_for_drawing = []
        
        # Only consider edges with non-zero weight
        for u, v in G.edges():
            weight = abs(adj_matrix[u, v])
            if weight > 0:
                edges_for_drawing.append((u, v))
                # Normalize weight for coloring - we expect weights close to 1.0
                edge_colors.append(weight)
                edge_widths.append(1.0 + weight)
        
        # Draw edges with color mapping
        edges = nx.draw_networkx_edges(
            G, 
            pos=pos, 
            edgelist=edges_for_drawing,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,  # You can choose a different colormap
            edge_vmin=0.0,
            edge_vmax=1.5,  # Adjust this range based on your expected weights
            ax=ax
        )
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0.0, vmax=1.5))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Edge Weight (normalized)')
    else:
        # Draw all edges with the same color
        nx.draw_networkx_edges(G, pos=pos, ax=ax)

    # Draw nodes
    nx.draw_networkx_nodes(G, node_size=node_size, pos=pos, nodelist=a_nodes, ax=ax).set_edgecolor('black')
    nx.draw_networkx_nodes(G, node_size=node_size, node_color="darkorange", pos=pos, nodelist=b_nodes, ax=ax).set_edgecolor('black')

    offset = np.array([0.,0.2])

    if with_labels:
        if labels_type == 'number':
            labels_pos = {node: position + offset for node, position in pos.items()}
            nx.draw_networkx_labels(G, pos=labels_pos)

    plt.box(False)
    plt.show()

    return fig, ax