import networkx as nx
import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import bmat, diags


class Graphene:
  def __init__(
        self,
        nmax,
        onsite_energy_sign=(1,-1),
        J = 1.,
        delta_0 = 0.,
        x0 = 0.,
        lambda_0 = 0.1,
        domain_wall_angle=0.,
        J_prime = 0,
        anisotropy_J=0,
        pbc=None,
        eps_onsite = 0,
        seed=42):
    
    self.nmax = nmax
    self.mid_point = np.array([nmax - 1, nmax-1], dtype=int) / 2
    self.la = self.lb = nmax ** 2
    self.onsite_energy_sign = {"A": onsite_energy_sign[0], "B": onsite_energy_sign[1]}
    self.J = J
    self.J_prime = J_prime
    self.anisotropy_J= anisotropy_J

    self.a = 1.
    # self.a1 = 0.5 * np.array([np.sqrt(3.), 3.])
    # self.a2 = 0.5 * np.array([-np.sqrt(3.), 3.])


  # Lattice Vectors for graphene
    self.a1 = np.sqrt(3)/2 * np.array([np.sqrt(3), 1.])
    self.a2 = np.sqrt(3) * np.array([0, 1.])
    self.d1 = -0.5 * np.array([1., np.sqrt(3)])

    self.delta_0 = delta_0
    self.x0 = x0
    self.lambd = lambda_0

    self.theta = np.deg2rad(domain_wall_angle)

    self.eps_onsite = eps_onsite

    if self.eps_onsite:
      np.random.seed(seed)
    

    if pbc not in ["x", "y", "xy", None]:
        raise Exception("The PBC conditions can be only 'x' , 'y' or 'xy'." )

    self.pbc = pbc
    
    self.coord_map = self._create_coordinates_map(shape=(nmax, nmax))
    self._compute_domain_wall()

  def _generate_off_diag_block(self):
    ks = [0, 1, self.nmax]

    diagonals = [np.full(self.la - k, -self.J) for k in ks]
    if self.anisotropy_J:
       diagonals[1][:] = -self.anisotropy_J

    diagonals[1][self.nmax - 1 :: self.nmax] = 0.

    non_pbc_part = diags(diagonals, ks)

    if not self.pbc:
      return non_pbc_part

    if self.pbc == "y":
      offset = -self.nmax + 1
      extra_diagonals = np.zeros((self.nmax ** 2) - (self.nmax - 1))
      extra_diagonals[::self.nmax] = -self.J

      other_diagonals = diags(extra_diagonals, offset, shape=(self.la, self.la))

      return non_pbc_part + other_diagonals

    if self.pbc == "x":
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

  def _compute_domain_wall(self):
    na, nb = self.n_real_pos

# Note that the direction of the domain wall is always perpendicular to the 
# direction of the frequency gradient
    # m = np.array([-np.sin(self.theta), np.cos(self.theta)]) 

    m = np.array([np.cos(self.theta), np.sin(self.theta)])
    self.delta = {
      "A" : self.delta_0 * np.tanh((na.T @ m)/self.lambd),
      "B" : self.delta_0 * np.tanh((nb.T @ m)/self.lambd)
      }

  def _generate_main_diag_block(self, site):
    energy_sign = self.onsite_energy_sign[site]
    n = self.num_sites(site)
    onsite_energies_diag = np.full(n, energy_sign) * self.delta[site]

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
  def n_real_pos(self):
    coord_map = self.coord_map
    inv_mapA = {value : key for key, value in coord_map['A'].items()}

    m = np.array([(nx, ny) for _, (nx,ny) in sorted(inv_mapA.items())])

    Ma = np.array([self.a1, self.a2]).T

    na = Ma @ m.T

    nb = na.copy() + self.d1.reshape(2,1)


    return na, nb
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

    a1 = self.a1
    a2 = self.a2

    delta = self.d1

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

class GrapheneArmchair:
    """
    Graphene with ARMCHAIR edges (x-direction) and ZIGZAG edges (y-direction).

    Built from a 4-atom rectangular supercell (2 A + 2 B), tiled nx x ny times.
        Cell vectors: R1 = (sqrt(3), 0),  R2 = (0, 3)        (bond length = 1)
        Basis:  A1 = (sqrt(3)/2, 0)     A2 = (0,         3/2)
                B1 = (0,         1/2)   B2 = (sqrt(3)/2, 2  )

    Indexing:
        - Every site has a flat index in [0, 4*nx*ny).
        - A-sublattice occupies [0, 2*nx*ny);  B occupies [2*nx*ny, 4*nx*ny).
        - Within a sublattice, flat_idx = 2*(j*nx + i) + b   where b in {0, 1}.
    """

    def __init__(self, nx, ny=None, onsite_energy_sign=(1, -1), J=1.,
                 delta_0=0., lambda_0=0.1, domain_wall_angle=0., x0=0.,
                 pbc=None, eps_onsite=0, seed=42):
        if ny is None:
            ny = nx
        self.nx, self.ny = nx, ny
        self.J = J
        self.la = self.lb = 2 * nx * ny
        self.onsite_energy_sign = {"A": onsite_energy_sign[0], "B": onsite_energy_sign[1]}

        # --- Geometry ---
        self.R1 = np.array([np.sqrt(3), 0.])
        self.R2 = np.array([0.,          3.])
        self.basis = {
            ("A", 0): np.array([np.sqrt(3) / 2, 0.0]),   # A1
            ("A", 1): np.array([0.0,            1.5]),   # A2
            ("B", 0): np.array([0.0,            0.5]),   # B1
            ("B", 1): np.array([np.sqrt(3) / 2, 2.0]),   # B2
        }

        # --- Physics params ---
        self.delta_0 = delta_0
        self.lambd = lambda_0
        self.theta = np.deg2rad(domain_wall_angle)
        self.x0 = x0
        self.eps_onsite = eps_onsite
        if eps_onsite:
            np.random.seed(seed)

        if pbc not in (None, "x", "y", "xy"):
            raise ValueError("pbc must be None, 'x', 'y', or 'xy'")
        self.pbc = pbc

        # --- Maps + domain wall ---
        self.coord_map = self._create_coordinates_map(nx, ny, la=self.la)
        self._compute_domain_wall()

    # ======================================================================
    # Methods kept from the original class (same names / same contracts)
    # ======================================================================

    @staticmethod
    def _create_coordinates_map(nx_, ny_, la):
        """
        {"A": {(i_centered, j_centered, b): flat_idx}, "B": {...}}

        Keys are 3-tuples because each rectangular cell now carries two
        atoms per sublattice (b in {0, 1}).  Cell indices are centered on
        (avg_x, avg_y) = (nx//2, ny//2) -- same convention as the original.
        """
        avg_x, avg_y = nx_ // 2, ny_ // 2
        cmap = {"A": {}, "B": {}}
        for sub, offset in (("A", 0), ("B", la)):
            for j in range(ny_):
                for i in range(nx_):
                    for b in (0, 1):
                        flat = offset + 2 * (j * nx_ + i) + b
                        cmap[sub][(i - avg_x, j - avg_y, b)] = flat
        return cmap

    @property
    def n_real_pos(self):
        """
        Real-space positions of every site, separated by sublattice.

        Returns:
            na : np.ndarray, shape (2, la)   -- columns are (x, y) of A sites
            nb : np.ndarray, shape (2, lb)   -- columns are (x, y) of B sites
        Same shape/contract as the original class, so _compute_domain_wall
        and any downstream code keep working unchanged.
        """
        inv = {v: k for k, v in self.coord_map["A"].items()}
        keys_sorted = [inv[k] for k in sorted(inv)]  # ascending flat index
        na = np.empty((2, self.la))
        nb = np.empty((2, self.lb))
        for flat, (ic, jc, b) in enumerate(keys_sorted):
            i, j = ic + self.nx // 2, jc + self.ny // 2
            cell_origin = i * self.R1 + j * self.R2
            na[:, flat] = cell_origin + self.basis[("A", b)]
            nb[:, flat] = cell_origin + self.basis[("B", b)]
        return na, nb

    def num_sites(self, kind="all") -> int:
        match kind.upper():
            case "A":   return self.la
            case "B":   return self.lb
            case "ALL": return self.la + self.lb
            case _:     return None

    def _compute_domain_wall(self):
        """Same as original: tanh wall along m=(cos θ, sin θ)."""
        na, nb = self.n_real_pos
        m = np.array([np.cos(self.theta), np.sin(self.theta)])
        self.delta = {
            "A": self.delta_0 * np.tanh((na.T @ m) / self.lambd),
            "B": self.delta_0 * np.tanh((nb.T @ m) / self.lambd),
        }

    @property
    def edge_points(self):
        """(x_max, y_max, x_min, y_min) of centered cell indices."""
        xs = [k[0] for k in self.coord_map["A"].keys()]
        ys = [k[1] for k in self.coord_map["A"].keys()]
        return max(xs), max(ys), min(xs), min(ys)

    def get_edge_points(self):
        """
        Flat indices of atoms on the OPEN edges, grouped by sublattice.
            pbc='y'  -> two armchair edges (left/right)
            pbc='x'  -> two zigzag edges (top/bottom)
            pbc='xy' -> {} (no open edges)
            pbc=None -> all four edges
        """
        if self.pbc == "xy":
            return {"A": [], "B": []}

        ax, ay = self.nx // 2, self.ny // 2
        def fA(i, j, b): return self.coord_map["A"][(i - ax, j - ay, b)]
        def fB(i, j, b): return self.coord_map["B"][(i - ax, j - ay, b)]

        edges = {"A": [], "B": []}

        if self.pbc in (None, "y"):   # armchair L/R are open
            left_A  = [fA(0, j, 1)            for j in range(self.ny)]   # A2
            right_A = [fA(self.nx - 1, j, 0)  for j in range(self.ny)]   # A1
            left_B  = [fB(0, j, 0)            for j in range(self.ny)]   # B1
            right_B = [fB(self.nx - 1, j, 1)  for j in range(self.ny)]   # B2
            edges["A"] += [left_A, right_A]
            edges["B"] += [left_B, right_B]

        if self.pbc in (None, "x"):   # zigzag T/B are open
            bottom_A = [fA(i, 0, 0)           for i in range(self.nx)]   # A1 row
            top_B    = [fB(i, self.ny - 1, 1) for i in range(self.nx)]   # B2 row
            edges["A"] += [bottom_A]
            edges["B"] += [top_B]

        return edges

    # ======================================================================
    # Hamiltonian
    # ======================================================================

    def _bond_list(self):
        """
        A1(i,j) -> B1(i,j),  B1(i+1,j),  B2(i,j-1)
        A2(i,j) -> B1(i,j),  B2(i,j),    B2(i-1,j)
        """
        nx_, ny_ = self.nx, self.ny
        wrap_x = self.pbc in ("x", "xy")
        wrap_y = self.pbc in ("y", "xy")

        def wrap(i, j):
            if not (0 <= i < nx_):
                if not wrap_x: return None
                i %= nx_
            if not (0 <= j < ny_):
                if not wrap_y: return None
                j %= ny_
            return i, j

        ax, ay = self.nx // 2, self.ny // 2
        fA = lambda i, j, b: self.coord_map["A"][(i - ax, j - ay, b)]
        fB = lambda i, j, b: self.coord_map["B"][(i - ax, j - ay, b)]

        bonds = []
        for j in range(ny_):
            for i in range(nx_):
                bonds.append((fA(i, j, 0), fB(i, j, 0)))
                w = wrap(i + 1, j)
                if w: bonds.append((fA(i, j, 0), fB(*w, 0)))
                w = wrap(i, j - 1)
                if w: bonds.append((fA(i, j, 0), fB(*w, 1)))

                bonds.append((fA(i, j, 1), fB(i, j, 0)))
                bonds.append((fA(i, j, 1), fB(i, j, 1)))
                w = wrap(i - 1, j)
                if w: bonds.append((fA(i, j, 1), fB(*w, 1)))
        return bonds

    @property
    def hamiltonian(self):
        N = self.la + self.lb
        H = lil_matrix((N, N))
        for sub in ("A", "B"):
            s = self.onsite_energy_sign[sub]
            d = self.delta[sub]
            base = 0 if sub == "A" else self.la
            diag = s * d
            if self.eps_onsite:
                diag = diag + (np.random.rand(self.la) - 0.5) * self.eps_onsite
            for k in range(self.la):
                H[base + k, base + k] = diag[k]
        for a, b in self._bond_list():
            H[a, b] = -self.J
            H[b, a] = -self.J
        return H.todok()

    @property
    def graph(self):
        M = self.hamiltonian.todense() / (-self.J)
        np.fill_diagonal(M, 0.)  # onsite terms are not bonds; avoid self-loops
        return nx.from_numpy_array(M)

    # ======================================================================
    # Plotting (kept from original, adapted for this layout)
    # ======================================================================

    def _graphene_layout(self):
        """{flat_idx: (x, y)} for networkx drawing."""
        na, nb = self.n_real_pos
        layout = {k: na[:, k]                for k in range(self.la)}
        layout.update({k + self.la: nb[:, k] for k in range(self.lb)})
        return layout

    def plot(self, with_labels=False, labels_type='number',
             theta_rot=False, color_by_weight=False):
        G = self.graph
        node_size = 60
        figsize = (15, 9) if not color_by_weight else (17, 9)
        fig, ax = plt.subplots(figsize=figsize)

        pos = self._graphene_layout()
        if theta_rot:
            M = np.array([[np.cos(theta_rot), -np.sin(theta_rot)],
                          [np.sin(theta_rot),  np.cos(theta_rot)]])
            pos = {n: M @ p for n, p in pos.items()}

        a_nodes = range(self.la)
        b_nodes = range(self.la, self.la + self.lb)

        if color_by_weight:
            W = np.abs(self.hamiltonian.todense())
            emin = np.min(W[W > 0]); emax = np.max(W)
            colors, widths, elist = [], [], []
            for u, v in G.edges():
                w = abs(W[u, v])
                if w > 0:
                    elist.append((u, v)); colors.append(w); widths.append(1.0 + w)
            nx.draw_networkx_edges(G, pos=pos, edgelist=elist,
                                   width=widths, edge_color=colors,
                                   edge_cmap=plt.cm.viridis,
                                   edge_vmin=emin, edge_vmax=emax, ax=ax)
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(vmin=emin, vmax=emax))
            sm.set_array([])
            plt.colorbar(sm, ax=ax).set_label('Edge Weight')
        else:
            nx.draw_networkx_edges(G, pos=pos, ax=ax)

        nx.draw_networkx_nodes(G, pos=pos, nodelist=a_nodes,
                               node_size=node_size, ax=ax).set_edgecolor('black')
        nx.draw_networkx_nodes(G, pos=pos, nodelist=b_nodes, node_color="darkorange",
                               node_size=node_size, ax=ax).set_edgecolor('black')

        if with_labels and labels_type == 'number':
            labels_pos = {n: p + np.array([0., 0.2]) for n, p in pos.items()}
            nx.draw_networkx_labels(G, pos=labels_pos, ax=ax)

        ax.set_aspect("equal")
        plt.box(False)
        return fig, ax

if __name__ == "__main__":
  CMAP = 'hot'
  lattice=Graphene(nmax=4, onsite_energy_sign=(1, -1), pbc="y")
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

