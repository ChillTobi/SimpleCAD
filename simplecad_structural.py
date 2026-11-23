"""
SimpleCAD Structural — linear static stress and natural frequencies, the "Structural" study type
of the Thermal window (simplecad_thermal.py loads this file; it has no window of its own).

  * every body is meshed with tetrahedra by gmsh (its surface is re-parametrised and re-meshed
    at the element size, finer on curved faces); quadratic tetrahedra (10 nodes) by default,
    since linear ones are far too stiff in bending.
  * stiffness and mass matrices of the straight-sided elements are assembled in numpy; isotropic
    linear elastic materials (E, ν, density, thermal expansion α from the material list).
  * bodies that touch are bonded: the nodes of one body on the other's surface follow that
    surface (tied by its quadratic shape functions).
  * supports (fixed, sliding / roller, pinned on a round face) remove the constrained
    directions of their nodes exactly; loads: force, pressure, torque on faces, gravity,
    thermal expansion (uniform temperature, or the temperatures of a Conduction run).
  * static: one linear solve (direct for small models, else conjugate gradients with a two-level
    preconditioner: Gauss–Seidel around an exact solve on the corner nodes); stresses are recovered at the element corners and averaged per body. Every run
    checks the force balance (reactions against applied loads).
  * modal: the lowest natural frequencies and mode shapes (shift-invert Lanczos).
Units in the window: mm, N, MPa, N·m; the solver works in mm, N, MPa, t (tonnes), s.
"""
import math, time

import numpy as np

import simplecad_thermal as th

GRAVITY_MM_S2 = 9810.0
DEFAULT_MAX_ELEMENTS = 30000
DIRECT_LIMIT = 20000                 # unknowns up to which the sparse direct solver is used
EIGEN_DIRECT_LIMIT = 50000           # natural frequencies: direct factorisation up to here
TIE_TOLERANCE = 0.02                 # share of the element size within which surfaces touch
PIN_PLANAR = 0.9                     # share of Σ n nᵀ along one direction: a flat face


class StructuralUnavailable(th.ThermalError):
    pass


def missing_packages():
    """Names of the pip packages a structural run needs and that are not installed."""
    missing = []
    for module, package in (("gmsh", "gmsh"),):
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    return missing


def _require():
    missing = missing_packages()
    if missing:
        raise StructuralUnavailable(
            "Structural studies need the Python package" + ("s " if len(missing) > 1 else " ") +
            " and ".join(missing) + ". Install with 'pip install " + " ".join(missing) +
            "' (or run 'python build.py --venv').")


# ── materials ─────────────────────────────────────────────────────────────────
def mechanical_properties(props):
    """{"E" MPa, "nu", "yield" MPa or None, "alpha" 1/K, "rho" t/mm³, "brittle"} from the
    material data of the Thermal add-on, or None when the elastic data are missing."""
    if props is None or not props.get("E"):
        return None
    nu = float(props.get("nu") if props.get("nu") is not None else 0.3)
    return {"E": float(props["E"]) * 1000.0, "nu": min(0.49, max(-0.99, nu)),
            "yield": None if not props.get("yield") else float(props["yield"]),
            "alpha": float(props.get("alpha") or 0.0) * 1e-6,
            "rho": float(props.get("density") or 0.0) * 1e-12,
            "brittle": bool(props.get("brittle", False))}


def _lame(E, nu):
    return E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), E / (2.0 * (1.0 + nu))


# ── meshing ───────────────────────────────────────────────────────────────────
def _weld(V, T):
    V = np.asarray(V, float).reshape(-1, 3)
    T = np.asarray(T, np.int64).reshape(-1, 3)
    scale = max(float(np.ptp(V, axis=0).max()) if len(V) else 1.0, 1e-9)
    _keys, first, inverse = np.unique(np.round(V / (scale * 1e-7)).astype(np.int64), axis=0,
                                      return_index=True, return_inverse=True)
    W = inverse.ravel()[T]
    keep = (W[:, 0] != W[:, 1]) & (W[:, 1] != W[:, 2]) & (W[:, 0] != W[:, 2])
    return V[first], W[keep]


ROUND_DIVISIONS = 24                 # elements around a full circle of a round face


def _patch_radius(gmsh, tag):
    """Smallest typical radius of curvature (mm) of a surface patch from its input triangles:
    the turn of the normal between neighbouring triangles over their distance; None for a
    flat patch."""
    try:
        types, _elements, nodes = gmsh.model.mesh.getElements(2, tag)
        node_tags, coords, _p = gmsh.model.mesh.getNodes(2, tag, includeBoundary=True)
    except Exception:
        return None
    if not len(types) or not len(node_tags):
        return None
    lookup = {int(t): i for i, t in enumerate(node_tags)}
    try:
        F = np.array([lookup[int(n)] for n in nodes[0]], np.int64).reshape(-1, 3)
    except KeyError:
        return None
    X = coords.reshape(-1, 3)
    if len(F) < 2:
        return None
    P = X[F]
    normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    area2 = np.linalg.norm(normal, axis=1)                      # twice the triangle areas
    normal /= np.maximum(area2[:, None], 1e-30)
    edges = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((edges[:, 1], edges[:, 0]))
    edges, owner = edges[order], owner[order]
    shared = np.flatnonzero(np.all(edges[1:] == edges[:-1], axis=1))
    a, b = owner[shared], owner[shared + 1]
    if not len(a):
        return None
    angle = np.arccos(np.clip((normal[a] * normal[b]).sum(1), -1.0, 1.0))
    curved = angle > 0.02
    if curved.sum() < max(2, 0.1 * len(a)):
        return None
    # the normal turns by `angle` over the widths of the two triangles across their shared edge
    # (their heights over it): radius = mean width / angle
    length = np.linalg.norm(X[edges[shared, 0]] - X[edges[shared, 1]], axis=1)
    width = 0.5 * (area2[a] + area2[b]) / np.maximum(length, 1e-30)
    radius = width[curved] / angle[curved]
    return float(np.percentile(radius, 20))


def mesh_body(V, T, size, name="body"):
    """Tetrahedra (vertices (n, 3) mm, tets (m, 4)) of a closed triangle mesh at the element
    size `size` mm (finer on curved faces)."""
    import gmsh
    W, F = _weld(V, T)
    if len(F) < 4:
        raise th.ThermalError(f"'{name}' has no closed surface to mesh.")
    gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("body")
        surface = gmsh.model.addDiscreteEntity(2)
        gmsh.model.mesh.addNodes(2, surface, np.arange(1, len(W) + 1), W.ravel())
        gmsh.model.mesh.addElementsByType(surface, 2, [], (F + 1).ravel())
        try:
            gmsh.model.mesh.classifySurfaces(math.radians(40.0), True, True, math.pi)
            gmsh.model.mesh.createGeometry()
            surfaces = [tag for _dim, tag in gmsh.model.getEntities(2)]
            # round faces (holes, shafts, fillets): finer elements near them, about
            # ROUND_DIVISIONS around a full circle, never below a fifth of the element size
            thresholds = []
            for tag in surfaces:
                radius = _patch_radius(gmsh, tag)
                if radius is None:
                    continue
                local = max(float(size) / 5.0, 2.0 * math.pi * radius / ROUND_DIVISIONS)
                if local >= 0.95 * float(size):
                    continue
                distance = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(distance, "SurfacesList", [tag])
                gmsh.model.mesh.field.setNumber(distance, "Sampling", 30)
                threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMin", local)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMax", float(size))
                gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.5 * local)
                gmsh.model.mesh.field.setNumber(threshold, "DistMax", 0.5 * local + 2.0 * float(size))
                thresholds.append(threshold)
            loop = gmsh.model.geo.addSurfaceLoop(surfaces)
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()
            if thresholds:
                smallest = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(smallest, "FieldsList", thresholds)
                gmsh.model.mesh.field.setAsBackgroundMesh(smallest)
            gmsh.option.setNumber("Mesh.MeshSizeMax", float(size))
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
            gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0 if thresholds else 1)
            gmsh.option.setNumber("Mesh.Algorithm3D", 1)      # Delaunay
            gmsh.option.setNumber("Mesh.Optimize", 1)
            gmsh.model.mesh.generate(3)
        except Exception as ex:
            raise th.ThermalError(f"'{name}' could not be meshed with tetrahedra (is its surface "
                                  f"closed and free of self-intersections?): {ex}")
        tags, coords, _param = gmsh.model.mesh.getNodes()
        element_tags, nodes = gmsh.model.mesh.getElementsByType(4)
        if not len(element_tags):
            raise th.ThermalError(f"'{name}' gave no tetrahedra (is its surface closed?).")
    finally:
        gmsh.finalize()
    tags = tags.astype(np.int64)
    lookup = np.full(int(tags.max()) + 1, -1, np.int64)
    lookup[tags] = np.arange(len(tags))
    tets = lookup[nodes.astype(np.int64)].reshape(-1, 4)
    X = coords.reshape(-1, 3)
    used = np.unique(tets)                          # drop nodes of the surface only
    renumber = np.full(len(X), -1, np.int64)
    renumber[used] = np.arange(len(used))
    X, tets = X[used], renumber[tets]
    P = X[tets]
    volume = np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0])
    flip = volume < 0                                 # positive orientation for scikit-fem
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    return X, tets


def auto_element_size(jobs, settings):
    """(element size mm, notes) for the visible bodies: about the element budget in total, and
    at least two elements through thin parts while the budget allows."""
    limit = max(2000, int(settings.get("struct_max_elements", DEFAULT_MAX_ELEMENTS)))
    volumes, thickness = [], []
    for job in jobs:
        volume = abs(th._mesh_volume(job["V"], job["T"]))
        area = th._mesh_area(job["V"], job["T"])
        volumes.append(volume)
        thickness.append(2.0 * volume / max(area, 1e-9))
    total = max(sum(volumes), 1e-9)
    # gmsh fills a volume with about 1.6 × (volume / regular tetrahedron) elements
    budget_size = (1.6 * total * 6.0 * math.sqrt(2.0) / (0.75 * limit)) ** (1.0 / 3.0)
    thin = min(thickness) / 1.6 if thickness else budget_size
    size = min(budget_size, max(thin, budget_size / 2.5))
    notes = []
    if thickness and size > min(thickness) / 1.5:
        notes.append("Some parts are thin for the element size: they get fewer than two elements "
                     "through their thickness (set a smaller element size for more accuracy).")
    return th._nice(size), notes


# ── geometry helpers ──────────────────────────────────────────────────────────
_EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))      # scikit-fem / VTK local edge order
_TRI_QUAD = (np.array([[0.445948490915965, 0.445948490915965, 0.108103018168070],
                       [0.445948490915965, 0.108103018168070, 0.445948490915965],
                       [0.108103018168070, 0.445948490915965, 0.445948490915965],
                       [0.091576213509771, 0.091576213509771, 0.816847572980459],
                       [0.091576213509771, 0.816847572980459, 0.091576213509771],
                       [0.816847572980459, 0.091576213509771, 0.091576213509771]]),
             np.array([0.223381589678011] * 3 + [0.109951743655322] * 3))   # degree 4, Σw = 1


def _tri_shape(L, order):
    """Shape functions of a triangle at barycentric points L (q, 3): (q, 3) linear or (q, 6)
    quadratic (corners, then edges 01, 12, 20)."""
    if order == 1:
        return L
    corners = L * (2.0 * L - 1.0)
    edges = 4.0 * np.stack([L[:, 0] * L[:, 1], L[:, 1] * L[:, 2], L[:, 2] * L[:, 0]], axis=1)
    return np.concatenate([corners, edges], axis=1)


def _fit_axis(normals, areas, centres):
    """(unit axis, point on it, planar) of a face from its facet normals: a flat face's axis is
    its normal through its centroid, a round face's the direction normal to all its normals
    through the fitted circle centre."""
    S = (normals[:, :, None] * normals[:, None, :] * areas[:, None, None]).sum(0)
    values, vectors = np.linalg.eigh(S)
    centroid = (centres * areas[:, None]).sum(0) / max(float(areas.sum()), 1e-30)
    if values[2] >= PIN_PLANAR * max(float(values.sum()), 1e-30):
        axis = vectors[:, 2]
        if float((normals * areas[:, None]).sum(0) @ axis) < 0:
            axis = -axis
        return axis, centroid, True
    axis = vectors[:, 0]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    u = np.cross(axis, np.eye(3)[int(np.argmin(np.abs(axis)))])
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    x, y = (centres - centroid) @ u, (centres - centroid) @ v
    A = np.stack([x, y, np.ones_like(x)], axis=1) * np.sqrt(areas)[:, None]
    b = (x * x + y * y) * np.sqrt(areas)
    try:
        cx, cy, _c = np.linalg.lstsq(A, b, rcond=None)[0] / np.array([2.0, 2.0, 1.0])
    except np.linalg.LinAlgError:
        cx = cy = 0.0
    return axis, centroid + cx * u + cy * v, False


class _Mesh:
    """Vertices p (3, nv), tetrahedra t (4, m), edges (2, ne), element edges t2e (6, m)."""

    def __init__(self, p, t, edges, t2e):
        self.p, self.t, self.edges, self.t2e = p, t, edges, t2e


# ── element matrices (straight-sided tetrahedra, vectorised over the elements) ──
_TET_QUAD = (np.array([[0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
                       [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
                       [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
                       [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685]]),
             np.full(4, 0.25))                              # degree 2, barycentric, Σw = 1


def _tet_geometry(X, t):
    """(∇L of the four corners (m, 4, 3), volumes (m,)) of straight tetrahedra."""
    Xv = X[t]
    J = np.stack([Xv[:, 1] - Xv[:, 0], Xv[:, 2] - Xv[:, 0], Xv[:, 3] - Xv[:, 0]], axis=2)
    volume = np.linalg.det(J) / 6.0
    inverse = np.linalg.inv(J)                                # rows: ∇L1, ∇L2, ∇L3
    G = np.concatenate([-inverse.sum(1, keepdims=True), inverse], axis=1)
    return G, volume


def _shape_gradients(G, L, order):
    """∇N of the element's shape functions at barycentric points L (q, 4): (m, q, n, 3)."""
    if order == 1:
        return np.broadcast_to(G[:, None], (G.shape[0], L.shape[0], 4, 3))
    out = np.empty((G.shape[0], L.shape[0], 10, 3))
    for i in range(4):
        out[:, :, i] = (4.0 * L[None, :, i, None] - 1.0) * G[:, None, i]
    for e, (i, j) in enumerate(_EDGES):
        out[:, :, 4 + e] = 4.0 * (L[None, :, j, None] * G[:, None, i] + L[None, :, i, None] * G[:, None, j])
    return out


def _reference_mass(order):
    """∫ N_a N_b dV / V of a tetrahedron (exact, from ∫ L^α dV = 6 V α! / (|α| + 3)!)."""
    if order == 1:
        return (np.ones((4, 4)) + np.eye(4)) / 20.0
    polys = []                                               # {exponents (4,): coefficient}
    for i in range(4):
        e2, e1 = [0] * 4, [0] * 4
        e2[i], e1[i] = 2, 1
        polys.append({tuple(e2): 2.0, tuple(e1): -1.0})
    for i, j in _EDGES:
        e = [0] * 4
        e[i], e[j] = 1, 1
        polys.append({tuple(e): 4.0})

    def integral(alpha):
        return 6.0 * np.prod([math.factorial(a) for a in alpha]) / math.factorial(sum(alpha) + 3)
    M = np.zeros((10, 10))
    for a, pa in enumerate(polys):
        for b, pb in enumerate(polys):
            M[a, b] = sum(ca * cb * integral(tuple(x + y for x, y in zip(ea, eb)))
                          for ea, ca in pa.items() for eb, cb in pb.items())
    return M


def _assemble_elements(model, block, chunk=4000):
    """Global sparse matrix from element blocks: block(elements) → (k, 3n, 3n)."""
    import scipy.sparse as sp
    el = model.element_nodes()
    n_dof = 3 * model.n_nodes
    dofs = (3 * el[:, :, None] + np.arange(3)).reshape(len(el), -1)
    total = None
    for start in range(0, len(el), chunk):
        part = np.arange(start, min(start + chunk, len(el)))
        Ke = block(part)
        d = dofs[part]
        rows = np.repeat(d[:, :, None], d.shape[1], axis=2).ravel()
        cols = np.repeat(d[:, None, :], d.shape[1], axis=1).ravel()
        matrix = sp.coo_matrix((Ke.ravel(), (rows, cols)), shape=(n_dof, n_dof)).tocsr()
        total = matrix if total is None else total + matrix
    return total


def _stiffness(model, lam, mu):
    X, t = model.mesh.p.T, model.mesh.t.T
    G, volume = _tet_geometry(X, t)
    L, w = _TET_QUAD
    n = 4 if model.order == 1 else 10

    def block(part):
        dN = _shape_gradients(G[part], L, model.order)                       # (k, q, n, 3)
        T1 = np.einsum("kqai,kqbj,q->kabij", dN, dN, w)                     # Σ w ∂N_a/∂x_i ∂N_b/∂x_j
        trace = np.einsum("kabii->kab", T1)
        Ke = lam[part, None, None, None, None] * T1 + \
            mu[part, None, None, None, None] * T1.transpose(0, 1, 2, 4, 3) + \
            (mu[part, None, None] * trace)[..., None, None] * np.eye(3)
        Ke *= volume[part, None, None, None, None]
        return Ke.transpose(0, 1, 3, 2, 4).reshape(len(part), 3 * n, 3 * n)
    return _assemble_elements(model, block)


def _mass(model, rho):
    X, t = model.mesh.p.T, model.mesh.t.T
    _G, volume = _tet_geometry(X, t)
    reference = _reference_mass(model.order)
    n = reference.shape[0]

    def block(part):
        Me = (rho[part] * volume[part])[:, None, None] * reference
        return np.einsum("kab,ij->kaibj", Me, np.eye(3)).reshape(len(part), 3 * n, 3 * n)
    return _assemble_elements(model, block)


def _body_load_weights(model):
    """∫ N_a dV per element node (m, n)."""
    _G, volume = _tet_geometry(model.mesh.p.T, model.mesh.t.T)
    if model.order == 1:
        return np.repeat(volume[:, None] / 4.0, 4, axis=1)
    return np.concatenate([np.repeat(-volume[:, None] / 20.0, 4, axis=1),
                           np.repeat(volume[:, None] / 5.0, 6, axis=1)], axis=1)


def _thermal_load(model, k3, delta_vertex):
    """∫ (3λ + 2μ) α ΔT ∂N_a/∂x_i dV per degree of freedom (ΔT linear in each element)."""
    G, volume = _tet_geometry(model.mesh.p.T, model.mesh.t.T)
    L, w = _TET_QUAD
    dN = _shape_gradients(G, L, model.order)                                    # (m, q, n, 3)
    delta_q = delta_vertex[model.mesh.t.T] @ L.T                               # (m, q)
    nodal = np.einsum("mqai,mq,q->mai", dN, delta_q, w) * (k3 * volume)[:, None, None]
    f = np.zeros(3 * model.n_nodes)
    el = model.element_nodes()
    for c in range(3):
        np.add.at(f, 3 * el.ravel() + c, nodal[:, :, c].ravel())
    return f


class _Model:
    """The tetrahedra of all bodies in one mesh, their nodes (corners, then edge midpoints of
    quadratic elements) and the boundary facets with the body face they lie on."""

    def __init__(self, meshes, order):
        self.order = order
        points, tets, body = [], [], []
        offset = 0
        for index, (X, T) in enumerate(meshes):
            points.append(X)
            tets.append(T + offset)
            body.append(np.full(len(T), index, np.int64))
            offset += len(X)
        X = np.concatenate(points)
        t = np.concatenate(tets)
        self.nv = len(X)
        # edges (sorted vertex pairs) and the element-to-edge table in _EDGES order
        pairs = np.sort(t[:, np.array(_EDGES)].reshape(-1, 2), axis=1)
        keys = pairs[:, 0] * self.nv + pairs[:, 1]
        self._edge_keys, first, t2e = np.unique(keys, return_index=True, return_inverse=True)
        edges = pairs[first].T                                    # (2, ne)
        self.mesh = _Mesh(X.T, t.T, edges, t2e.reshape(-1, 6).T)
        self.element_body = np.concatenate(body)
        self.vertex_body = np.zeros(self.nv, np.int64)
        self.vertex_body[t.ravel()] = np.repeat(self.element_body, 4)
        self.ne = edges.shape[1] if order == 2 else 0
        self.n_nodes = self.nv + self.ne
        self.nodes = X if order == 1 else np.concatenate([X, 0.5 * (X[edges[0]] + X[edges[1]])])
        # boundary facets (a face of one element only), outward
        faces = t[:, [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]].reshape(-1, 3)
        face_keys = np.sort(faces, axis=1)
        _u, index, counts = np.unique(face_keys, axis=0, return_index=True, return_counts=True)
        single = index[counts == 1]
        F = faces[single].copy()
        owner = single // 4
        P = X[F]
        normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        apex = X[self.mesh.t[:, owner].T].sum(1) - P.sum(1)             # the fourth corner
        inward = np.einsum("ij,ij->i", normal, apex - P[:, 0]) > 0
        F[inward] = F[inward][:, [0, 2, 1]]
        normal[inward] *= -1.0
        self.facets = F                               # (k, 3) corner vertices, outward
        self.facet_area = 0.5 * np.linalg.norm(normal, axis=1)
        self.facet_normal = normal / np.maximum(2.0 * self.facet_area[:, None], 1e-30)
        self.facet_centre = P.mean(1)
        self.facet_body = self.element_body[owner]
        self.facet_face = np.full(len(F), -1, np.int64)
        self.facet_nodes = self._facet_nodes(F)

    def edge_index(self, a, b):
        """Indices of the mesh edges between vertices a and b (arrays)."""
        keys = np.minimum(a, b) * self.nv + np.maximum(a, b)
        return np.minimum(np.searchsorted(self._edge_keys, keys), len(self._edge_keys) - 1)

    def _facet_nodes(self, F):
        """Nodes of the facets: (k, 3) linear or (k, 6) quadratic (corners, edges 01, 12, 20)."""
        if self.order == 1:
            return F
        e = [self.nv + self.edge_index(F[:, a], F[:, b]) for a, b in ((0, 1), (1, 2), (2, 0))]
        return np.concatenate([F, np.stack(e, axis=1)], axis=1)

    def element_nodes(self):
        """(m, 4) or (m, 10) nodes per element (corners, then edges in _EDGES order)."""
        t = self.mesh.t.T
        if self.order == 1:
            return t
        e = self.nv + self.mesh.t2e.T
        return np.concatenate([t, e], axis=1)


def _label_facets(model, jobs, size):
    """Body face index of every boundary facet: the nearest point sampled on the body's
    original faces whose normal agrees."""
    from scipy.spatial import cKDTree
    for index, job in enumerate(jobs):
        mine = np.flatnonzero(model.facet_body == index)
        if not len(mine) or not job["faces"]:
            continue
        T = np.asarray(job["T"], np.int64).reshape(-1, 3)
        V = np.asarray(job["V"], float)
        face_of = np.full(len(T), -1, np.int64)
        for f, tris in enumerate(job["faces"]):
            face_of[np.asarray(tris, np.int64)] = f
        points, _tris, source = th.refine_triangles(V, T, max(size / 3.0, 1e-3), with_source=True)
        P = V[T]
        tri_normal = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
        tri_normal /= np.maximum(np.linalg.norm(tri_normal, axis=1, keepdims=True), 1e-30)
        tree = cKDTree(points)
        k = min(12, len(points))
        _d, near = tree.query(model.facet_centre[mine], k=k)
        near = near.reshape(len(mine), k)
        agree = (tri_normal[source[near]] * model.facet_normal[mine][:, None, :]).sum(2) > 0.5
        choice = np.where(agree.any(1), np.argmax(agree, axis=1), 0)
        model.facet_face[mine] = face_of[source[near[np.arange(len(mine)), choice]]]


# ── supports, ties and the reduced system ─────────────────────────────────────
def _nullspace_rows(vectors):
    """(constrained basis (k, 3), free basis (3 − k, 3)) of the directions in `vectors`."""
    if not len(vectors):
        return np.zeros((0, 3)), np.eye(3)
    _u, s, vt = np.linalg.svd(np.asarray(vectors, float))
    rank = int((s > 1e-6 * max(s[0], 1e-30)).sum())
    return vt[:rank], vt[rank:]


def _ties(model, jobs, size, supported):
    """(slave node → [(master node, weight)]) for bonded contact: the boundary nodes of one
    body that lie on another body's surface follow that surface."""
    from scipy.spatial import cKDTree
    tolerance = max(TIE_TOLERANCE * size, 1e-4)
    order = model.order
    n_local = 3 if order == 1 else 6
    nodes = model.nodes
    boxes = []
    for index in range(len(jobs)):
        mine = model.facet_nodes[model.facet_body == index].ravel()
        boxes.append((nodes[mine].min(0) - tolerance, nodes[mine].max(0) + tolerance)
                     if len(mine) else None)
    slaves = {}
    for a in range(len(jobs)):
        for b in range(len(jobs)):
            if a == b or boxes[a] is None or boxes[b] is None:
                continue
            if np.any(boxes[a][1] < boxes[b][0]) or np.any(boxes[b][1] < boxes[a][0]):
                continue
            # the body with the finer surface mesh follows the other one (once per pair)
            area_a = model.facet_area[model.facet_body == a].mean()
            area_b = model.facet_area[model.facet_body == b].mean()
            if (area_a, a) > (area_b, b):
                continue
            candidates = np.unique(model.facet_nodes[model.facet_body == a].ravel())
            inside = np.all((nodes[candidates] >= boxes[b][0]) & (nodes[candidates] <= boxes[b][1]), axis=1)
            candidates = candidates[inside]
            if not len(candidates):
                continue
            target = np.flatnonzero(model.facet_body == b)
            tree = cKDTree(model.facet_centre[target])
            k = min(8, len(target))
            _d, near = tree.query(nodes[candidates], k=k)
            near = target[near.reshape(len(candidates), k)]
            P = nodes[model.facets[near]]                              # (c, k, 3, 3)
            p = nodes[candidates][:, None, :]
            e0, e1 = P[:, :, 1] - P[:, :, 0], P[:, :, 2] - P[:, :, 0]
            d = p - P[:, :, 0]
            a00, a01, a11 = (e0 * e0).sum(-1), (e0 * e1).sum(-1), (e1 * e1).sum(-1)
            b0, b1 = (d * e0).sum(-1), (d * e1).sum(-1)
            det = np.maximum(a00 * a11 - a01 * a01, 1e-30)
            s = (a11 * b0 - a01 * b1) / det
            t = (a00 * b1 - a01 * b0) / det
            margin = 1e-3
            inside_tri = (s >= -margin) & (t >= -margin) & (s + t <= 1.0 + margin)
            closest = P[:, :, 0] + s[..., None] * e0 + t[..., None] * e1
            distance = np.linalg.norm(p - closest, axis=-1)
            distance = np.where(inside_tri, distance, np.inf)
            best = np.argmin(distance, axis=1)
            rows = np.arange(len(candidates))
            ok = distance[rows, best] <= tolerance
            for c in np.flatnonzero(ok):
                node = int(candidates[c])
                if node in slaves or node in supported:
                    continue
                facet = near[c, best[c]]
                ss, tt = float(np.clip(s[c, best[c]], 0, 1)), float(np.clip(t[c, best[c]], 0, 1))
                L = np.array([[max(0.0, 1.0 - ss - tt), ss, tt]])
                weights = _tri_shape(L, order)[0]
                masters = model.facet_nodes[facet][:n_local]
                slaves[node] = [(int(m), float(w)) for m, w in zip(masters, weights) if abs(w) > 1e-12]
    return slaves


def _reduction(model, constraints, slaves):
    """Sparse T (3 n_nodes × n_reduced) with u = T q: tied nodes follow their masters, supported
    nodes keep only their free directions."""
    import scipy.sparse as sp
    n = model.n_nodes
    rows, cols, vals = list(range(n)), list(range(n)), [1.0] * n
    is_slave = np.zeros(n, bool)
    for node, masters in slaves.items():
        is_slave[node] = True
        rows[node] = None
    rows = [r for r in rows if r is not None]
    cols = list(rows)
    vals = [1.0] * len(rows)
    for node, masters in slaves.items():
        for master, weight in masters:
            rows.append(node); cols.append(master); vals.append(weight)
    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    for _round in range(8):                     # chains: a master that follows another body
        if not is_slave[W.indices].any():
            break
        W = (W @ W).tocsr()
        W.eliminate_zeros()
    if is_slave[W.indices].any():                 # a loop of ties: cut it
        W = W.tocoo()
        keep = ~is_slave[W.col]
        W = sp.csr_matrix((W.data[keep], (W.row[keep], W.col[keep])), shape=(n, n))
    independent = np.flatnonzero(~is_slave)
    # supports: free directions of the independent nodes
    r_rows, r_cols, r_vals, column = [], [], [], 0
    free_bases = {}
    for node in independent:
        node = int(node)
        vectors = constraints.get(node)
        if vectors is None:
            basis = np.eye(3)
        else:
            _constrained, basis = _nullspace_rows(vectors)
            free_bases[node] = basis
        for direction in basis:
            for c in range(3):
                if abs(direction[c]) > 1e-14:
                    r_rows.append(3 * node + c); r_cols.append(column); r_vals.append(direction[c])
            column += 1
    R = sp.csr_matrix((r_vals, (r_rows, r_cols)), shape=(3 * n, column))
    W3 = sp.kron(W, sp.identity(3, format="csr"), format="csr")
    return (W3 @ R).tocsr(), R, W3


def _rigid_modes(points):
    """(3n × 6) translations and rotations of the nodes (near null space for multigrid)."""
    n = len(points)
    B = np.zeros((3 * n, 6))
    x, y, z = (points - points.mean(0)).T / max(float(np.ptp(points)), 1e-9)
    for c in range(3):
        B[c::3, c] = 1.0
    B[0::3, 3], B[1::3, 3] = -y, x
    B[1::3, 4], B[2::3, 4] = -z, y
    B[0::3, 5], B[2::3, 5] = z, -x
    return B


def _coarse_prolongation(model, R):
    """P (n_reduced × coarse) from the corner displacements (a linear-element field: the edge
    nodes take the mean of their corners) to the reduced unknowns; columns that vanish under
    the supports and ties are left out."""
    import scipy.sparse as sp
    nv, ne = model.nv, model.ne
    if model.order == 1:
        return None
    edges = model.mesh.edges
    rows = np.concatenate([np.arange(nv), nv + np.arange(ne), nv + np.arange(ne)])
    cols = np.concatenate([np.arange(nv), edges[0], edges[1]])
    vals = np.concatenate([np.ones(nv), np.full(ne, 0.5), np.full(ne, 0.5)])
    P_nodes = sp.csr_matrix((vals, (rows, cols)), shape=(nv + ne, nv))
    P = (R.T @ sp.kron(P_nodes, sp.identity(3, format="csr"), format="csr")).tocsc()
    keep = np.flatnonzero(np.diff(P.indptr) > 0)
    return P[:, keep].tocsr()


class _TwoLevel:
    """Preconditioner of quadratic-element elasticity: symmetric Gauss–Seidel on the full
    system around an exact solve on the corner (linear-element) space."""

    def __init__(self, A, P, B=None):
        import scipy.sparse.linalg as spla
        self.A, self.P = A.tocsr(), P
        coarse = (P.T @ self.A @ P).tocsc()
        self.n_coarse = coarse.shape[0]
        if self.n_coarse <= 40000:
            factor = spla.splu(coarse, permc_spec="COLAMD")
            self.coarse_solve = factor.solve
        else:
            import pyamg
            ml = pyamg.smoothed_aggregation_solver(coarse.tocsr(), max_coarse=2000,
                                                   B=None if B is None else P.T @ B)
            self.coarse_solve = lambda r: ml.solve(r, tol=1e-3, accel="cg", maxiter=50)
        try:
            from pyamg.relaxation.relaxation import gauss_seidel
            self.smooth = lambda x, b, sweep: gauss_seidel(self.A, x, b, iterations=1, sweep=sweep)
        except Exception:                                   # Jacobi without pyamg
            inverse = 1.0 / self.A.diagonal()
            self.smooth = lambda x, b, sweep: x.__iadd__(0.6 * inverse * (b - self.A @ x))

    def __call__(self, r):
        x = np.zeros_like(r)
        self.smooth(x, r, "forward")
        x += self.P @ self.coarse_solve(self.P.T @ (r - self.A @ x))
        self.smooth(x, r, "backward")
        return x


def _solve(A, b, P=None, B=None, cancelled=None):
    """x of the symmetric positive definite system A x = b: direct when small, else conjugate
    gradients with the two-level preconditioner."""
    import scipy.sparse.linalg as spla
    n = A.shape[0]
    if n <= DIRECT_LIMIT or P is None:
        factor = spla.splu(A.tocsc(), permc_spec="COLAMD")
        return factor.solve(b), "direct"
    pre = _TwoLevel(A, P, B)
    x, iterations = _pcg(A, b, pre, 1e-10, 2000, cancelled)
    return x, f"conjugate gradients, {iterations} iterations"


def _eigen(K, M, shift, wanted, P=None, B=None, cancelled=None):
    """(ω² values, vectors, method) of the `wanted` lowest modes of K φ = ω² M φ by shift-invert
    Lanczos: (K + shift·M)⁻¹ by a direct factorisation up to EIGEN_DIRECT_LIMIT unknowns (far
    faster), above by conjugate gradients with the two-level preconditioner (less memory)."""
    import os
    import scipy.sparse.linalg as spla
    n = K.shape[0]
    A = (K + shift * M).tocsr()
    method = os.environ.get("SIMPLECAD_EIGEN", "")
    if (n <= EIGEN_DIRECT_LIMIT or P is None) and method != "iterative" or method == "direct":
        factor = spla.splu(A.tocsc(), permc_spec="COLAMD")
        op = spla.LinearOperator(A.shape, matvec=factor.solve, dtype=float)
        how = "direct"
    else:
        pre = _TwoLevel(A, P, B)
        op = spla.LinearOperator(A.shape, matvec=lambda b: _pcg(A, b, pre, 1e-9, 2000, cancelled)[0],
                                 dtype=float)
        how = "conjugate gradients"
    values, vectors = spla.eigsh(K, k=wanted, M=M, sigma=-shift, which="LM", OPinv=op)
    return values, vectors, f"shift-invert Lanczos, {how}"


def _pcg(A, b, pre, tolerance, limit, cancelled=None):
    """Preconditioned conjugate gradients to a relative residual `tolerance`."""
    x = np.zeros_like(b)
    r = b.copy()
    norm_b = max(float(np.linalg.norm(b)), 1e-300)
    z = pre(r)
    p = z.copy()
    rz = float(r @ z)
    for iteration in range(1, limit + 1):
        Ap = A @ p
        step = rz / float(p @ Ap)
        x += step * p
        r -= step * Ap
        if float(np.linalg.norm(r)) <= tolerance * norm_b:
            return x, iteration
        if cancelled is not None and iteration % 10 == 0 and cancelled():
            raise th.ThermalCancelled()
        z = pre(r)
        rz_new = float(r @ z)
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, limit


# ── the run ───────────────────────────────────────────────────────────────────
STRUCTURAL_LOADS = ("force", "pressure", "torque")
STRUCTURAL_SUPPORTS = ("support_fixed", "support_roller", "support_pin")


def solve_structural(jobs, settings, progress=None, cancelled=None, conditions=(),
                     temperatures=None):
    """Linear static stresses or natural frequencies of the bodies (`jobs` of study_jobs).
    `temperatures`: callable world points (n, 3) → °C for thermal expansion, or None."""
    _require()
    import scipy.sparse as sp
    started = time.perf_counter()

    def step(percent, text):
        if cancelled is not None and cancelled():
            raise th.ThermalCancelled()
        if progress is not None:
            progress(int(percent), text)

    modal = settings.get("struct_analysis") == "modal"
    order = 1 if int(settings.get("struct_order", 2)) == 1 else 2
    messages, notes = [], []
    mech = []
    for job in jobs:
        data = mechanical_properties(job.get("props"))
        if data is None:
            raise th.ThermalError(f"'{job['name']}' has no elastic data (Young's modulus) for its "
                                  f"material {job.get('material') or '—'}; choose a material with "
                                  "mechanical data or add them in Materials.")
        if (modal or settings.get("struct_gravity")) and data["rho"] <= 0:
            raise th.ThermalError(f"'{job['name']}': its material has no density.")
        mech.append(data)

    # 1. mesh ---------------------------------------------------------------------------
    size = float(settings.get("struct_mesh_mm") or 0.0)
    if size <= 0:
        size, size_notes = auto_element_size(jobs, settings)
        notes += size_notes
    meshes = []
    for index, job in enumerate(jobs):
        step(2 + 25 * index / max(len(jobs), 1), f"Meshing '{job['name']}' ({size:g} mm)...")
        X, T = mesh_body(job["V"], job["T"], size, job["name"])
        expected = abs(th._mesh_volume(job["V"], job["T"]))
        P = X[T]
        got = float(np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                              P[:, 3] - P[:, 0]).sum() / 6.0)
        if expected > 0 and abs(got - expected) > 0.03 * expected:
            messages.append(f"⚠ The mesh of '{job['name']}' has {got / expected:.1%} of the part's "
                            "volume (a smaller element size follows the shape better).")
        meshes.append((X, T))
    limit = int(settings.get("struct_max_elements", DEFAULT_MAX_ELEMENTS))
    count = sum(len(T) for _X, T in meshes)
    if count > 2.0 * limit:
        raise th.ThermalError(f"{count:,} elements at {size:g} mm — more than twice the limit of "
                              f"{limit:,}. Choose a larger element size or raise the limit.")
    step(28, "Building the model...")
    model = _Model(meshes, order)
    _label_facets(model, jobs, size)
    E = np.array([m["E"] for m in mech])[model.element_body]
    nu = np.array([m["nu"] for m in mech])[model.element_body]
    rho = np.array([m["rho"] for m in mech])[model.element_body]
    alpha = np.array([m["alpha"] for m in mech])[model.element_body]
    lam, mu = _lame(E, nu)
    n_dof = 3 * model.n_nodes
    step(32, f"Assembling ({len(model.mesh.t.T):,} elements, {n_dof:,} unknowns)...")
    K = _stiffness(model, lam, mu)
    f = np.zeros(n_dof)
    nodes = model.nodes

    # 2. conditions ----------------------------------------------------------------------
    def facets_of(condition):
        chosen = np.zeros(len(model.facets), bool)
        for job_index, faces in condition.get("targets", {}).items():
            chosen |= (model.facet_body == job_index) & np.isin(model.facet_face, faces)
        return np.flatnonzero(chosen)

    constraints, supported, support_entries = {}, set(), []
    load_entries = []
    n_local = 3 if order == 1 else 6
    L_q, w_q = _TRI_QUAD
    N_q = _tri_shape(L_q, order)                                  # (q, n_local)
    for condition in conditions:
        kind = condition["kind"]
        if kind not in STRUCTURAL_SUPPORTS + STRUCTURAL_LOADS:
            continue
        facets = facets_of(condition)
        entry = {"name": condition["name"], "kind": kind, "facets": facets,
                 "area_mm2": float(model.facet_area[facets].sum())}
        if not len(facets):
            messages.append(f"Condition '{condition['name']}' has no faces on the simulated bodies.")
            continue
        normals, areas = model.facet_normal[facets], model.facet_area[facets]
        if kind in STRUCTURAL_SUPPORTS:
            fn = model.facet_nodes[facets][:, :n_local]
            node_normal = np.zeros((model.n_nodes, 3))
            np.add.at(node_normal, fn.ravel(), np.repeat(normals * areas[:, None], n_local, axis=0))
            mine = np.unique(fn)
            entry["nodes"] = np.array([n for n in mine if int(n) not in supported], np.int64)
            axis = None
            if kind == "support_pin":
                axis, centre, planar = _fit_axis(normals, areas, model.facet_centre[facets])
                if planar:
                    messages.append(f"Pinned support '{condition['name']}' is on a flat face: it "
                                    "acts as a sliding support there (pick a round face, e.g. a "
                                    "hole on a pin).")
                entry["axis"], entry["centre"] = axis, centre
            for node in mine:
                node = int(node)
                n = node_normal[node]
                n = n / max(float(np.linalg.norm(n)), 1e-30)
                if kind == "support_fixed":
                    new = list(np.eye(3))
                elif kind == "support_roller" or axis is None:
                    new = [n]
                else:
                    # radial from the fitted axis (not the node normal of the faceted hole, which
                    # leans a little and would hold the rotation about the pin)
                    offset = nodes[node] - centre
                    radial = offset - (offset @ axis) * axis
                    new = [radial / max(float(np.linalg.norm(radial)), 1e-30)] if \
                        np.linalg.norm(radial) > 1e-9 else []
                    if not condition.get("pin_axial_free", False):
                        new.append(axis)
                constraints.setdefault(node, []).extend(new)
                supported.add(node)
            support_entries.append(entry)
            continue
        # loads: consistent nodal forces from the traction t(x) on the facets
        P = nodes[model.facets[facets]]                           # corners (k, 3, 3)
        X_q = np.einsum("qc,kcd->kqd", L_q, P)                    # quadrature points (k, q, 3)
        dA = areas[:, None] * w_q[None, :]                        # (k, q)
        if kind == "force":
            F = np.array([condition.get("fx", 0.0), condition.get("fy", 0.0),
                          condition.get("fz", 0.0)], float)
            traction = np.broadcast_to(F / max(float(areas.sum()), 1e-30), X_q.shape)
        elif kind == "pressure":
            traction = -float(condition.get("pressure_mpa", 0.0)) * \
                np.broadcast_to(normals[:, None, :], X_q.shape)
        else:                                                     # torque about the face axis
            axis, centre, _planar = _fit_axis(normals, areas, model.facet_centre[facets])
            if condition.get("torque_reverse"):
                axis = -axis
            r = X_q - centre
            r_perp = r - (r @ axis)[..., None] * axis
            J = float((np.einsum("kqd,kqd->kq", r_perp, r_perp) * dA).sum())
            torque = float(condition.get("torque_nm", 0.0)) * 1000.0          # N·mm
            traction = torque / max(J, 1e-30) * np.cross(axis, r_perp)
            entry["axis"], entry["centre"] = axis, centre
        nodal = np.einsum("qn,kqd,kq->knd", N_q, traction, dA)       # (k, n_local, 3)
        fn = model.facet_nodes[facets][:, :n_local]
        for c in range(3):
            np.add.at(f, 3 * fn.ravel() + c, nodal[:, :, c].ravel())
        entry["resultant"] = nodal.reshape(-1, 3).sum(0)
        load_entries.append(entry)
    if not support_entries and not modal:
        raise th.ThermalError("Nothing holds the parts: add a support (Face Condition → Fixed "
                              "support, Sliding support or Pinned support).")
    if not modal and not load_entries and not settings.get("struct_gravity") and \
            settings.get("struct_thermal", "none") == "none":
        raise th.ThermalError("There is no load: add a force, pressure or torque on faces, or "
                              "switch on gravity or thermal expansion.")

    # gravity and thermal expansion (body loads)
    gravity_force = np.zeros(3)
    if settings.get("struct_gravity") and not modal:
        share = np.zeros(model.n_nodes)                                 # ∫ ρ N dV per node
        np.add.at(share, model.element_nodes().ravel(),
                  (_body_load_weights(model) * rho[:, None]).ravel())
        f[2::3] -= GRAVITY_MM_S2 * share
        gravity_force = np.array([0.0, 0.0, -GRAVITY_MM_S2 * float(share.sum())])
    thermal_mode = settings.get("struct_thermal", "none")
    reference = float(settings.get("struct_reference_c", 20.0))
    delta_vertex = None
    if thermal_mode != "none" and not modal:
        if thermal_mode == "uniform":
            delta_vertex = np.full(model.nv, float(settings.get("struct_uniform_c", 70.0)) - reference)
        elif temperatures is not None:
            delta_vertex = np.asarray(temperatures(model.mesh.p.T), float) - reference
        if delta_vertex is not None:
            f += _thermal_load(model, (3.0 * lam + 2.0 * mu) * alpha, delta_vertex)

    # 3. bonded contact and the reduced system -------------------------------------------
    step(40, "Bonding the parts that touch...")
    slaves = _ties(model, jobs, size, supported) if len(jobs) > 1 else {}
    T_red, R_support, W_ties = _reduction(model, constraints, slaves)
    groups = _groups(model, jobs, slaves)
    held = {model.vertex_body[n] if n < model.nv else None for n in supported}
    loose = [g for g in groups if not any(b in held for b in g)]
    if loose and not modal:
        names = ", ".join(f"'{jobs[b]['name']}'" for b in loose[0][:4])
        raise th.ThermalError(f"Nothing holds {names}: it touches no supported part. Add a "
                              "support or move it so it touches a held part.")
    K_red = (T_red.T @ K @ T_red).tocsr()
    f_red = T_red.T @ f
    B_red = T_red.T @ _rigid_modes(nodes)
    n_red = K_red.shape[0]
    tie_count = len(slaves)

    # 4. solve ---------------------------------------------------------------------------
    result_modes = None
    if modal:
        import scipy.sparse.linalg as spla
        M = _mass(model, rho)
        M_red = (T_red.T @ M @ T_red).tocsr()
        count = max(1, min(30, int(settings.get("struct_modes", 6))))
        rigid = 0 if support_entries else 6 * len(groups)
        wanted = min(count + rigid, n_red - 2)
        diag_k = K_red.diagonal().mean()
        diag_m = max(M_red.diagonal().mean(), 1e-30)
        shift = 1e-6 * diag_k / diag_m
        step(55, f"Natural frequencies ({n_red:,} unknowns)...")
        values, vectors, how = _eigen(K_red, M_red, shift, wanted,
                                      _coarse_prolongation(model, R_support), B_red, cancelled)
        order_ = np.argsort(values)
        values, vectors = values[order_], vectors[:, order_]
        omega2 = np.maximum(values, 0.0)
        frequencies = np.sqrt(omega2) / (2.0 * math.pi)
        if rigid:
            frequencies, vectors = frequencies[rigid:], vectors[:, rigid:]
            notes.append(f"No supports: the parts are free; their {rigid} rigid-body modes "
                         "(0 Hz) are left out.")
        U_modes = np.stack([T_red @ vectors[:, k] for k in range(vectors.shape[1])])   # (k, 3n)
        for k in range(len(U_modes)):                 # largest displacement of a mode shape = 1
            largest = np.linalg.norm(U_modes[k].reshape(-1, 3), axis=1).max()
            U_modes[k] /= max(float(largest), 1e-300)
        result_modes = (frequencies[:count], U_modes[:count])
        u = np.zeros(n_dof)
        solve_text = how
    else:
        step(55, f"Solving ({n_red:,} unknowns)...")
        q, solve_text = _solve(K_red, f_red, _coarse_prolongation(model, R_support), B_red, cancelled)
        u = T_red @ q
        residual = np.linalg.norm(K_red @ q - f_red) / max(np.linalg.norm(f_red), 1e-30)
        if not np.all(np.isfinite(u)) or residual > 1e-4:
            raise th.ThermalError("The supports do not hold the parts in place (a part can still "
                                  "slide or turn). Add a fixed support or more supports.")
    step(85, "Stresses...")

    # 5. results -------------------------------------------------------------------------
    def stresses(u_full):
        """Von Mises stress and the stress tensor (6 components) at the corner vertices."""
        el = model.element_nodes()
        Xv = model.mesh.p.T[model.mesh.t.T]                       # (m, 4, 3)
        J = np.stack([Xv[:, 1] - Xv[:, 0], Xv[:, 2] - Xv[:, 0], Xv[:, 3] - Xv[:, 0]], axis=2)
        Jinv = np.linalg.inv(J)                                   # (m, 3, 3): rows = ∇L1..3
        G = np.concatenate([-Jinv.sum(1, keepdims=True), Jinv], axis=1)   # (m, 4, 3) ∇L_i
        Ue = u_full.reshape(-1, 3)[el]                            # (m, nodes, 3)
        grads = np.zeros((len(el), 4, 3, 3))
        for k in range(4):
            if order == 1:
                grads[:, k] = np.einsum("mia,mib->mab", Ue[:, :4], G)
                continue
            for i in range(4):
                coefficient = 4.0 * (i == k) - 1.0
                grads[:, k] += coefficient * np.einsum("ma,mb->mab", Ue[:, i], G[:, i])
            for e, (i, j) in enumerate(_EDGES):
                if j == k:
                    grads[:, k] += 4.0 * np.einsum("ma,mb->mab", Ue[:, 4 + e], G[:, i])
                if i == k:
                    grads[:, k] += 4.0 * np.einsum("ma,mb->mab", Ue[:, 4 + e], G[:, j])
        eps = 0.5 * (grads + grads.transpose(0, 1, 3, 2))
        if delta_vertex is not None:
            thermal = (alpha[:, None] * delta_vertex[model.mesh.t.T])[:, :, None, None] * np.eye(3)
            eps = eps - thermal
        tr = np.trace(eps, axis1=2, axis2=3)
        sigma = 2.0 * mu[:, None, None, None] * eps + (lam[:, None] * tr)[:, :, None, None] * np.eye(3)
        comps = np.stack([sigma[..., 0, 0], sigma[..., 1, 1], sigma[..., 2, 2],
                          sigma[..., 0, 1], sigma[..., 1, 2], sigma[..., 0, 2]], axis=-1)
        total = np.zeros((model.nv, 6))
        count = np.zeros(model.nv)
        np.add.at(total, model.mesh.t.T.ravel(), comps.reshape(-1, 6))
        np.add.at(count, model.mesh.t.T.ravel(), 1.0)
        s = total / np.maximum(count, 1.0)[:, None]
        vm = np.sqrt(0.5 * ((s[:, 0] - s[:, 1]) ** 2 + (s[:, 1] - s[:, 2]) ** 2 +
                            (s[:, 2] - s[:, 0]) ** 2) + 3.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
        return vm, s

    U = u.reshape(-1, 3)
    vm_vertex, sigma_vertex = stresses(u) if not modal else (np.zeros(model.nv), np.zeros((model.nv, 6)))
    # reactions of the supports and the force balance
    reactions = []
    applied = sum((entry["resultant"] for entry in load_entries), np.zeros(3)) + gravity_force
    if not modal:
        # nodal residuals = reactions; a tied node passes its share on to its masters
        r = (W_ties.T @ (K @ u - f)).reshape(-1, 3)
        held_total = np.zeros(3)
        for entry in support_entries:
            nodes_r = entry["nodes"]
            force = r[nodes_r].sum(0)
            centre = (model.facet_centre[entry["facets"]] * model.facet_area[entry["facets"]][:, None]).sum(0) / \
                max(entry["area_mm2"], 1e-30)
            moment = np.cross(nodes[nodes_r] - centre, r[nodes_r]).sum(0) / 1000.0        # N·m
            held_total += force
            reactions.append({"name": entry["name"], "kind": entry["kind"], "force": force,
                              "moment": moment, "area_mm2": entry["area_mm2"], "centre": centre,
                              "points": nodes[nodes_r], "nodal": r[nodes_r]})
        scale = max(float(np.linalg.norm(applied)), float(np.abs(f).sum()) / max(len(f), 1), 1e-12)
        balance = float(np.linalg.norm(held_total + applied)) / max(float(np.linalg.norm(applied)), 1e-12) \
            if np.linalg.norm(applied) > 1e-12 else float(np.linalg.norm(held_total)) / scale
    else:
        balance = 0.0

    # surfaces for the 3-D view: the boundary facets of every body, quadratic facets split in four
    surface, bodies = {}, []
    for index, job in enumerate(jobs):
        mine = np.flatnonzero(model.facet_body == index)
        fn = model.facet_nodes[mine]
        used, local = np.unique(fn, return_inverse=True)
        local = local.reshape(fn.shape)
        if order == 2:
            a, b, c, ab, bc, ca = local.T
            tris = np.concatenate([np.stack([a, ab, ca], 1), np.stack([ab, b, bc], 1),
                                   np.stack([ca, bc, c], 1), np.stack([ab, bc, ca], 1)])
        else:
            tris = local
        vertex_value = lambda values: _node_values(model, values, used)
        item = {"vertices": nodes[used], "triangles": tris, "disp": U[used],
                "vm": vertex_value(vm_vertex), "nodes": used}
        if result_modes is not None:
            item["modes"] = np.stack([mode.reshape(-1, 3)[used] for mode in result_modes[1]])
        surface[job["key"]] = item
        elements = model.element_body == index
        corner = np.unique(model.mesh.t[:, elements])
        data = mech[index]
        vm_max = float(vm_vertex[corner].max()) if len(corner) else 0.0
        hot = int(corner[np.argmax(vm_vertex[corner])]) if len(corner) else 0
        disp = np.linalg.norm(U[np.unique(model.element_nodes()[elements])], axis=1)
        P = model.mesh.p.T[model.mesh.t[:, elements].T]
        volume = float(np.einsum("ij,ij->i", np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]),
                                 P[:, 3] - P[:, 0]).sum() / 6.0)
        bodies.append({"name": job["name"], "key": job["key"], "material": job.get("material"),
                       "power": 0.0, "max_c": None,
                       "vm_max": vm_max, "vm_point": model.mesh.p[:, hot],
                       "disp_max": float(disp.max()) if len(disp) else 0.0,
                       "mass_kg": volume * data["rho"] * 1000.0,
                       "yield": data["yield"], "brittle": data["brittle"],
                       "safety": (data["yield"] / vm_max) if data["yield"] and vm_max > 1e-12 else None,
                       "elements": int(elements.sum()),
                       # the temperature table columns of the Thermal window: min, mean, max
                       "min": 0.0, "mean": 0.0, "max": vm_max})
    elapsed = time.perf_counter() - started
    summary = [f"Structural, {'natural frequencies' if modal else 'linear static'}: "
               f"{len(model.mesh.t.T):,} {'quadratic' if order == 2 else 'linear'} tetrahedra, "
               f"element size {size:g} mm, {model.n_nodes:,} nodes, {n_red:,} unknowns "
               f"({solve_text}), solved in {elapsed:.1f} s"]
    if tie_count:
        summary.append(f"Bonded contact: {tie_count:,} nodes follow the surface of a touching part.")
    if modal:
        frequencies = result_modes[0]
        summary.append("Natural frequencies: " + ", ".join(
            f"{k + 1}: {value:.4g} Hz" for k, value in enumerate(frequencies)))
    else:
        for entry in load_entries:
            F = entry["resultant"]
            summary.append(f"Load '{entry['name']}' ({_KIND_TITLES[entry['kind']]}, "
                           f"{entry['area_mm2']:.4g} mm²): resultant {_vector_text(F)} N")
        if settings.get("struct_gravity"):
            summary.append(f"Gravity: weight {abs(gravity_force[2]):.4g} N")
        if delta_vertex is not None:
            summary.append(f"Thermal expansion: {'uniform ' + format(float(settings.get('struct_uniform_c', 70.0)), 'g') + ' °C' if thermal_mode == 'uniform' else 'temperatures of the Conduction run'}, "
                           f"stress-free at {reference:g} °C (ΔT {delta_vertex.min():+.3g} … {delta_vertex.max():+.3g} K)")
        for entry in reactions:
            summary.append(f"Reaction at '{entry['name']}' ({_KIND_TITLES[entry['kind']]}): force "
                           f"{_vector_text(entry['force'])} N (|F| {np.linalg.norm(entry['force']):.4g} N), "
                           f"moment {_vector_text(entry['moment'])} N·m about the face centre")
        summary.append(f"Force balance: reactions against applied loads, difference {balance:.2e}")
        for body in bodies:
            text = f"'{body['name']}' ({body['material']}): max von Mises {body['vm_max']:.4g} MPa, " \
                   f"max displacement {body['disp_max']:.4g} mm"
            if body["safety"] is not None:
                text += f", safety factor {body['safety']:.3g} against " + \
                    ("strength" if body["brittle"] else "yield") + f" ({body['yield']:g} MPa)"
            summary.append(text)
            if body["safety"] is not None and body["safety"] < 1.0:
                messages.append(f"⚠ '{body['name']}' is stressed beyond its "
                                f"{'strength' if body['brittle'] else 'yield strength'} "
                                f"({body['vm_max']:.4g} > {body['yield']:g} MPa): a linear analysis "
                                "is not valid there; the part would yield or break.")
            elif body["safety"] is not None and body["safety"] < 1.5:
                messages.append(f"'{body['name']}': safety factor {body['safety']:.2g} — little reserve.")
        if balance > 1e-3:
            messages.append(f"⚠ Force balance off by {balance:.1%}: check the result.")
        messages.append("Stresses at sharp inside corners grow with finer meshes (a singularity); "
                        "judge stresses a little away from such corners or round the corner.")
    hot_body = max(bodies, key=lambda b: b["vm_max"]) if bodies and not modal else None
    disp_all = np.linalg.norm(U, axis=1)
    result = {"mode": "structural", "analysis": "modal" if modal else "static",
              "elements": int(len(model.mesh.t.T)), "nodes": int(model.n_nodes), "dofs": int(n_red),
              "element_size": size, "order": order, "bodies": bodies, "surface": surface,
              "reactions": reactions, "loads": [{"name": e["name"], "kind": e["kind"],
                                                 "resultant": e["resultant"], "area_mm2": e["area_mm2"]}
                                                for e in load_entries],
              "gravity": gravity_force, "balance_error": balance, "ties": tie_count,
              "vm_max": float(vm_vertex.max()) if len(vm_vertex) else 0.0,
              "disp_max": float(disp_all.max()) if len(disp_all) else 0.0,
              "hot_point": hot_body["vm_point"] if hot_body else nodes.mean(0),
              "hot_value": hot_body["vm_max"] if hot_body else 0.0,
              "frequencies": None if result_modes is None else result_modes[0],
              "volume": {"points": nodes, "elements": model.element_nodes(), "order": order,
                         "element_body": model.element_body, "disp": U,
                         "vm": _node_values(model, vm_vertex, np.arange(model.n_nodes)),
                         "sigma": _node_values(model, sigma_vertex, np.arange(model.n_nodes)),
                         "modes": None if result_modes is None else
                         np.stack([m.reshape(-1, 3) for m in result_modes[1]])},
              "seconds": elapsed, "summary": summary, "messages": notes + messages,
              # fields of a Thermal result the window looks at
              "series": None, "frame_times": np.zeros(1), "frame_steps": np.zeros(1, int),
              "hot_points": None, "hot_values": None, "conditions": [], "contacts": [],
              "thermostats": [], "liquids": [], "junctions": [], "power": 0.0}
    step(100, "Done")
    return result


_KIND_TITLES = {"force": "Force", "pressure": "Pressure", "torque": "Torque",
                "support_fixed": "Fixed support", "support_roller": "Sliding support",
                "support_pin": "Pinned support"}


def _vector_text(v):
    return "(" + ", ".join(f"{x:.4g}" for x in v) + ")"


def _node_values(model, vertex_values, nodes):
    """Values at nodes from corner values: corners as they are, edge midpoints the mean of
    their two corners."""
    vertex_values = np.asarray(vertex_values)
    nodes = np.asarray(nodes, np.int64)
    out = np.empty((len(nodes),) + vertex_values.shape[1:], vertex_values.dtype)
    corner = nodes < model.nv
    out[corner] = vertex_values[nodes[corner]]
    if (~corner).any():
        edges = model.mesh.edges[:, nodes[~corner] - model.nv]
        out[~corner] = 0.5 * (vertex_values[edges[0]] + vertex_values[edges[1]])
    return out


def _groups(model, jobs, slaves):
    """Bodies joined by bonded contact: [[body indices]]."""
    parent = list(range(len(jobs)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for node, masters in slaves.items():
        a = int(model.vertex_body[node]) if node < model.nv else \
            int(model.vertex_body[model.mesh.edges[0, node - model.nv]])
        for master, _w in masters:
            b = int(model.vertex_body[master]) if master < model.nv else \
                int(model.vertex_body[model.mesh.edges[0, master - model.nv]])
            parent[find(a)] = find(b)
    groups = {}
    for index in range(len(jobs)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def export_vtu(result, path):
    """The whole structural result for ParaView: quadratic (or linear) tetrahedra with the
    displacement, von Mises stress and stress tensor at the nodes (natural frequencies: one
    mode shape per array), the body index per element."""
    volume = result["volume"]
    order = volume["order"]
    point_data = []
    if result["analysis"] == "modal":
        for k, (frequency, mode) in enumerate(zip(result["frequencies"], volume["modes"])):
            point_data.append((f"Mode{k + 1}_{frequency:.4g}Hz", mode.astype(np.float32)))
    else:
        sigma = volume["sigma"]
        point_data += [("Displacement", volume["disp"].astype(np.float32)),
                       ("VonMises", volume["vm"].astype(np.float32)),
                       ("Stress", sigma.astype(np.float32))]   # xx yy zz xy yz xz
    comment = ("SimpleCAD Structural result: mm, N, MPa. " +
               ("Mode shapes normalised to a largest displacement of 1."
                if result["analysis"] == "modal" else
                "Stress components xx yy zz xy yz xz (MPa)."))
    th.write_vtu(path, volume["points"], volume["elements"], point_data,
                 [("Body", volume["element_body"].astype(np.int32))], comment,
                 cell_type=24 if order == 2 else 10)
    return [path]
