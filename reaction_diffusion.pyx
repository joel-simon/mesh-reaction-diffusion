# cython: boundscheck=False
# cython: wraparound=True
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: cdivision=True

cimport numpy as np
import numpy as np
from cymesh.mesh cimport Mesh

cpdef void gray_scott(int steps, Mesh mesh, double diffU, double diffV, \
                      double F, double K):
    cdef int i = 0
    cdef int n_verts = len(mesh.verts)
    cdef int nidx
    cdef double n, uvv, u, v, lapU, lapV
    cdef double[:] U = np.zeros(len(mesh.verts))
    cdef double[:] V = np.zeros(len(mesh.verts))
    cdef double[:] dU = np.zeros(len(mesh.verts))
    cdef double[:] dV = np.zeros(len(mesh.verts))
    cdef list neighbors = []
    cdef double[:] n_neighbors = np.zeros(len(mesh.verts))

    for i, vert in enumerate(mesh.verts):
        U[i] = vert.data.get('U', 1)
        V[i] = vert.data.get('V', 0)

        neighbors.append([])
        for nvert in vert.neighbors():
            neighbors[i].append(nvert.id)

        n_neighbors[i] = len(neighbors[i])

    for _ in range(steps):
        for i in range(n_verts):
            u = U[i]
            v = V[i]
            n = n_neighbors[i]
            uvv = u*v*v
            lapU = -(n*u)
            lapV = -(n*v)

            for nidx in neighbors[i]:
                lapU += U[nidx]
                lapV += V[nidx]

            dU[i] = diffU * lapU - uvv + F*(1 - u)
            dV[i] = diffV * lapV + uvv - (K+F)*v

        for i in range(n_verts):
            U[i] += dU[i]
            V[i] += dV[i]

    for i in range(n_verts):
        mesh.verts[i].data['U'] = U[i]
        mesh.verts[i].data['V'] = V[i]
