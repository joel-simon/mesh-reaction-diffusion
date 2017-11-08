from random import random
from time import time
from cymesh.mesh import Mesh
from cymesh.view import Viewer, AnimationViewer
from cymesh import collisions
import numpy as np
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from reaction_diffusion import gray_scott

diffU = 0.01
diffV = 0.005
F = 0.02
K = 0.05
growth = .02
time_steps = 60

mesh = Mesh.from_obj('triangulated_sphere_2.obj')
view = AnimationViewer((1000, 1000))

################################################################################
# Initialize.

for _ in range(2):
    mesh.splitEdges()
    mesh.shortenEdges()

max_length = 1.8 * mesh.edges[0].length()

for v in mesh.verts:
    v.data['U'] = 1.0
    v.data['V'] = 0.0
    v.data['dU'] = 0.0
    v.data['dV'] = 0.0

    if random() < .2:
        v.data['V'] = 1

################################################################################
# Main Loop.

for i in range(time_steps):
    t1 = time()
    mesh.calculateNormals()

    for vert in mesh.verts:
        vert.data['old_x'] = vert.p[0]
        vert.data['old_y'] = vert.p[1]
        vert.data['old_z'] = vert.p[2]
        u = vert.data['U']

        if u > .5:
            vert.p[0] += vert.normal[0] * (u-.5) * growth
            vert.p[1] += vert.normal[1] * (u-.5) * growth
            vert.p[2] += vert.normal[2] * (u-.5) * growth

    t2 = time()
    c = collisions.findCollisions(mesh)

    print('findCollisions', time() - t2)

    for vi, collided in enumerate(c):
        if collided:
            vert = mesh.verts[vi]
            vert.data['collided'] = True
            vert.p[0] = vert.data['old_x']
            vert.p[1] = vert.data['old_y']
            vert.p[2] = vert.data['old_z']

    mesh.splitEdges(max_length)
    mesh.shortenEdges()

    t3 = time()
    gray_scott(100, mesh, diffU, diffV, F, K)

    print('gray_scott', time() - t3)
    print('total', time() - t1)
    print(i, len(mesh.verts))
    print()

    view.startFrame()
    for vert in mesh.verts:
        vert.data['color'] = (0.74/2, 0.87/2, vert.data['U'])
    view.drawMesh(mesh, edges=True)
    view.endFrame()

mesh.writeObj('mesh.obj')
view.mainLoop()
