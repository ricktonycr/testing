import numpy as np


def calculate_displacement_matrix(subshape, patch_centres):
    displacements_matrix = []

    for c in patch_centres:
        displacements = []
        for l in subshape:
            dx = l[0] - c[0]
            dy = l[1] - c[1]
            displacements.append(dx)
            displacements.append(dy)
        displacements_matrix.append(displacements)

    displacements_matrix = np.array(displacements_matrix).transpose()

    return displacements_matrix
