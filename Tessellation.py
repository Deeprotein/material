##### Tessellation Code
##### By Aliye Hashemi


import os
import math
from pyhull.delaunay import DelaunayTri


# Calculate the edge length of tetrahedron.
def distance(a, b):
    dist = math.sqrt((float(a[0]) - float(b[0])) ** 2 + (float(a[1]) - float(b[1])) ** 2 +
                     (float(a[2]) - float(b[2])) ** 2)
    return str('%.3f' % dist)


# Calculate the volume of tetrahedron.
def determinant_3x3(m):
    return (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[1][0] * (m[0][1] * m[2][2] - m[0][2] * m[2][1]) +
            m[2][0] * (m[0][1] * m[1][2] - m[0][2] * m[1][1]))


def subtract(a, b):
    return (a[0] - b[0],
            a[1] - b[1],
            a[2] - b[2])


def tetrahedron_calc_volume(a, b, c, d):
    return str('%.3f' % (abs(determinant_3x3((subtract(a, b), subtract(b, c), subtract(c, d),))) / 6.0))


# Calculate the tetrahedrality of tetrahedron.
def tetrahedron_calc_tetrahedrality(l1, l2, l3, l4, l5, l6):
    list = [l1, l2, l3, l4, l5, l6]
    list = [float(i) for i in list]
    average = sum(list) / len(list)
    sum_edge = 0
    for i in range(0, 5):
        for j in range(i + 1, 6):
            sum_edge += (list[i] - list[j]) ** 2
    return str('%.3f' % (sum_edge / ((average ** 2) * 15)))


# Summary of edges length, volume and tetrahedrality of tetrahedron.
def tetrahedron_calculation(a, b, c, d):
    l1 = distance(a, b)
    l2 = distance(a, c)
    l3 = distance(a, d)
    l4 = distance(b, c)
    l5 = distance(b, d)
    l6 = distance(c, d)

    volume = tetrahedron_calc_volume(a, b, c, d)
    tetrahedrality = tetrahedron_calc_tetrahedrality(l1, l2, l3, l4, l5, l6)

    return l1, l2, l3, l4, l5, l6, volume, tetrahedrality


# Write tessellation file based on pdb files, all simplex whose edges greater than distance_cut will be removed.
def tessellation_file(input_dir, input_file, cut_distance):
    f = list(open(os.path.join(input_dir, input_file), 'r'))
    g = open(os.path.join(input_dir, "%s_tessellation.csv" % input_file[:7]), 'a')

    aa_list = []
    residue_num = []
    coordinates = []
    # dssp_list = []

    # Extract codon, residue number, coordinates, dssp info.
    for i in range(len(f)):
        aa = f[i][0]
        res = [m.split('\t', 2)[1] for m in [f[i][:-1]]][0]
        cor_x = float([m.split('\t', 3)[2] for m in [f[i][:-1]]][0])
        cor_y = float([m.split('\t', 4)[3] for m in [f[i][:-1]]][0])
        cor_z = float([m.split('\t', 5)[4] for m in [f[i][:-1]]][0])
        # dssp = f[i][-2]

        aa_list.append(aa)
        coordinates.append([cor_x, cor_y, cor_z])
        residue_num.append(res)
        # dssp_list.append(dssp)
    # print (residue_num)

    # Perform Delaunay Tessellation using pyhull, inputs are coordinates.
    tri = DelaunayTri(coordinates)
    print (tri.vertices)
    # print(aa_list)
    # Sort DelaunayTri.vertices inside list first, then entire list.
    # tri.vertices = []
    # for i in range(len(tri.vertices)):
    #     cache0_tri = sorted(tri.vertices[i])
    #     tri.vertices.append(cache0_tri)
    # tri.vertices = sorted(tri.vertices)

    # import itertools
    # # tri.vertices.sort()
    # list(tri.vertices for tri.vertices, _ in itertools.groupby(tri.vertices))
    # # tri.vertices = [list(x) for x in set(tuple(x) for x in tri.vertices)]

    # print (tri.vertices)
    # print((len(tri.vertices)))
    # print(len(aa_list))
    # print(len(tri.vertices[597]))
    ################################################### Filter #########################################################
    errors = []
    nn = 0
    for ii in range(0,(len(tri.vertices)-1)):
        # cache_tri = tri.vertices[i]
        if (len(tri.vertices[ii]) != 4):
            nn = nn + 1
            errors.append(tri.vertices[ii])
            # print(errors)
            # print (nn)
            tri.vertices.remove(tri.vertices[ii])
    print(nn)
    print(errors)
    ####################################################################################################################
    # print(aa_list[2])
    print(tri.vertices)
    # print(tri.vertices[0][3])
    # print(len(tri.vertices[0]))
    for i in range(len(tri.vertices)):
        # cache_tri = tri.vertices[i]
        # if (len(tri.vertices[i])!=4):
        #     tri.vertices.remove(tri.vertices[i])
        # Four neighbor codons as a simplex.
        simplex = aa_list[tri.vertices[i][0]] + aa_list[tri.vertices[i][1]] + aa_list[tri.vertices[i][2]] + aa_list[tri.vertices[i][3]]
        # print(simplex)
        # Rewrite simplex based on alphabet.
        simplex_sort = sorted([aa_list[tri.vertices[i][0]],aa_list[tri.vertices[i][1]],aa_list[tri.vertices[i][2]],aa_list[tri.vertices[i][3]]])
        sorted_simplex = simplex_sort[0] + simplex_sort[1] + simplex_sort[2] + simplex_sort[3]

        # Write residue number based on simplex (not sorted simplex).
        r1 = residue_num[tri.vertices[i][0]]
        r2 = residue_num[tri.vertices[i][1]]
        r3 = residue_num[tri.vertices[i][2]]
        r4 = residue_num[tri.vertices[i][3]]

        # Calculate residue distance.
        residue_distance1 = abs(int(r2) - int(r1))
        residue_distance2 = abs(int(r3) - int(r2))
        residue_distance3 = abs(int(r4) - int(r3))
        cor1 = coordinates[tri.vertices[i][0]]
        cor2 = coordinates[tri.vertices[i][1]]
        cor3 = coordinates[tri.vertices[i][2]]
        cor4 = coordinates[tri.vertices[i][3]]

        # Calculate tetrahedron edges length, volume and tetrahedrality.
        l1, l2, l3, l4, l5, l6, volume, tetrahedrality = tetrahedron_calculation(cor1, cor2, cor3, cor4)

        # Write dssp based on simplex (not sorted simplex).
        # ss1 = dssp_list[cache_tri[0]]
        # ss2 = dssp_list[cache_tri[1]]
        # ss3 = dssp_list[cache_tri[2]]
        # ss4 = dssp_list[cache_tri[3]]

        edge_list = [l1, l2, l3, l4, l5, l6]
        edge_list = [float(j) for j in edge_list]

        paras = [str(residue_distance1), str(residue_distance2), str(residue_distance3)]

        #Write to tessellation file.
        if all(edge <= cut_distance for edge in edge_list):
            g.writelines(simplex)
            for item in paras:
                g.writelines('\t' + item)
            g.writelines('\n')

    g.close()


if __name__ == '__main__':
    input_dir = '/Users/Al1ye/OneDrive - George Mason University - O365 Production/'
    pdb_file = '2v3aA02_clean.txt'
    tessellation_file(input_dir, pdb_file, 12)