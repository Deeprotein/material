##### This code creates array B based on the corresponding Array A
##### By Aliye Hashemi


from itertools import permutations
import csv

# Template
f = open('template.csv', 'r')

# Tessellated Protein
g = open('2v3aA02_tessellation.csv', 'r')


lines = g.readlines()
simplex = []
for x in lines:
    simplex.append(x[0:4])


tess_len = len(lines)
all_perms = []
for i in range(0,tess_len):
    perms = [''.join(p) for p in permutations(simplex[i])]
    all_perms.append(perms)



all_perms_unique = []
for jj in all_perms:
    all_perms_unique.append(set(jj))


linesf = f.readlines()
simplex_template = []
for xf in linesf:
    simplex_template.append(xf[0:4])


######################################################################
## Modifying R

r = csv.reader(open('template.csv')) # Here your csv file
lines2 = list(r)


rr = csv.reader(open('2v3aA02_tessellation.csv')) # Here's your csv file
lines3 = list(rr)



my_index = []
for a in range(0,len(simplex)):   # for all simplex in the tessellation file
    for aa in all_perms_unique[a]:       # for all permutation of that simplex
        if aa in simplex_template: # if any permutation exists in the template (at least one must exist)
            # print("yes")
            # print(simplex_template.index(aa))
            my_index.append(simplex_template.index(aa)) # Show me which row in the template



#*************************************************
# Old Ds are from lines2 (template)
# New Ds are from lines3 (Tessellated Protein)
#*************************************************

c = 0
for b in my_index:
    if lines2[b][4] == '0':   # R = 0
        lines2[b][1] = lines3[c][1]  # D1
        lines2[b][2] = lines3[c][2]  # D2
        lines2[b][3] = lines3[c][3]  # D3
        lines2[b][4] = '1' # R = 1
    else:
        lines2[b][1] = (float(lines2[b][1]) * float(lines2[b][4]) + float(lines3[c][1])) / (float(lines2[b][4]) + 1)  # D1
        lines2[b][2] = (float(lines2[b][2]) * float(lines2[b][4]) + float(lines3[c][2])) / (float(lines2[b][4]) + 1)  # D2
        lines2[b][3] = (float(lines2[b][3]) * float(lines2[b][4]) + float(lines3[c][3])) / (float(lines2[b][4]) + 1)  # D3
        lines2[b][4] = str(int(lines2[b][4]) + 1)
    c = c + 1

writer = csv.writer(open('template.csv', 'w', newline=''))
writer.writerows(lines2)


##########################################################################


f.close()
g.close()