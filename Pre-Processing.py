##### Cleaning and Pre-processing before tessellation
##### By Aliye Hashemi



import pandas as pd

aa_lib = {'ARG': 'R', 'HIS': 'H', 'LYS': 'K', 'ASP': 'D', 'GLU': 'E',
          'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'CYS': 'C',
          'SEC': 'U', 'GLY': 'G', 'PRO': 'P', 'ALA': 'A', 'VAL': 'V',
          'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'PHE': 'F', 'TYR': 'Y',
          'TRP': 'W'}
aa = ['ARG', 'HIS', 'LYS', 'ASP', 'GLU','SER', 'THR', 'ASN', 'GLN', 'CYS','SEC', 'GLY', 'PRO', 'ALA', 'VAL','ILE',
      'LEU', 'MET', 'PHE', 'TYR','TRP']


file = open('2v3a.pdb')

f = open("2v3a_clean.txt", "w")

for line in file:
    # line.strip().split('/n')
    if line.startswith('ATOM'):
        if line.find('CA') != -1:
            for res in aa_lib:
                for i in aa:
                    line = line.replace(i, aa_lib[i])
            print(line)
            f.write(line)


file.close()
f.close()