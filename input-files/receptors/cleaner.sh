grep "^ATOM" target_protein.pdb | \
grep -v "HOH\|WAT\|SO4\|CL\|NA\|MG" > target_protein_clean.pdb
