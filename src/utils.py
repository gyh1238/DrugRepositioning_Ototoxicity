import numpy as np
from rdkit import Chem

def validate_mols(mols):
    """
    input: rdkit mols
    ops:
        1. kekulize
        2. remove Hs
        3. sanitize
    returns: valid_mols, valid_index
    """
    valid_mols = []
    valid_index = []
    for i in range(len(mols)):
        mol = mols[i]
        if mol:
            Chem.Kekulize(mol)
            mol = Chem.AddHs(mol)
            mol = Chem.RemoveHs(mol)
            Chem.SanitizeMol(mol)
            valid_index.append(i)
            valid_mols.append(mol)
    return valid_mols, valid_index

def validate_smiles(smiles):
    """
    input: smiles
    ops:
        1. validate_mols
        2. remove duplicates by np.unique
    returns: mols_unique, smiles_unique
    """
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_mols, valid_index = validate_mols(mols)
    print('Number of valid mols: ' + str(len(valid_mols))
                 + ', Number of discarded mols: ' + str(len(mols) - len(valid_mols)))
    valid_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
    valid_smiles_unique = np.unique(np.array(valid_smiles))
    valid_mols_unique = [Chem.MolFromSmiles(s) for s in valid_smiles_unique]
    print('Number of valid mols: ' + str(len(valid_mols))
                 + ', Number of unique mols: ' + str(len(valid_mols_unique)))
    return valid_mols_unique, valid_smiles_unique
