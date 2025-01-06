from typing import List, Tuple
import numpy as np

from deepchem.utils.typing import RDKitAtom, RDKitBond, RDKitMol
from deepchem.feat.graph_data import GraphData
from deepchem.feat.base_classes import MolecularFeaturizer
from deepchem.utils.molecule_feature_utils import get_atom_type_one_hot
from deepchem.utils.molecule_feature_utils import construct_hydrogen_bonding_info
from deepchem.utils.molecule_feature_utils import get_atom_hydrogen_bonding_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_hybridization_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_total_num_Hs_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_is_in_aromatic_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_chirality_one_hot
from deepchem.utils.molecule_feature_utils import get_atom_formal_charge
from deepchem.utils.molecule_feature_utils import get_atom_partial_charge
from deepchem.utils.molecule_feature_utils import get_atom_total_degree_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_type_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_in_same_ring_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_is_conjugated_one_hot
from deepchem.utils.molecule_feature_utils import get_bond_stereo_one_hot

ATOM_TYPE_SET = [
    'Ag', 'Al', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi',
    'Br', 'C', 'Ca', 'Cd', 'Cl', 'Co', 'Cr', 'Cu',
    'Dy', 'F', 'Fe', 'Gd', 'Ge', 'H', 'Hg', 'I',
    'In', 'K', 'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na',
    'Nd', 'Ni', 'O', 'P', 'Pb', 'Pd', 'Pt', 'S',
    'Sb', 'Se', 'Si', 'Sn', 'Sr', 'Ti', 'Tl', 'V',
    'Yb', 'Zn', 'Zr'
]

HYBRIDIZATION_SET = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]

def _construct_atom_feature(
        atom: RDKitAtom, h_bond_infos: List[Tuple[int, str]], use_chirality: bool,
        use_partial_charge: bool) -> np.ndarray:
    """Construct an atom feature from a RDKit atom object.
    Parameters
    ----------
    atom: rdkit.Chem.rdchem.Atom
    RDKit atom object
    h_bond_infos: List[Tuple[int, str]]
    A list of tuple `(atom_index, hydrogen_bonding_type)`.
    Basically, it is expected that this value is the return value of
    `construct_hydrogen_bonding_info`. The `hydrogen_bonding_type`
    value is "Acceptor" or "Donor".
    use_chirality: bool
    Whether to use chirality information or not.
    use_partial_charge: bool
    Whether to use partial charge data or not.
    Returns
    -------
    np.ndarray
    A one-hot vector of the atom feature.
    """
    atom_type = get_atom_type_one_hot(atom, allowable_set=ATOM_TYPE_SET, include_unknown_set=True)
    formal_charge = get_atom_formal_charge(atom)
    hybridization = get_atom_hybridization_one_hot(atom, allowable_set=HYBRIDIZATION_SET, include_unknown_set=True)
    acceptor_donor = get_atom_hydrogen_bonding_one_hot(atom, h_bond_infos)
    aromatic = get_atom_is_in_aromatic_one_hot(atom)
    degree = get_atom_total_degree_one_hot(atom)
    total_num_Hs = get_atom_total_num_Hs_one_hot(atom)

    radical_electrons = [float(atom.GetNumRadicalElectrons())]

    atom_feat = np.concatenate([
        atom_type, formal_charge, hybridization, acceptor_donor, aromatic, degree,
        total_num_Hs, radical_electrons
    ])

    if use_chirality:
        chirality = get_atom_chirality_one_hot(atom)
        atom_feat = np.concatenate([atom_feat, chirality])

    if use_partial_charge:
        partial_charge = get_atom_partial_charge(atom)
        atom_feat = np.concatenate([atom_feat, partial_charge])
    return atom_feat


def _construct_bond_feature(bond: RDKitBond) -> np.ndarray:
    """Construct a bond feature from a RDKit bond object.
    Parameters
    ---------
    bond: rdkit.Chem.rdchem.Bond
    RDKit bond object
    Returns
    -------
    np.ndarray
    A one-hot vector of the bond feature.
    """
    bond_type = get_bond_type_one_hot(bond, include_unknown_set=True)
    same_ring = get_bond_is_in_same_ring_one_hot(bond)
    conjugated = get_bond_is_conjugated_one_hot(bond)
    stereo = get_bond_stereo_one_hot(bond, include_unknown_set=True)
    return np.concatenate([bond_type, same_ring, conjugated, stereo])

class MolGraphConvFeaturizer(MolecularFeaturizer):
    def __init__(self,
                 use_edges: bool = False,
                 use_chirality: bool = False,
                 use_partial_charge: bool = False):
        self.use_edges = use_edges
        self.use_partial_charge = use_partial_charge
        self.use_chirality = use_chirality

    def _featurize(self, mol: RDKitMol) -> GraphData:
        if self.use_partial_charge:
            try:
                mol.GetAtomWithIdx(0).GetProp('_GasteigerCharge')
            except:
                # If partial charges were not computed
                try:
                    from rdkit.Chem import AllChem
                    AllChem.ComputeGasteigerCharges(mol)
                except ModuleNotFoundError:
                    raise ImportError("This class requires RDKit to be installed.")

        # construct atom (node) feature
        h_bond_infos = construct_hydrogen_bonding_info(mol)
        atom_features = np.asarray(
            [
                _construct_atom_feature(atom, h_bond_infos, self.use_chirality, self.use_partial_charge)
                for atom in mol.GetAtoms()
            ],
            dtype=float,
        )

        # construct edge (bond) index
        src, dest = [], []
        for bond in mol.GetBonds():
            # add edge list considering a directed graph
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            src += [start, end]
            dest += [end, start]

        # construct edge (bond) feature
        bond_features = None  # deafult None
        if self.use_edges:
            features = []
            for bond in mol.GetBonds():
                features += 2 * [_construct_bond_feature(bond)]
            bond_features = np.asarray(features, dtype=float)

        return GraphData(
            node_features=atom_features,
            edge_index=np.asarray([src, dest], dtype=int),
            edge_features=bond_features)
