import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import numpy as np

def generate_rdkit_descriptors(smiles_string):
    """
    Generates RDKit molecular descriptors for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        print(f"Warning: Invalid SMILES string for descriptors: {smiles_string}")
        return None
    descriptor_values = Descriptors.CalcMolDescriptors(mol)
    # Convert dict to numpy array, handling potential errors if descriptor list is inconsistent
    # For consistency, we'll get all descriptor names first and then populate values
    descriptor_names = [desc[0] for desc in Descriptors.descList]
    descriptors_array = np.array([descriptor_values.get(name, 0.0) for name in descriptor_names])
    return descriptors_array

def generate_maccs_keys(smiles_string):
    """
    Generates MACCS keys fingerprint for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        print(f"Warning: Invalid SMILES string for MACCS keys: {smiles_string}")
        return None
    fingerprint = AllChem.GetMACCSKeysFingerprint(mol)
    return np.array(fingerprint)

def generate_ecfp_fingerprint(smiles_string, radius=2, nBits=2048):
    """
    Generates ECFP fingerprints (Morgan Fingerprints) for a given SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        print(f"Warning: Invalid SMILES string for ECFP: {smiles_string}")
        return None
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    return np.array(fingerprint)

def one_hot_encode_protein(sequence, max_len=1000):
    """
    One-hot encodes a protein sequence with a fixed maximum length.

    Args:
        sequence (str): The amino acid sequence of the protein.
        max_len (int): The maximum length for the one-hot encoded vector.
                       Sequences shorter than max_len will be padded with zeros,
                       and sequences longer than max_len will be truncated.

    Returns:
        np.ndarray: A 2D numpy array representing the one-hot encoded sequence.
                    Shape will be (max_len, num_amino_acids).
    """
    # Define the amino acid alphabet including a padding character
    amino_acids = 'ACDEFGHIKLMNPQRSTVWYX'
    # X is used for unknown or non-standard amino acids
    # Add a padding character 'Z' for sequences shorter than max_len
    aa_to_int = {aa: i for i, aa in enumerate(amino_acids)}
    num_amino_acids = len(amino_acids)

    # Initialize an empty one-hot encoded matrix
    one_hot_matrix = np.zeros((max_len, num_amino_acids), dtype=np.float32)

    # Truncate or pad the sequence
    processed_sequence = sequence[:max_len].ljust(max_len, 'X') # Pad with 'X' to handle unknown/padding

    for i, aa in enumerate(processed_sequence):
        if i < max_len:
            int_encode = aa_to_int.get(aa, aa_to_int['X']) # Use 'X' for unknown amino acids
            one_hot_matrix[i, int_encode] = 1.0

    return one_hot_matrix