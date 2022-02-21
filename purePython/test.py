# from pprint import pprint
# from chembl_webresource_client.new_client import new_client
# from rdkit import Chem
# from rdkit.Chem import AllChem


# molecule = new_client.molecule
# m1 = molecule.filter(chembl_id="CHEMBL192").only(["molecule_structures"])[0][
#     "molecule_structures"
# ]["canonical_smiles"]
# mol1 = Chem.MolFromSmiles(m1)
# f = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=1024)

# pprint(list(f))


def n(x: int):
    return x**2


def n(x: float):
    return x**3


print(n(int(3)))
print(n(float(3)))

inte = int(3)
is_int = isinstance(inte, int)
match is_int:
    case True: print(n(inte))
    case False: print(n(inte))
