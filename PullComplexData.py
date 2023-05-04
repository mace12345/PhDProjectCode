from time import perf_counter

start = perf_counter()

from ccdc import io
from ccdc.molecule import Molecule, Bond, Atom
import numpy as np
import pandas as pd


class PullData:
    def __init__(
        self, read_from_location, save_to_location, ccdc_refcode_list, metal_centre
    ):
        self.read_from_location = read_from_location
        self.save_to_location = save_to_location
        self.ccdc_refcode_list = ccdc_refcode_list
        self.metal_centre = metal_centre

        with open(self.read_from_location + self.ccdc_refcode_list, "r") as f:
            refcode_list = f.readlines()
            f.close()
        refcode_list = [i.split("\t")[0] for i in refcode_list]

        df = pd.DataFrame(
            {
                "refcode": [],
                "n_atom": [],
                "n_atom_bond_num": [],
                "d_atoms": [],
                "coordinates": [],
            }
        )

        csd_reader = io.EntryReader("CSD")
        for refcode in refcode_list:
            idx = 0
            molecule = csd_reader.molecule(refcode)
            for atom in molecule.atoms:
                if atom.atomic_symbol == self.metal_centre:
                    neighbour_atoms = atom.neighbours
                    for n_atom in neighbour_atoms:
                        d_atoms = n_atom.neighbours
                        d_atoms = [i.atomic_symbol for i in d_atoms]
                        n_atom_bond_num = 0
                        for bond in n_atom.bonds:
                            if bond.bond_type == 1:
                                n_atom_bond_num = (
                                    n_atom_bond_num + 1
                                )  # ccdc single bond value is 1
                            elif bond.bond_type == 5:
                                n_atom_bond_num = (
                                    n_atom_bond_num + 1.5
                                )  # ccdc aromatic bond value is 5
                            elif bond.bond_type == 2:
                                n_atom_bond_num = (
                                    n_atom_bond_num + 2
                                )  # ccdc double bond value is 2
                            elif bond.bond_type == 3:
                                n_atom_bond_num = (
                                    n_atom_bond_num + 3
                                )  # ccdc triple bond value is 3
                            elif bond.bond_type == 7:
                                n_atom_bond_num = np.nan
                                break
                            elif bond.bond_type == 4:
                                n_atom_bond_num = (
                                    n_atom_bond_num + 4
                                )  # ccdc quadruple bond is 4
                            elif bond.bond_type == 9:
                                n_atom_bond_num = (
                                    n_atom_bond_num + 1
                                )  # ccdc pi bond is 9, so bond order of 1
                            elif bond.bond_type == 0:
                                n_atom_bond_num = np.nan  # unknown bond type
                                break
                        try:
                            n_atom_coor = np.array(n_atom.coordinates) - np.array(
                                atom.coordinates
                            )
                        except TypeError:
                            n_atom_coor = np.nan
                        df = df.append(
                            {
                                "refcode": refcode + "_" + str(idx),
                                "n_atom": n_atom.atomic_symbol,
                                "n_atom_bond_num": n_atom_bond_num,
                                "d_atoms": d_atoms,
                                "coordinates": n_atom_coor,
                            },
                            ignore_index=True,
                        )
                    idx = idx + 1
        df.to_csv(self.save_to_location + metal_centre + "_analysis.csv")


if __name__ == "__main__":

    PullData(
        read_from_location="C:/Users/cmsma/OneDrive - University of Leeds/Samuel Mace PhD Project/Li_extraction/",
        save_to_location="C:/Users/cmsma/OneDrive - University of Leeds/Samuel Mace PhD Project/Li_extraction/",
        ccdc_refcode_list="Lithium_containing_compounds.tab",
        metal_centre="Li",
    )

end = perf_counter()
print(str(round(end - start, 3)) + " seconds to execute")
