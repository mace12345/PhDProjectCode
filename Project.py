# Must use the ccdc miniconda python 3.7.12 environment
from time import perf_counter

start = perf_counter()

from ccdc import io
from ccdc.molecule import Molecule, Bond, Atom
from ccdc.descriptors import MolecularDescriptors
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.optimize import basinhopping
import os


class ProcessCompounds:
    def __init__(
        self,
        read_from_location,
        save_to_location,
        excluded_vol_mol2_file=None,
        with_metal_mol2_file=None,
        mol2_file=None,
    ):
        self.read_from_location = read_from_location
        self.save_to_location = save_to_location
        self.excluded_vol_mol2_file = excluded_vol_mol2_file
        self.with_metal_mol2_file = with_metal_mol2_file
        self.mol2_file = mol2_file

        # It is important to remember that this code is built around how CSD cross miner
        # searchers and saves the ligands
        if self.excluded_vol_mol2_file != None and self.with_metal_mol2_file != None:
            # Read metal centre containing molecules with the ccdc reader
            metal_molecules = io.MoleculeReader(
                self.read_from_location + self.with_metal_mol2_file
            )
            # Load molecules from ccdc list type object to list
            metal_molecules = [molecule for molecule in metal_molecules]
            print(
                "Size of ligand set with metal centre is: " + str(len(metal_molecules))
            )
            # As it is only the ligand and not the metal centre that is needed
            # the metal centre must be removed
            # So the metal atom closest to the cartisien centre of (0, 0, 0) will be removed
            for molecule in metal_molecules:
                normal_list = []
                for atom in molecule.atoms:
                    normal_list.append(
                        [np.linalg.norm(np.array(atom.coordinates)), atom]
                    )
                # sort list in ascending order based on normal of atomic vector
                normal_list = sorted(normal_list, key=lambda x: x[0])
                # Identifie the molecule coming from a metal complex
                molecule.identifier = molecule.identifier + "_m"
                for atom in normal_list:
                    atomic_symbol = atom[1].atomic_symbol
                    # Note how these are alkaline metals and 1st row TM metals
                    # CSD only has these metals saved in its CrossMiner database
                    if (
                        atomic_symbol == "Li"
                        or atomic_symbol == "Na"
                        or atomic_symbol == "K"
                        or atomic_symbol == "Rb"
                        or atomic_symbol == "Cs"
                        or atomic_symbol == "Be"
                        or atomic_symbol == "Mg"
                        or atomic_symbol == "Ca"
                        or atomic_symbol == "Sr"
                        or atomic_symbol == "Ba"
                        or atomic_symbol == "Sc"
                        or atomic_symbol == "Ti"
                        or atomic_symbol == "V"
                        or atomic_symbol == "Cr"
                        or atomic_symbol == "Mn"
                        or atomic_symbol == "Fe"
                        or atomic_symbol == "Co"
                        or atomic_symbol == "Ni"
                        or atomic_symbol == "Cu"
                        or atomic_symbol == "Zn"
                    ):
                        molecule.remove_atom(atom[1])
                        # Break away from the for loop as we only need to remove
                        # the centre atom
                        break
            # Load the excluded volume mol2 file
            self.molecules = io.MoleculeReader(
                self.read_from_location + self.excluded_vol_mol2_file
            )
            # Load molecules from ccdc list type object to list
            self.molecules = [molecule for molecule in self.molecules]
            print(
                "Size of ligand set with excluded volume is: "
                + str(len(self.molecules))
            )
            # Final molecule list object
            self.molecules = self.molecules + metal_molecules

        elif self.excluded_vol_mol2_file != None:
            self.molecules = io.MoleculeReader(
                self.read_from_location + self.excluded_vol_mol2_file
            )
            # Load molecules from ccdc list type object to list
            self.molecules = [molecule for molecule in self.molecules]
            print(
                "Size of ligand set with excluded volume is: "
                + str(len(self.molecules))
            )

        elif self.with_metal_mol2_file != None:
            # Read metal centre containing molecules with the ccdc reader
            metal_molecules = io.MoleculeReader(
                self.read_from_location + self.with_metal_mol2_file
            )
            # Load molecules from ccdc list type object to list
            metal_molecules = [molecule for molecule in metal_molecules]
            print(
                "Size of ligand set with metal centre is: " + str(len(metal_molecules))
            )
            # As it is only the ligand and not the metal centre that is needed
            # the metal centre must be removed
            # So the metal atom closest to the cartisien centre of (0, 0, 0) will be removed
            for molecule in metal_molecules:
                normal_list = []
                for atom in molecule.atoms:
                    normal_list.append(
                        [np.linalg.norm(np.array(atom.coordinates)), atom]
                    )
                # sort list in ascending order based on normal of atomic vector
                normal_list = sorted(normal_list, key=lambda x: x[0])
                # Identifie the molecule coming from a metal complex
                molecule.identifier = molecule.identifier + "_m"
                for atom in normal_list:
                    atomic_symbol = atom[1].atomic_symbol
                    # Note how these are alkaline metals and 1st row TM metals
                    # CSD only has these metals saved in its CrossMiner database
                    if (
                        atomic_symbol == "Li"
                        or atomic_symbol == "Na"
                        or atomic_symbol == "K"
                        or atomic_symbol == "Rb"
                        or atomic_symbol == "Cs"
                        or atomic_symbol == "Be"
                        or atomic_symbol == "Mg"
                        or atomic_symbol == "Ca"
                        or atomic_symbol == "Sr"
                        or atomic_symbol == "Ba"
                        or atomic_symbol == "Sc"
                        or atomic_symbol == "Ti"
                        or atomic_symbol == "V"
                        or atomic_symbol == "Cr"
                        or atomic_symbol == "Mn"
                        or atomic_symbol == "Fe"
                        or atomic_symbol == "Co"
                        or atomic_symbol == "Ni"
                        or atomic_symbol == "Cu"
                        or atomic_symbol == "Zn"
                    ):
                        molecule.remove_atom(atom[1])
                        # Break away from the for loop as we only need to remove the centre atom
                        break
            self.molecules = metal_molecules

        elif self.mol2_file != None:
            self.molecules = io.MoleculeReader(self.read_from_location + self.mol2_file)
            # Load molecules from ccdc list type object to list
            self.molecules = [molecule for molecule in self.molecules]
            print("Size of ligand set is: " + str(len(self.molecules)))

        self.molecules_dict = {}
        for molecule in self.molecules:
            self.molecules_dict[molecule.identifier] = molecule

    def FilterCrossMinerHits(self):
        # Remove molecule if contains 3d, group 1 or group 2 metal
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            for atom in molecule.atoms:
                if atom.atomic_symbol == "Li":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Be":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Na":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Mg":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "K":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ca":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Sc":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ti":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "V":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Cr":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Mn":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Fe":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Co":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ni":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Cu":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Zn":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Rb":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Sr":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Cs":
                    remove_molecule_index.append(idx)
                    break
                elif atom.atomic_symbol == "Ba":
                    remove_molecule_index.append(idx)
                    break
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(
            "Size of ligand set after removal of metal containing compounds is: "
            + str(len(self.molecules))
        )
        # Remove molecule if carbon is not present
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            contains_carbon = False
            for atom in molecule.atoms:
                if atom.atomic_symbol == "C":
                    contains_carbon = True
                    break
            if contains_carbon == False:
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(
            "Size of ligand set after removal of compounds not containing carbon is: "
            + str(len(self.molecules))
        )
        # Add protons to molecules containing absolutly no protons at all.
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            contains_hydrogen = False
            for atom in molecule.atoms:
                if atom.atomic_symbol == "H":
                    contains_hydrogen = True
                # Sometimes the Carbon has incorrect valence so this must be corrected
                elif atom.atomic_symbol == "C":
                    valence = 0
                    for bond in atom.bonds:
                        if bond.bond_type == 1:
                            valence = valence + 1  # Single bond valence is 1
                        elif bond.bond_type == 5:  # ccdc aromatic bond value is 5
                            valence = valence + 1.5  # aromatic bond valence is 1.5
                        elif bond.bond_type == 2:
                            valence = valence + 2  # double bond valence is 2
                        elif bond.bond_type == 3:
                            valence = valence + 3  # Triple bond valence is 3
                    if valence < 4:
                        try:
                            molecule.add_hydrogens(mode="all", add_sites=True)
                            contains_hydrogen = True
                            break
                        except RuntimeError:
                            remove_molecule_index.append(idx)
                            print(
                                "Could not add H to "
                                + molecule.to_string("mol2").split("\n")[1]
                            )
                            break
            if contains_hydrogen == False:
                try:
                    molecule.add_hydrogens(mode="all", add_sites=True)
                except RuntimeError:
                    remove_molecule_index.append(idx)
                    print(
                        "Could not add H to "
                        + molecule.to_string("mol2").split("\n")[1]
                    )
        remove_molecule_index = list(set(remove_molecule_index))
        remove_molecule_index.sort()
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        # Set formal charges on all molecules
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            try:
                molecule.set_formal_charges()
            except RuntimeError:
                print(
                    "Could not add formal charges "
                    + molecule.to_string("mol2").split("\n")[1]
                )
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        # Remove molecules with the same smiles string
        # A random metal atom is added. Sometimes the CSD CrossMiner will find a ligand
        # with 2 or more possible configurations of ligand binding
        # So we do not want to loose that in the same SMILES purge
        smiles_list = []
        remove_molecule_index = []
        metal_atom = Atom("Mn", coordinates=(0, 0, 0))
        max_bond_distance = 3
        for idx, molecule in enumerate(self.molecules):
            copy_molecule = molecule.copy()
            a_id = copy_molecule.add_atom(metal_atom)
            for atom in copy_molecule.atoms[:-1]:
                normal = np.linalg.norm(np.array(atom.coordinates))
                atomic_symbol = atom.atomic_symbol
                if (
                    normal <= max_bond_distance
                    and atomic_symbol != "C"
                    and atomic_symbol != "H"
                ):
                    b_id = copy_molecule.add_bond(Bond.BondType(1), a_id, atom)
            smiles = copy_molecule.smiles
            if smiles not in smiles_list:
                smiles_list.append(smiles)
            else:
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(
            "Size of ligand set after removal of compounds with the same SMILES string is: "
            + str(len(self.molecules))
        )
        # Want to remove ligands that are actually two or more components
        remove_molecule_index = []
        for idx, molecule in enumerate(self.molecules):
            if len(molecule.components) >= 2:
                remove_molecule_index.append(idx)
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        with open(
            self.save_to_location + "filtered_ligand_set.mol2", "w"
        ) as filtered_ligand_set:
            for molecule in self.molecules:
                string = molecule.to_string("mol2")
                filtered_ligand_set.write(string + "\n")
            filtered_ligand_set.close()

    def RemoveProtonfromONS(self, atom, molecule):
        # remove protons from oxygens, nitrogens and sulphers as appropiate
        atomic_symbol = atom.atomic_symbol
        atom_neighbours = atom.neighbours
        # Testing for alcohols and thiols
        if (
            (atomic_symbol == "O" or atomic_symbol == "S")
            and len(atom_neighbours) == 2
            and (
                atom_neighbours[0].atomic_symbol == "H"
                or atom_neighbours[1].atomic_symbol == "H"
            )
        ):
            for neighbour_atom in atom_neighbours:
                if neighbour_atom.atomic_symbol == "H":
                    molecule.remove_atom(neighbour_atom)
                    atom.formal_charge = -1
                    break
        # Testing for protonated nitrogens
        elif atomic_symbol == "N":
            """
            ccdc's bond integer
            Type	Integer
            Unknown	0
            Single	1
            Double	2
            Triple	3
            Quadruple	4
            Aromatic	5
            Delocalised	7
            Pi	9
            """
            valence = 0
            for bond in atom.bonds:
                if bond.bond_type == 1:
                    valence = valence + 1  # Single bond valence is 1
                elif bond.bond_type == 5:  # ccdc aromatic bond value is 5
                    valence = valence + 1.5  # aromatic bond valence is 1.5
                elif bond.bond_type == 2:
                    valence = valence + 2  # double bond valence is 2
                elif bond.bond_type == 3:
                    valence = valence + 3  # Triple bond valence is 3
            # nitrogen is always has a positive formal charge if it has a valence of 4.
            # If there is a proton, the proton will be removed
            if valence == 4:
                for neighbour_atom in atom_neighbours:
                    if neighbour_atom.atomic_symbol == "H":
                        molecule.remove_atom(neighbour_atom)
                        atom.formal_charge = 0
                        break

    def AddHydrogensToAtom(
        self, atom, molecule, num_of_H_to_add, bond_length, new_hydrogen_idx
    ):
        c_atom_coor = np.array(atom.coordinates)
        n_atom_coors = [np.array(i.coordinates) for i in atom.neighbours]
        if num_of_H_to_add == 1:
            resultant_vector = np.array([0, 0, 0])
            for n_atom_coor in n_atom_coors:
                resultant_vector = resultant_vector + (n_atom_coor - c_atom_coor)
            norm_of_resultant_vector = np.linalg.norm(resultant_vector)
            unit_resultant_vector = resultant_vector / norm_of_resultant_vector
            new_H_coor_1 = (unit_resultant_vector * -1 * bond_length) + c_atom_coor
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_1[0], new_H_coor_1[1], new_H_coor_1[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
        elif num_of_H_to_add == 2:
            resultant_vector = np.array([0, 0, 0])
            for n_atom_coor in n_atom_coors:
                resultant_vector = resultant_vector + (n_atom_coor - c_atom_coor)
            resultant_vector = (
                resultant_vector / np.linalg.norm(resultant_vector)
            ) * bond_length
            try:
                rotation_axis = np.cross(
                    np.cross(
                        n_atom_coors[0] - c_atom_coor, n_atom_coors[1] - c_atom_coor
                    ),
                    resultant_vector,
                )
            except IndexError:
                rotation_axis = np.array([1, 0, 0])
            rotation_axis = rotation_axis / np.norm(rotation_axis)
            new_H_coors = [
                self.RotateVector(
                    vector_to_rotate=resultant_vector,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(125),
                ),
                self.RotateVector(
                    vector_to_rotate=resultant_vector,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(-125),
                ),
            ]
            new_H_coors = [
                ((new_H_coor / np.norm(new_H_coor)) * bond_length) + c_atom_coor
                for new_H_coor in new_H_coors
            ]
            for new_H_coor_1 in new_H_coors:
                new_atom_id = molecule.add_atom(
                    Atom(
                        "H",
                        coordinates=(new_H_coor_1[0], new_H_coor_1[1], new_H_coor_1[2]),
                        label="H" + str(new_hydrogen_idx),
                    )
                )
                new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
                new_hydrogen_idx = new_hydrogen_idx + 1
        elif num_of_H_to_add == 3:
            n_atom = atom.neighbours[0]
            n_n_atoms = n_atom.neighbours[0:2]
            n_n_atoms_vectors = [
                np.array(i.coordinates) - np.array(n_atom.coordinates)
                for i in n_n_atoms
            ]
            n_n_atoms_cross = np.cross(n_n_atoms_vectors[0], n_n_atoms_vectors[1])
            n_n_atoms_cross_unit = n_n_atoms_cross / np.linalg.norm(n_n_atoms_cross)
            rotation_axis = np.cross(
                n_n_atoms_cross, c_atom_coor - np.array(n_atom.coordinates)
            )
            rotation_axis = rotation_axis / np.norm(rotation_axis)
            new_H_coor_1 = (
                self.RotateVector(
                    vector_to_rotate=n_n_atoms_cross_unit * bond_length,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(-(109 - 90)),
                )
                + c_atom_coor
            )
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_1[0], new_H_coor_1[1], new_H_coor_1[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
            new_hydrogen_idx = new_hydrogen_idx + 1
            rotation_axis = np.array(n_atom.coordinates) - c_atom_coor
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            new_H_coor_2 = (
                self.RotateVector(
                    vector_to_rotate=new_H_coor_1 - c_atom_coor,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(120),
                )
                + c_atom_coor
            )
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_2[0], new_H_coor_2[1], new_H_coor_2[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
            new_hydrogen_idx = new_hydrogen_idx + 1
            new_H_coor_3 = (
                self.RotateVector(
                    vector_to_rotate=new_H_coor_1 - c_atom_coor,
                    rotation_axis=rotation_axis,
                    theta=np.deg2rad(-120),
                )
                + c_atom_coor
            )
            new_atom_id = molecule.add_atom(
                Atom(
                    "H",
                    coordinates=(new_H_coor_3[0], new_H_coor_3[1], new_H_coor_3[2]),
                    label="H" + str(new_hydrogen_idx),
                )
            )
            new_bond_id = molecule.add_bond(Bond.BondType(1), new_atom_id, atom)
            new_hydrogen_idx = new_hydrogen_idx + 1
        return new_hydrogen_idx

    def AddMetalCentre(
        self,
        metal,
        oxidation_state,
        max_bond_dist,
        output_file_name,
        number_of_bonds_formed,
        symmetric_coordination_environment=True,
        vary_protons=True,
    ):
        # Add metal centre to ligand at 0, 0, 0
        # Care is required when adding bonds between the ligand and the metal centre.
        # Atom is not a carbon, hydrogen or halogen
        # Can not be 4 coordinate Sulpher or 5 coordinate phosphours
        # Some times there will be more than 5 potential metal to ligand bonds
        # Different combinations will have to be made and then tested to see which combination gives the lowest energy in the xTB output file
        metal_atom = Atom(metal, coordinates=(0, 0, 0), formal_charge=oxidation_state)
        for molecule in self.molecules:
            m_id = molecule.add_atom(metal_atom)
            molecule.atoms[-1].label = metal + "1"
            bonding_atoms = []  # list of bonding atoms
            for atom in molecule.atoms[:-1]:
                atomic_symbol = atom.atomic_symbol
                num_neighbours = len(atom.neighbours)
                contains_H = False
                for is_H in atom.neighbours:
                    if is_H.atomic_symbol == "H":
                        contains_H = True
                        break
                normal = np.linalg.norm(np.array(atom.coordinates))
                # Add bond between metal and coordinating atoms.
                # Must filter for appropiate potential coordianting atom based on type and coordination number
                if (
                    max_bond_dist >= normal
                    and atomic_symbol != "H"
                    and atomic_symbol != "C"
                    and atomic_symbol != "F"
                    and atomic_symbol != "Cl"
                    and atomic_symbol != "Br"
                    and atomic_symbol != "I"
                    and atomic_symbol != "B"
                ):
                    if atomic_symbol == "P" and num_neighbours >= 4:
                        pass
                    elif atomic_symbol == "S" and num_neighbours >= 4:
                        pass
                    elif (
                        atomic_symbol == "N"
                        and num_neighbours >= 4
                        and contains_H == False
                    ):
                        pass
                    else:
                        bonding_atoms.append(atom)
            # Add bonds if there is the right amount of coordinating atoms to metal centre
            if len(bonding_atoms) == number_of_bonds_formed:
                for atom in bonding_atoms:
                    ProcessCompounds.RemoveProtonfromONS(
                        self, atom=atom, molecule=molecule
                    )
                    b_id = molecule.add_bond(Bond.BondType(1), m_id, atom)

            # Most of the time there are more potential coordinating atoms
            # then the specified amount of coordinating atoms we need
            # Find the magnitude of all possible combinations of vectors
            # The group of vectors with the smallest possible magnitude will be
            # bonded to the metal
            # If symmetric_coordination_environment=True
            elif len(bonding_atoms) < number_of_bonds_formed:
                print("WARNING FIX THIS: " + molecule.identifier)
            else:
                if symmetric_coordination_environment == True:
                    combs = combinations(bonding_atoms, number_of_bonds_formed)
                    magnitude = []
                    for comb in list(combs):
                        magx = 0
                        magy = 0
                        magz = 0
                        for atom in comb:
                            magx = magx + np.array(atom.coordinates)[0]
                            magy = magy + np.array(atom.coordinates)[1]
                            magz = magz + np.array(atom.coordinates)[2]
                        normal = np.linalg.norm([magx, magy, magz])
                        magnitude.append([normal, comb])
                    magnitude = sorted(magnitude, key=lambda x: x[0])
                    for atom in magnitude[0][1]:
                        ProcessCompounds.RemoveProtonfromONS(
                            self, atom=atom, molecule=molecule
                        )
                        b_id = molecule.add_bond(Bond.BondType(1), m_id, atom)

        # Add protons to molecules
        if vary_protons == True:
            molecules_to_be_protonated = []
            # find and make copys of molecules that need to be protonated
            for molecule in self.molecules:
                for atom in molecule.atoms:
                    if atom.atomic_symbol == metal:
                        n_atoms = atom.neighbours
                        for n_atom in n_atoms:
                            n_atom_type = n_atom.atomic_symbol
                            n_atom_charge = n_atom.formal_charge
                            n_atom_total_bond_order = 0
                            for bond in n_atom.bonds:
                                if bond.bond_type == 1:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1
                                    )
                                elif bond.bond_type == 2:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 2
                                    )
                                elif bond.bond_type == 3:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 3
                                    )
                                elif bond.bond_type == 4:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 4
                                    )
                                elif bond.bond_type == 5:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1.5
                                    )
                            if (
                                n_atom_type == "O"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                            elif (
                                n_atom_type == "s"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                            elif (
                                n_atom_type == "N"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                            elif (
                                n_atom_type == "P"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                molecule_copy = molecule.copy()
                                molecule_copy.identifier = (
                                    molecule_copy.identifier + "_protonated"
                                )
                                molecules_to_be_protonated.append(molecule_copy)
                                break
                        break
            # protonate the molecules
            for molecule in molecules_to_be_protonated:
                for atom in molecule.atoms:
                    if atom.atomic_symbol == metal:
                        n_atoms = atom.neighbours
                        for n_atom in n_atoms:
                            n_atom_type = n_atom.atomic_symbol
                            n_atom_charge = n_atom.formal_charge
                            n_atom_total_bond_order = 0
                            for bond in n_atom.bonds:
                                if bond.bond_type == 1:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1
                                    )
                                elif bond.bond_type == 2:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 2
                                    )
                                elif bond.bond_type == 3:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 3
                                    )
                                elif bond.bond_type == 4:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 4
                                    )
                                elif bond.bond_type == 5:
                                    n_atom_total_bond_order = (
                                        n_atom_total_bond_order + 1.5
                                    )
                            if (
                                n_atom_type == "O"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                            elif (
                                n_atom_type == "s"
                                and len(n_atom.neighbours) == 2
                                and n_atom_total_bond_order == 2
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                            elif (
                                n_atom_type == "N"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                            elif (
                                n_atom_type == "P"
                                and len(n_atom.neighbours) == 3
                                and n_atom_total_bond_order == 3
                            ):
                                n_atom.formal_charge = n_atom_charge + 1
                                self.AddHydrogensToAtom(
                                    atom=n_atom,
                                    molecule=molecule,
                                    num_of_H_to_add=1,
                                    bond_length=0.5,
                                    new_hydrogen_idx="H" + str(len(molecule.atoms)),
                                )
                        break
            # molecules have been protonated and added to the set of molecules where they will be saved
            self.molecules = self.molecules + molecules_to_be_protonated
        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            for molecule in self.molecules:
                string = molecule.to_string("mol2")
                f.write(string + "\n")
            f.close()

    def AddWaterAlongZAxis(
        self,
        Add_north_water=False,
        Add_south_water=False,
        bond_dist=1,
        output_file_name="added_water_ligands",
    ):
        for molecule in self.molecules:
            metal = molecule.atoms[-1]
            if Add_north_water == True and Add_south_water == False:
                label = "ON"
                oxygen = Atom("O", coordinates=(0, 0, bond_dist * 1), label=label)
                oxygen_id = molecule.add_atom(oxygen)
                bond_id = molecule.add_bond(Bond.BondType(1), metal, oxygen_id)
                molecule.add_hydrogens(mode="all", add_sites=True)
            elif Add_south_water == True and Add_north_water == False:
                label = "OS"
                oxygen = Atom("O", coordinates=(0, 0, bond_dist * -1), label=label)
                oxygen_id = molecule.add_atom(oxygen)
                bond_id = molecule.add_bond(Bond.BondType(1), metal, oxygen_id)
                molecule.add_hydrogens(mode="all", add_sites=True)
            elif Add_north_water == True and Add_south_water == True:
                Noxygen = Atom("O", coordinates=(0, 0, bond_dist), label="ON")
                Soxygen = Atom("O", coordinates=(0, 0, -bond_dist), label="OS")
                N_id = molecule.add_atom(Noxygen)
                S_id = molecule.add_atom(Soxygen)
                Nb_id = molecule.add_bond(Bond.BondType(1), metal, N_id)
                Sb_id = molecule.add_bond(Bond.BondType(1), metal, S_id)
                molecule.add_hydrogens(mode="all", add_sites=True)
            else:
                raise Exception(
                    "Water can only be North or South or both, 1 or -1 or 0"
                )
        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            for molecule in self.molecules:
                string = molecule.to_string("mol2")
                f.write(string + "\n")
            f.close()

    def mol2_to_openbabel(self, output_dir_name, keywords, maxiter):
        try:
            os.mkdir(self.save_to_location + output_dir_name)
        except FileExistsError:
            pass
        bat_script = (
            "@echo off\n"
            + "cd "
            + '"'
            + self.save_to_location
            + output_dir_name
            + '"'
            + "\n"
        )
        for molecule in self.molecules:
            identifier = molecule.identifier
            mol2_string = molecule.to_string("mol2")
            bat_script = (
                bat_script
                + keywords
                + " "
                + identifier
                + ".mol2 > "
                + identifier
                + "-openbabelOutput.mol2 -x "
                + str(maxiter)
                + " 2> "
                + identifier
                + "-openbabelOutput.txt\n"
            )
            with open(
                self.save_to_location + output_dir_name + "/" + identifier + ".mol2",
                "w",
            ) as f:
                f.write(mol2_string)
                f.close()
        with open(
            self.save_to_location + output_dir_name + "/openbabel_run.bat", "w"
        ) as f:
            f.write(bat_script)
            f.close()
        # Optimise using openbabel UFF in the command line
        # 'conda activate rdkit-env' This is where obminimize function exists
        # conda activate C:\ProgramData\Anaconda3\envs\rdkit-env (manually)
        # run the PreOptOutput file (manually)
        # The output files from openbabel UFF loose formal charges.
        # This should not matter as we will already have used ccdc
        # to work out the formal charges

    def RemoveImpossibleComplexes(self, metal_centre, output_file_name):
        # removes complexes where the non-bonding atoms to the metal centre are closer than the bonding atoms to the metal centre
        remove_molecule_index = []
        for index, molecule in enumerate(self.molecules):
            m_atom = None
            for atom in molecule.atoms:
                if atom.atomic_symbol == metal_centre:
                    m_atom = atom
                    break
            atoms = molecule.atoms
            coor_atom_BD = []
            coor_atom_labels = []
            for atom in m_atom.neighbours:
                coor_atom_BD.append(
                    np.linalg.norm(
                        np.array(atom.coordinates) - np.array(m_atom.coordinates)
                    )
                )
                coor_atom_labels.append(atom.label)
            remove_atom_index = []
            for coor_atom_label in coor_atom_labels:
                for idx, atom in enumerate(atoms):
                    if atom.label == coor_atom_label:
                        remove_atom_index.append(idx)
            for idx, atom in enumerate(atoms):
                if atom.label == m_atom.label:
                    remove_atom_index.append(idx)
            remove_atom_index.reverse()
            for idx in remove_atom_index:
                del atoms[idx]

            non_coor_atom_BD = []
            for atom in atoms:
                non_coor_atom_BD.append(
                    np.linalg.norm(
                        np.array(atom.coordinates) - np.array(m_atom.coordinates)
                    )
                )
            try:
                if min(non_coor_atom_BD) < min(coor_atom_BD):
                    remove_molecule_index.append(index)
            except ValueError:
                pass
        remove_molecule_index.reverse()
        for idx in remove_molecule_index:
            del self.molecules[idx]
        print(len(self.molecules))
        new_mol_file = ""
        for molecule in self.molecules:
            new_mol_file = new_mol_file + molecule.to_string("mol2")
        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            f.write(new_mol_file)
            f.close()


class AnalyseCompounds:
    def __init__(
        self,
        read_from_location,
        save_to_location,
        xTB_output_files_dir=None,
        mopac_output_files_dir=None,
        mol2_files=None,
        orca_output_files_dir=None,
        OB_output_files_dir=None,
        OB_input_name=None,
        OB_output_name=None,
        metal_centre=None,
        metal_Ox_state=None,
        total_run_time=None,
        xyz_files=False,
        pre_mol2_files=None,
        output_xTB_file_name=None,
        output_mol2_file_name=None,
        g09_TS_output_log_file=None,
        pull_mullikin_pop=False,
        pull_loewdin_pop=False,
        pull_mayer_pop=False,
    ):

        self.read_from_location = read_from_location
        self.save_to_location = save_to_location
        self.xTB_output_files_dir = xTB_output_files_dir
        self.mopac_output_files_dir = mopac_output_files_dir
        self.mol2_files = mol2_files
        self.mol2_molecules = None
        self.xyz_molecules = None
        self.orca_output_files_dir = orca_output_files_dir
        self.time_taken_list = None
        self.OB_output_files_dir = OB_output_files_dir
        self.OB_input_name = OB_input_name
        self.OB_output_name = OB_output_name
        self.input_molecules = []
        self.output_molecules = []
        self.mol2_dict = {}
        self.xyz_files = xyz_files
        self.pre_mol2_files = pre_mol2_files
        self.output_xTB_file_name = output_xTB_file_name
        self.metal_centre = metal_centre
        self.output_mol2_file_name = output_mol2_file_name
        self.g09_TS_output = g09_TS_output_log_file
        self.atomic_number_to_atomic_symbol_dict = {
            "1": "H",
            "2": "He",
            "3": "Li",
            "4": "Be",
            "5": "B",
            "6": "C",
            "7": "N",
            "8": "O",
            "9": "F",
            "10": "Ne",
            "11": "Na",
            "12": "Mg",
            "13": "Al",
            "14": "Si",
            "15": "P",
            "16": "S",
            "17": "Cl",
            "18": "Ar",
            "50": "Sn",
        }

        if self.OB_output_files_dir != None:
            # This code rebuilds the molecules after they have been pre-optimised in open babel using the UFF model
            files = [
                file.split("-")[0]
                for file in os.listdir(
                    self.read_from_location + self.OB_output_files_dir
                )
                if file.split("-")[-1] == self.OB_output_name + ".mol2"
            ]
            mol2_string = ""
            time_string = "identifier,time / sec\n"
            for file in files:
                try:
                    with open(
                        self.read_from_location
                        + self.OB_output_files_dir
                        + "/"
                        + file
                        + "-"
                        + self.OB_output_name
                        + ".mol2",
                        "r",
                    ) as f:
                        output_file = f.readlines()
                        f.close()
                    output_molecule = Molecule(identifier=file)
                    input_molecule = io.MoleculeReader(
                        self.read_from_location
                        + self.OB_output_files_dir
                        + "/"
                        + file
                        + self.OB_input_name
                        + ".mol2"
                    )[0]
                    # add atoms to the output molecule
                    atom_id_dict = {}
                    for line in output_file:
                        line = line.split(" ")
                        line = [i for i in line if i != ""]
                        if line[0] == "ATOM":
                            label = line[2]
                            formal_charge = int(
                                input_molecule.atom(label).formal_charge
                            )
                            atomic_symbol = line[-2]
                            x_coor = float(line[6])
                            y_coor = float(line[7])
                            z_coor = float(line[8])
                            if formal_charge != 0:
                                try:
                                    atom = Atom(
                                        atomic_symbol=atomic_symbol,
                                        coordinates=(x_coor, y_coor, z_coor),
                                        label=label,
                                        formal_charge=formal_charge,
                                    )
                                except RuntimeError:
                                    atomic_symbol = line[-1][:-3]
                                    atom = Atom(
                                        atomic_symbol=atomic_symbol,
                                        coordinates=(x_coor, y_coor, z_coor),
                                        label=label,
                                        formal_charge=formal_charge,
                                    )
                            else:
                                atom = Atom(
                                    atomic_symbol=atomic_symbol,
                                    coordinates=(x_coor, y_coor, z_coor),
                                    label=label,
                                )
                            atom_id = output_molecule.add_atom(atom)
                            atom_id_dict[label] = atom_id
                    # add bonds to the output molecule from the input molecule
                    for atom in input_molecule.atoms:
                        for bond in atom.bonds:
                            label1 = bond.atoms[0].label
                            label2 = bond.atoms[1].label
                            """
                            ccdc's bond integer
                            Type	Integer
                            Unknown	0
                            Single	1
                            Double	2
                            Triple	3
                            Quadruple	4
                            Aromatic	5
                            Delocalised	7
                            Pi	9
                            """
                            bond_type = str(bond.bond_type)
                            if bond_type == "Unknown":
                                bond_type = 0
                            elif bond_type == "Single":
                                bond_type = 1
                            elif bond_type == "Double":
                                bond_type = 2
                            elif bond_type == "Triple":
                                bond_type = 3
                            elif bond_type == "Quadruple":
                                bond_type = 4
                            elif bond_type == "Aromatic":
                                bond_type = 5
                            elif bond_type == "Delocalised":
                                bond_type = 7
                            elif bond_type == "Pi":
                                bond_type = 9
                            try:
                                bond_id = output_molecule.add_bond(
                                    Bond.BondType(bond_type),
                                    atom_id_dict[label1],
                                    atom_id_dict[label2],
                                )
                            except RuntimeError:
                                pass
                    mol2_string = mol2_string + output_molecule.to_string("mol2")
                except RuntimeError:
                    print(file)
                except KeyError:
                    print(file)
                # get time taken to do calculation
                try:
                    with open(
                        self.read_from_location
                        + self.OB_output_files_dir
                        + "/"
                        + file
                        + "-"
                        + self.OB_output_name
                        + ".txt",
                        "r",
                    ) as f:
                        output_file = f.readlines()
                        f.close()
                    time = str(output_file[-1].split(": ")[1].split("s")[0])
                    time_string = time_string + file + "," + time + "\n"
                except RuntimeError:
                    print(file)
                except KeyError:
                    print(file)
            with open(self.save_to_location + OB_output_files_dir + ".mol2", "w") as f:
                f.write(mol2_string)
                f.close()
            with open(
                self.save_to_location + OB_output_files_dir + "_time_taken.csv", "w"
            ) as f:
                f.write(time_string)
                f.close()

        if self.mol2_files != None:
            self.mol2_molecules = io.MoleculeReader(
                self.read_from_location + self.mol2_files
            )
            self.mol2_molecules = [i for i in self.mol2_molecules]
            for molecule in self.mol2_molecules:
                self.mol2_dict[molecule.identifier] = molecule

        if self.pre_mol2_files != None:
            self.mol2_molecules = io.MoleculeReader(
                self.save_to_location + self.pre_mol2_files
            )
            self.mol2_molecules = [i for i in self.mol2_molecules]
            for molecule in self.mol2_molecules:
                self.mol2_dict[molecule.identifier] = molecule

        if self.xTB_output_files_dir != None and self.output_xTB_file_name != None:
            dirs = os.listdir(self.read_from_location + self.xTB_output_files_dir)[:-1]
            mol2_string = ""
            time_string = "identifier,time / sec\n"
            for dir in dirs:
                try:
                    with open(
                        self.read_from_location
                        + self.xTB_output_files_dir
                        + "/"
                        + dir
                        + "/xtbopt.xyz",
                        "r",
                    ) as f:
                        file = f.readlines()
                        f.close()
                    identifier = dir
                    mol = Molecule(identifier=identifier)
                    template_molecule = self.mol2_dict[identifier]
                    # add atoms
                    bond_dict = {}
                    for new_atom, old_atom in zip(file[2:], template_molecule.atoms):
                        new_atom = [i for i in new_atom.split(" ") if i != ""]
                        new_atom_type = new_atom[0]
                        x_coor = float(new_atom[1])
                        y_coor = float(new_atom[2])
                        z_coor = float(new_atom[3])
                        label = old_atom.label
                        charge = old_atom.formal_charge
                        a = Atom(
                            atomic_symbol=new_atom_type,
                            coordinates=(x_coor, y_coor, z_coor),
                            label=label,
                            formal_charge=charge,
                        )
                        a_id = mol.add_atom(a)
                        bond_dict[label] = a_id
                    # add bonds
                    for bond in template_molecule.bonds:
                        atoms = bond.atoms
                        mol.add_bond(
                            bond_type=bond.bond_type,
                            atom1=bond_dict[atoms[0].label],
                            atom2=bond_dict[atoms[1].label],
                        )
                    mol2_string = mol2_string + mol.to_string("mol2")
                    with open(
                        self.read_from_location
                        + self.xTB_output_files_dir
                        + "/"
                        + dir
                        + "/output.txt",
                        "r",
                        encoding="utf-8",
                    ) as f:
                        file = f.readlines()
                        f.close()
                    time = [i for i in file[-3].split(" ") if i != ""]
                    seconds = float(time[8])
                    minuets = int(time[6]) * 60
                    hours = int(time[4]) * 60 * 60
                    days = int(time[2]) * 60 * 60 * 24
                    total_seconds = seconds + float(minuets + hours + days)
                    time_string = time_string + dir + "," + str(total_seconds) + "\n"
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            with open(
                self.save_to_location + self.output_xTB_file_name + ".mol2", "w"
            ) as f:
                f.write(mol2_string)
                f.close()
            with open(
                self.save_to_location + self.output_xTB_file_name + "_time_taken.csv",
                "w",
            ) as f:
                f.write(time_string)
                f.close()

        if self.mopac_output_files_dir != None:
            dirs = os.listdir(self.read_from_location + self.mopac_output_files_dir)
            dirs = [i for i in dirs if i.split(".")[1] == "arc"]
            xyz_file = ""
            self.xyz_molecules = []
            for dir in dirs:
                with open(
                    self.read_from_location + self.mopac_output_files_dir + "/" + dir,
                    "r",
                ) as f:
                    file = f.readlines()
                    f.close()
                index = 0
                for idx, line in enumerate(file):
                    if line == "          FINAL GEOMETRY OBTAINED\n":
                        index = idx
                        break
                formal_charge = int(file[index + 1].split("=")[1].split(" ")[0])
                xyz_block = file[index + 4 :]
                xyz_string = (
                    str(len(xyz_block[:-1]))
                    + "\n"
                    + dir.split(".")[0]
                    + " formal charge = "
                    + str(formal_charge)
                    + "\n"
                )
                molecule = Molecule(identifier=dir.split(".")[0])
                for line in xyz_block[:-1]:
                    line = line.split(" ")
                    line = [i for i in line if i != ""]
                    atomic_symbol = line[0]
                    x_coor = float(line[1])
                    y_coor = float(line[3])
                    z_coor = float(line[5])
                    ai = molecule.add_atom(
                        Atom(
                            atomic_symbol=atomic_symbol,
                            coordinates=(x_coor, y_coor, z_coor),
                        )
                    )
                    xyz_string = (
                        xyz_string
                        + atomic_symbol
                        + " "
                        + str(x_coor)
                        + " "
                        + str(y_coor)
                        + " "
                        + str(z_coor)
                        + "\n"
                    )
                xyz_file = xyz_file + xyz_string
                self.xyz_molecules.append(molecule)
            with open(
                self.save_to_location + self.mopac_output_files_dir + ".xyz", "w"
            ) as f:
                f.write(xyz_file)
                f.close()

        if self.orca_output_files_dir != None:
            files = os.listdir(self.read_from_location + self.orca_output_files_dir)
            out_files_list = [
                file.split("-ORCA")[0]
                for file in files
                if "-" in file and file.split("-")[-1] == "ORCAInput.txt"
            ]
            print(out_files_list)
            mol2_string = ""
            place_holders = [np.nan for _ in range(len(out_files_list))]
            df = pd.DataFrame(
                {
                    "identifier": out_files_list,
                    "optimisation_successful": place_holders,
                    "IR_freq_analysis_successful": place_holders,
                    "running": place_holders,
                    "single_point_energy": place_holders,
                    "gibbs_free_en_" + self.metal_centre: place_holders,
                }
            )
            df.set_index("identifier", inplace=True)
            for file in out_files_list:
                # Get xyz coordinates
                # CARTESIAN COORDINATES (ANGSTROEM)\n
                # Get electronic energy
                # FINAL SINGLE POINT ENERGY      -931.057870311885
                # If frequency calculation get smallest vibration
                # If frequency calculation get thermodynamic energy
                # Get time of calculation to complete if successful
                # TOTAL RUN TIME: 0 days 1 hours 4 minutes 44 seconds 647 msec\n
                try:
                    with open(
                        self.read_from_location
                        + self.orca_output_files_dir
                        + "/"
                        + file
                        + "-ORCAOutput.out",
                        "r",
                    ) as f:
                        out_file = f.readlines()
                        f.close()
                    xyz_coordinates = []
                    single_point_energy = None
                    thermodynamic_energy = None
                    vibrational_frequency = None
                    calculation_time = None
                    for idx, line in enumerate(out_file):
                        if line == "CARTESIAN COORDINATES (ANGSTROEM)\n":
                            xyz_coordinates = []
                            for coor in out_file[idx + 2 :]:
                                if coor != "\n":
                                    xyz_coordinates.append(coor[2:])
                                else:
                                    break
                        elif (
                            line.split(" ")[0] == "FINAL"
                            and line.split(" ")[1] == "SINGLE"
                            and line.split(" ")[2] == "POINT"
                            and line.split(" ")[3] == "ENERGY"
                        ):
                            try:
                                single_point_energy = float(line.split(" ")[-1])
                                df.loc[
                                    file, "single_point_energy"
                                ] = single_point_energy
                            except ValueError:
                                pass
                        elif (
                            line.split(" ")[0] == "Final"
                            and line.split(" ")[1] == "Gibbs"
                            and line.split(" ")[2] == "free"
                            and line.split(" ")[3] == "energy"
                        ):
                            thermodynamic_energy = float(line.split(" ")[-2])
                            df.loc[
                                file, "gibbs_free_en_" + self.metal_centre
                            ] = thermodynamic_energy
                        elif (
                            line.split(" ")[0] == "VIBRATIONAL"
                            and line.split(" ")[1] == "FREQUENCIES\n"
                        ):
                            vibrational_frequency = out_file[idx + 11]
                            vibrational_frequency = float(
                                [
                                    i
                                    for i in vibrational_frequency.split(" ")
                                    if i != ""
                                ][1]
                            )
                            df.loc[
                                file, "IR_freq_analysis_successful"
                            ] = vibrational_frequency
                        elif (
                            line.split(" ")[0] == "TOTAL"
                            and line.split(" ")[1] == "RUN"
                            and line.split(" ")[2] == "TIME:"
                        ):
                            time = [i for i in line.split(" ") if i != ""]
                            time = (
                                (int(time[3]) * 24 * 60 * 60)
                                + (int(time[5]) * 60 * 60)
                                + (int(time[7]) * 60)
                                + int(time[9])
                            )
                            df.loc[file, "optimisation_successful"] = time
                    # rebuild complexes to give a .mol2 file that can then be analysed
                    identifier = file
                    mol = Molecule(identifier=identifier)
                    template_molecule = self.mol2_dict[identifier]
                    # add atoms
                    bond_dict = {}
                    for new_atom, old_atom in zip(
                        xyz_coordinates, template_molecule.atoms
                    ):
                        new_atom = [i for i in new_atom.split(" ") if i != ""]
                        new_atom_type = new_atom[0]
                        x_coor = float(new_atom[1])
                        y_coor = float(new_atom[2])
                        z_coor = float(new_atom[3])
                        label = old_atom.label
                        charge = old_atom.formal_charge
                        a = Atom(
                            atomic_symbol=new_atom_type,
                            coordinates=(x_coor, y_coor, z_coor),
                            label=label,
                            formal_charge=charge,
                        )
                        a_id = mol.add_atom(a)
                        bond_dict[label] = a_id
                    # add bonds
                    for bond in template_molecule.bonds:
                        atoms = bond.atoms
                        mol.add_bond(
                            bond_type=bond.bond_type,
                            atom1=bond_dict[atoms[0].label],
                            atom2=bond_dict[atoms[1].label],
                        )
                    mol2_string = mol2_string + mol.to_string("mol2")
                except FileNotFoundError:
                    pass
                except RuntimeError:
                    pass

            with open(
                self.save_to_location + self.output_mol2_file_name + "_ORCAOutput.mol2",
                "w",
            ) as f:
                f.write(mol2_string)
                f.close()

            df.to_csv(
                self.save_to_location + self.output_mol2_file_name + "_ORCAOutput.csv",
            )

            self.mol2_molecules = io.MoleculeReader(
                self.save_to_location + self.output_mol2_file_name + "_ORCAOutput.mol2"
            )
            self.mol2_molecules = [i for i in self.mol2_molecules]
            for molecule in self.mol2_molecules:
                self.mol2_dict[molecule.identifier] = molecule

            if pull_loewdin_pop == True:
                mol2_string = ""
                for file in out_files_list:
                    try:
                        with open(
                            self.read_from_location
                            + self.orca_output_files_dir
                            + "/"
                            + file
                            + "-ORCAOutput.out",
                            "r",
                        ) as f:
                            out_file = f.readlines()
                            f.close()
                        for idx, line in enumerate(out_file):
                            if line == "LOEWDIN ATOMIC CHARGES\n":
                                charge_list = []
                                for atomic_charge in out_file[idx + 2 :]:
                                    if atomic_charge == "\n":
                                        break
                                    charge_list.append(
                                        float(atomic_charge.split(" ")[-1])
                                    )
                                break
                        for charge, atom in zip(
                            charge_list, self.mol2_dict[file].atoms
                        ):
                            atom.partial_charge = charge
                        mol2_string = mol2_string + self.mol2_dict[file].to_string(
                            "mol2"
                        )
                    except FileNotFoundError:
                        pass
                    except RuntimeError:
                        pass
                with open(
                    self.save_to_location
                    + self.output_mol2_file_name
                    + "_LoewdinCharges_ORCAOutput.mol2",
                    "w",
                ) as f:
                    f.write(mol2_string)
                    f.close()

        if self.xyz_files == True:
            # this code converts the xyz files into a mol2 file
            xyz_files = [
                i
                for i in os.listdir(self.read_from_location)
                if i.split(".")[1] == "xyz"
            ]
            self.mol2_file = ""
            for xyz_file in xyz_files:
                with open(self.read_from_location + xyz_file, "r") as f:
                    file = f.readlines()
                    f.close()
                identifier = xyz_file.split("-")[0]
                mol = Molecule(identifier=identifier)
                template_molecule = self.mol2_dict[identifier]
                # add atoms
                bond_dict = {}
                for new_atom, old_atom in zip(file[2:], template_molecule.atoms):
                    new_atom = [i for i in new_atom.split(" ") if i != ""]
                    new_atom_type = new_atom[0]
                    x_coor = float(new_atom[1])
                    y_coor = float(new_atom[2])
                    z_coor = float(new_atom[3])
                    label = old_atom.label
                    charge = old_atom.formal_charge
                    a = Atom(
                        atomic_symbol=new_atom_type,
                        coordinates=(x_coor, y_coor, z_coor),
                        label=label,
                        formal_charge=charge,
                    )
                    a_id = mol.add_atom(a)
                    bond_dict[label] = a_id
                # add bonds
                for bond in template_molecule.bonds:
                    atoms = bond.atoms
                    mol.add_bond(
                        bond_type=bond.bond_type,
                        atom1=bond_dict[atoms[0].label],
                        atom2=bond_dict[atoms[1].label],
                    )
                self.mol2_file = self.mol2_file + mol.to_string("mol2")
            with open(
                self.save_to_location + self.output_mol2_file_name + ".mol2", "w"
            ) as f:
                f.write(self.mol2_file)
                f.close()

        if self.g09_TS_output != None:
            with open(self.read_from_location + self.g09_TS_output, "r") as f:
                g09_file = f.readlines()
                f.close()
            TS_mol2_file = ""
            num_of_atoms = None
            old_scf_energy = 0
            scf_energy = 0
            for idx, line in enumerate(g09_file):
                if (
                    line
                    == "                         Standard orientation:                         \n"
                ):
                    counter = 0
                    molecule_info = []
                    for xyz_line in g09_file[idx + 5 :]:
                        counter = counter + 1
                        if (
                            xyz_line
                            == " ---------------------------------------------------------------------\n"
                        ):
                            if num_of_atoms == None:
                                num_of_atoms = counter
                            break
                        else:
                            xyz_line = [i for i in xyz_line.split(" ") if i != ""]
                            atomic_symbol = self.atomic_number_to_atomic_symbol_dict[
                                xyz_line[1]
                            ]
                            atom_label = atomic_symbol + xyz_line[0]
                            xyz_coordinates = (
                                float(xyz_line[3]),
                                float(xyz_line[4]),
                                float(xyz_line[5]),
                            )
                            molecule_info.append(
                                [atomic_symbol, atom_label, xyz_coordinates]
                            )
                elif line.split(":  E")[0] == " SCF Done":
                    scf_energy = [i for i in line.split(" ") if i != ""]
                    scf_energy = scf_energy[4]
                if old_scf_energy != scf_energy:
                    old_scf_energy = scf_energy
                    # Build ccdc molecule object and then transform in into mol2 file
                    # Identifier will include the scf energy
                    og_molecule = self.mol2_molecules[0]
                    TS_molecule = og_molecule.copy()
                    TS_molecule.identifier = scf_energy
                    for info in molecule_info:
                        TS_molecule.atom(info[1]).coordinates = info[2]
                    TS_mol2_file = TS_mol2_file + TS_molecule.to_string("mol2")
            with open(
                self.save_to_location
                + "TS_output_"
                + self.g09_TS_output.split("-GaussianInput")[0]
                + ".mol2",
                "w",
            ) as f:
                f.write(TS_mol2_file)
                f.close()

    def ReconfigureCoordinationEnvironment(self, output_file_name, max_BD_dict):
        output_mol2_string = ""
        for molecule in self.mol2_molecules:
            for atom in molecule.atoms:
                if atom.atomic_symbol == self.metal_centre:
                    # remove all bonds as we need to consider all possible bonds to be reattached to the metal centre
                    molecule.remove_bonds(atom.bonds)
                    metal_coor = np.array(atom.coordinates)
                    for potential_atom in molecule.atoms:
                        potential_atomic_symbol = potential_atom.atomic_symbol
                        num_neighbours = len(potential_atom.neighbours)
                        bond_distance = np.linalg.norm(
                            metal_coor - np.array(potential_atom.coordinates)
                        )
                        try:
                            max_BD = max_BD_dict[potential_atomic_symbol]
                            if bond_distance <= max_BD:
                                if (
                                    potential_atomic_symbol == "S"
                                    and num_neighbours >= 4
                                ):
                                    pass
                                elif (
                                    potential_atomic_symbol == "P"
                                    and num_neighbours >= 4
                                ):
                                    pass
                                elif (
                                    potential_atomic_symbol == "N"
                                    and num_neighbours >= 4
                                ):
                                    pass
                                else:
                                    b_id = molecule.add_bond(
                                        Bond.BondType(1), atom, potential_atom
                                    )
                        except KeyError:
                            pass
            output_mol2_string = output_mol2_string + molecule.to_string("mol2")
        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            f.write(output_mol2_string)
            f.close()

    def SaveOnlyXCoordinate(self, x_coordinate, output_file_name):
        output_mol2_string = ""
        del_index = []
        for idx, molecule in enumerate(self.mol2_molecules):
            for atom in molecule.atoms:
                if atom.atomic_symbol == self.metal_centre:
                    if len(atom.neighbours) == x_coordinate:
                        output_mol2_string = output_mol2_string + molecule.to_string(
                            "mol2"
                        )
                        break
                    else:
                        del_index.append(idx)

        with open(self.save_to_location + output_file_name + ".mol2", "w") as f:
            f.write(output_mol2_string)
            f.close()
        del_index.reverse()
        for idx in del_index:
            del self.mol2_molecules[idx]

    def RotateVector(self, vector_to_rotate, rotation_axis, theta):
        x, y, z = rotation_axis[0], rotation_axis[1], rotation_axis[2]
        theta = -theta
        rotation_matrix = np.array(
            [
                [
                    np.cos(theta) + (x**2) * (1 - np.cos(theta)),
                    x * y * (1 - np.cos(theta)) - z * np.sin(theta),
                    x * z * (1 - np.cos(theta)) + y * np.sin(theta),
                ],
                [
                    y * x * (1 - np.cos(theta)) + z * np.sin(theta),
                    np.cos(theta) + (y**2) * (1 - np.cos(theta)),
                    y * z * (1 - np.cos(theta)) - x * np.sin(theta),
                ],
                [
                    z * x * (1 - np.cos(theta)) - y * np.sin(theta),
                    z * y * (1 - np.cos(theta)) + x * np.sin(theta),
                    np.cos(theta) + (z**2) * (1 - np.cos(theta)),
                ],
            ]
        ).reshape((3, 3))
        return rotation_matrix @ vector_to_rotate

    def resultant_vector(self, vectors_1, vectors_2):
        return np.array(
            [
                [np.linalg.norm(np.array(i) + np.array(j)) for j in vectors_2]
                for i in vectors_1
            ]
        )

    def find_max_resultant_vector(self, matrix):
        size = matrix.shape[0]
        tot_resultant_vector = 0
        for _ in range(0, size):
            max = np.amax(matrix)
            tot_resultant_vector = tot_resultant_vector + max
            max_index = np.argmax(matrix)
            indicies_of_max = np.unravel_index(max_index, matrix.shape)
            mat_size = matrix.shape[0]
            matrix = np.delete(matrix, indicies_of_max[0], axis=0)
            matrix = np.delete(matrix, indicies_of_max[1], axis=1)
        return abs(tot_resultant_vector - size * 2)

    def find_r(self, theta_phi, ideal_v, actual_v):
        rotate_v = [
            list(
                self.RotateVector(
                    vector_to_rotate=v,
                    rotation_axis=np.array([0, 0, 1]),
                    theta=np.deg2rad(theta_phi[0]),
                )
            )
            for v in actual_v
        ]
        rotate_v = [
            list(
                self.RotateVector(
                    vector_to_rotate=v,
                    rotation_axis=np.array([0, 1, 0]),
                    theta=np.deg2rad(theta_phi[1]),
                )
            )
            for v in rotate_v
        ]
        return self.find_max_resultant_vector(self.resultant_vector(ideal_v, rotate_v))

    def AnalysePointGroupDeviation(
        self, metal_centre, ideal_point_groups, output_file_name
    ):
        df = pd.DataFrame(
            columns=[pg[1] for pg in ideal_point_groups],
            index=[molecule.identifier for molecule in self.mol2_molecules],
        )
        for molecule in self.mol2_molecules:
            identifier = molecule.identifier
            for atom in molecule.atoms:
                if atom.atomic_symbol == metal_centre:
                    m_atom = np.array(atom.coordinates)
                    neighbour_atoms = atom.neighbours
                    neighbour_atoms = [np.array(i.coordinates) for i in neighbour_atoms]
                    # move neighbour atoms to where metal atom is at position 0,0,0
                    neighbour_atoms = [i - m_atom for i in neighbour_atoms]
                    # normalise the neighbour atom vectors so they have a magnitude of 1
                    neighbour_atoms = [i / np.linalg.norm(i) for i in neighbour_atoms]
                    break
            for ideal_pg in ideal_point_groups:
                pg_vectors = ideal_pg[0]
                pg_name = ideal_pg[1]
                results = basinhopping(
                    func=self.find_r,
                    x0=[0, 0],
                    minimizer_kwargs={
                        "method": "L-BFGS-B",
                        "args": (pg_vectors, neighbour_atoms),
                    },
                    niter=100,
                    stepsize=100,
                )
                df.loc[identifier, pg_name] = results.fun
        df.to_csv(self.save_to_location + output_file_name + ".csv")

    def AnalysePointGroupDeviationBetweenMethods(
        self, test_set, output_file_name, niter=50
    ):
        test_mol2_molecules = io.MoleculeReader(self.read_from_location + test_set)
        test_mol2_molecules = [i for i in test_mol2_molecules]
        test_mol2_dict = {}
        df = pd.DataFrame(
            columns=["r-value"],
            index=[molecule.identifier for molecule in self.mol2_molecules],
        )
        for molecule in test_mol2_molecules:
            test_mol2_dict[molecule.identifier] = molecule
        for molecule in self.mol2_molecules:
            try:
                identifier = molecule.identifier
                for atom in test_mol2_dict[identifier].atoms:
                    if atom.atomic_symbol == self.metal_centre:
                        m_atom = np.array(atom.coordinates)
                        test_atoms = atom.neighbours
                        test_atoms = [np.array(i.coordinates) for i in test_atoms]
                        test_atoms = [i - m_atom for i in test_atoms]
                        test_atoms = [i / np.linalg.norm(i) for i in test_atoms]
                        break
                for atom in molecule.atoms:
                    if atom.atomic_symbol == self.metal_centre:
                        m_atom = np.array(atom.coordinates)
                        ideal_atoms = atom.neighbours
                        ideal_atoms = [np.array(i.coordinates) for i in ideal_atoms]
                        # move neighbour atoms to where metal atom is at position 0,0,0
                        ideal_atoms = [i - m_atom for i in ideal_atoms]
                        # normalise the neighbour atom vectors so they have a magnitude of 1
                        ideal_atoms = [i / np.linalg.norm(i) for i in ideal_atoms]
                        break
                results = basinhopping(
                    func=self.find_r,
                    x0=[0, 0],
                    minimizer_kwargs={
                        "method": "L-BFGS-B",
                        "args": (ideal_atoms, test_atoms),
                    },
                    niter=niter,
                    stepsize=100,
                )
                df.loc[identifier, "r-value"] = results.fun
            except KeyError:
                pass
        df.to_csv(self.save_to_location + output_file_name + ".csv")

    def AnalyseRMSDBetweenMethods(self, test_set, output_file_name):
        test_mol2_molecules = io.MoleculeReader(self.read_from_location + test_set)
        test_mol2_molecules = [i for i in test_mol2_molecules]
        test_mol2_dict = {}
        df = pd.DataFrame(
            columns=["RMSD"],
            index=[molecule.identifier for molecule in self.mol2_molecules],
        )
        for molecule in test_mol2_molecules:
            test_mol2_dict[molecule.identifier] = molecule
        for mol1 in self.mol2_molecules:
            try:
                identifier = mol1.identifier
                mol2 = test_mol2_dict[identifier]
                try:
                    overlay = MolecularDescriptors.Overlay(mol1=mol1, mol2=mol2)
                    df.loc[identifier, "RMSD"] = overlay.rmsd
                except RuntimeError:
                    df.loc[identifier, "RMSD"] = np.nan
            except KeyError:
                pass
        df.to_csv(self.save_to_location + output_file_name + ".csv")


if __name__ == "__main__":
    pass

    complexes = AnalyseCompounds(
        read_from_location="C:/Users/cmsma/OneDrive - University of Leeds/Samuel Mace PhD Project/Please_delete_this_folder/",
        save_to_location="C:/Users/cmsma/OneDrive - University of Leeds/Samuel Mace PhD Project/Please_delete_this_folder/",
        pre_mol2_file="with_metal.mol2",
        orca_output_files_dir=",",
    )

    complexes.RemoveImpossibleComplexes(
        metal_centre="Mn", output_file_name="filtered_complexes"
    )

end = perf_counter()
print(str(round(end - start, 3)) + " seconds to execute")
