import numpy as np
import pandas as pd
from ase import Atoms
from ase import neighborlist
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.extxyz import read_extxyz, write_extxyz
import statsmodels.api as sm
# from gpaw import GPAW
# from gpaw.occupations import FermiDirac
# from ase.md import MDLogger
# from gpaw import setup_paths
#setup_paths.insert(0, "/kemi/williamb/opt/gpaw-datasets/gpaw-setups-0.9.20000")


H_MASS = 1.008


def repartition_masses(struct: Atoms, factor: int) -> Atoms:
    cut_off = neighborlist.natural_cutoffs(struct)
    neighborList = neighborlist.NeighborList(
        cut_off, self_interaction=False, bothways=True
    )
    neighborList.update(struct)
    conn = neighborList.get_connectivity_matrix().todense()

    symbols = mol.symbols
    masses = mol.get_masses()
    prior_mass = masses.sum()

    masses[symbols == "H"] *= factor

    for line in conn[symbols == "H"]:
        line = np.array(line).flatten()

        masses[line == 1] -= H_MASS * (factor - 1)

    post_mass = masses.sum()
    assert np.abs(prior_mass - post_mass) < 1e-5
    mol.set_masses(masses)

    return mol


def create_VAR_OLS_model(main, secondary, order=2):
    input = len(main)-order
    num_secondaries = secondary.shape[0]
    X = np.zeros([input,order*(num_secondaries+1)+1])
    for i in range(order):
        X[:,i*(num_secondaries+1)] = main[i:input+i]
        for j in range(num_secondaries):
            X[:,i*(num_secondaries+1)+j+1] = secondary[j,i:input+i]
    X[:,-1] = np.ones(len(main)-order)
    return sm.OLS(main[order:], X)

mol = read(r"C:\Users\birk\OneDrive - University of Copenhagen\Documents\KU tid\Gemma_group\PUK\Python_scripts\Data\EtOH_moldyn_xyz_test.xyz")
print(mol)
print(mol[0])
