import numpy as np
import pandas as pd
from ase import Atoms
from ase import neighborlist
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from gpaw import GPAW
from gpaw.occupations import FermiDirac
from gpaw import setup_paths
setup_paths.insert(0, "/kemi/williamb/opt/gpaw-datasets/gpaw-setups-0.9.20000")


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


T = 400  # K

convergence = {
    "energy": 1e-7,
    "density": 1e-7,
}

mol = read("./EtOH_opt.traj")
# mol = repartition_masses(mol, factor=4)

calc = GPAW(
    occupations=FermiDirac(0.05),
    xc="PBE",
    mode="lcao",
    basis="sz",
    convergence=convergence,
    kpts=(1, 1, 1),
    txt='moldyn.txt',
)
mol.set_calculator(calc)
mol.center(vacuum=10)
mol.set_pbc(True)

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(mol, temperature_K=T)
 
# We want to run MD with constant energy using the VelocityVerlet algorithm.
Forces_csv = "./forces_data.csv"
dyn = VelocityVerlet(mol, 0.5 * units.fs)  
# 0.5 fs time step.
with open(Forces_csv, "w") as file:
    file.write("C1, C2, O, H1, H2, H3, H4, H5, H6")
    file.write("\n")

def print_forces(a=mol):
    # Prints the forces
    force = a.get_forces()
    with open(Forces_csv, "a") as file:
        for i in range(force.shape[0]):
            file.write(str(force[i,:])+",")
        file.write("\n")


    

traj = Trajectory("EtOH_moldyn_NVT.traj", "w", mol)
dyn.attach(traj.write, interval=1)
dyn.attach(print_forces, interval=1)

# Run the dynamics
dyn.run(300)

