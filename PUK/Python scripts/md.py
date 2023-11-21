import numpy as np
from ase import Atoms
from ase import neighborlist
from ase import units
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from gpaw import GPAW
from gpaw.occupations import FermiDirac


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


T = 800  # K

convergence = {
    "energy": 1e-7,
    "density": 1e-7,
}

mol = read("./moldyn.traj")
# mol = repartition_masses(mol, factor=4)

calc = GPAW(
    occupations=FermiDirac(0.05),
    xc="PBE",
    mode="lcao",
    basis="dzp",
    convergence=convergence,
    kpts=(1, 1, 1),
)
mol.set_calculator(calc)
mol.set_pbc(True)

# MaxwellBoltzmannDistribution(mol, temp=T * units.kB, force_temp=True)

# Use Berendsen taut=1 * units.picosecond
dyn = NVTBerendsen(
    mol, timestep=2.0 * units.fs, temperature=T * units.kB, taut=0.5 * 1000 * units.fs
)


def print_energy(a=mol):
    # Prints the potential, kinetic and total energy
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)

    print(
        "Energy per atom: "
        f"Epot = {epot:.4} "
        f"Ekin = {ekin:.4} "
        f"(T={ekin/(1.5 * units.kB)}) "
        f"Etot = {epot + ekin}"
    )


traj = Trajectory("restarted_moldyn.traj", "w", mol)
dyn.attach(traj.write, interval=2)

# Run the dynamics
print_energy()
dyn.run(4096 * 2)
