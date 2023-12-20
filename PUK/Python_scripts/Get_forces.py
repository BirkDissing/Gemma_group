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
from gpaw import setup_paths
setup_paths.insert(0, "/kemi/williamb/opt/gpaw-datasets/gpaw-setups-0.9.20000")



traj = Trajectory("./EtOH_moldyn_NVT2.traj")

convergence = {
    "energy": 1e-7,
    "density": 1e-7,
}

calc = GPAW(
    occupations=FermiDirac(0.05),
    xc="PBE",
    mode="lcao",
    basis="dzp",
    convergence=convergence,
    kpts=(1, 1, 1),
)



force_array = np.zeros([len(traj[:50]), len(traj[0]), 3])
with open("forces_test.txt", "w") as f:
    f.write(str("Dimensions of force array:"+force_array.shape))

for i, atoms in enumerate(traj[:50]):
    atoms.set_calculator(calc)
    forces = atoms.get_forces()

    with open("forces_test.txt", "a") as f:
        f.write(str("Forces in atom object",i+1,":\n", forces))

    force_array[i,:,:] = forces

np.save("./EtOH_moldyn_NVT_forces.npy"+force_array)
