import numpy as np
from ase import Atoms
from ase import neighborlist
from ase import units
from ase.io.trajectory import Trajectory

traj = Trajectory("D:\KU\Masters\Gemma_group\PUK\Python_scripts\EtOH_moldyn_NVT2.traj")
oxygen_x_force = []
for i, atoms in enumerate(traj[:50]):
    oxygen_x_force.append(atoms.get_forces()[2,0])

print(oxygen_x_force)



