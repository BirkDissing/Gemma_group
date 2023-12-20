from ase.io import write
from ase.io import read
from gpaw import GPAW
from gpaw.occupations import FermiDirac
from ase.optimize import BFGS
from ase.constraints import FixAtoms


mol = read("./EtOH_unopt.traj")
# BASIS = {
#     "Au": "PBE.dzp",
# }
convergence = {
    "energy": 1e-8,
    "density": 1e-8,
}

calc = GPAW(
    xc='PBE',
    kpts=(1, 1, 1),
    mode='lcao',
    occupations=FermiDirac(0.0),
    # mixer=Mixer(0.025, 7, weight=100),
    convergence=convergence,
    basis="szp",
    txt='slab.txt',
)
mol.set_calculator(calc)
mol.set_pbc(True)

dyn = BFGS(mol, trajectory="EtOH_opt.traj", logfile="EtOH_opt.log")
dyn.run(fmax=0.01)
