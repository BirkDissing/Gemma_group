import numpy as np
import pandas as pd
from ase import Atoms
from ase import neighborlist
from ase import units
from ase.io import read, write
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.parallel import world
from gpaw import GPAW
from gpaw.occupations import FermiDirac
from statsmodels.tsa.arima.model import ARIMA
from ase.md import MDLogger
import statsmodels.api as sm
from gpaw import setup_paths
setup_paths.insert(0, "/kemi/williamb/opt/gpaw-datasets/gpaw-setups-0.9.20000")

input = 6
pred_step = 6
n_time_steps = 300
dt = 0.5*units.fs
T = 400  # K
file = "model_test.xyz"
md = True


#The function takes arrays of forces and returns a VAR OLS model
#_____Inputs______
# main - 1D np array with the time series which the model fits to.
# secondary - 2D np array with the secondary time series used to predict the main time series
# order - Positive integer which decides the numper of points in the past is used to predict the next point
#
#_____Output______
# Returns a OLS model that uses the main and secondary time series to predict the main time series. In order to predict t+2 of the main timeseries,
# t+1 of both main and secondary time series must be known. If the values of the secondary time series is not known this can be done by also creating models
# for the secondary time series and use those to predict the value at t+1
def create_arima_model(main, order=2):
    return ARIMA(main, order=(order, 0, 0))

# Returns 3 fitted models from create_VAR_OLS_model to create 3 models to respectively predict the forces in x, y, and z direction for an atom in a molecule.
# Uses L2 regularized fitting.
#_____Inputs______
# forces - 2D np array with the three force time series for the atom chosen. Has the shape [N, 3] where N is the number of input points in the model.
# order - Positive integer which decides the numper of points in the past is used to predict the next point
#
#_____Output______
# Returns 3 fitted VAR OLS which can be used to predict future points in the time series. t+1 must be predicted for all three time series before t+2 can be predicted.
def create_atom_arima_model(forces, order=2):
    model_x = create_arima_model(forces[0,:], order=order).fit_regularized(method="elastic_net", L1_wt=0)
    model_y = create_arima_model(forces[1,:], order=order).fit_regularized(method="elastic_net", L1_wt=0)
    model_z = create_arima_model(forces[2,:], order=order).fit_regularized(method="elastic_net", L1_wt=0)
    with model_x.fix_params({'const': 0}):
        model_x = model_x.fit()
    with model_y.fix_params({'const': 0}):
        model_y = model_x.fit()
    with model_z.fix_params({'const': 0}):
        model_z = model_x.fit()
    return model_x, model_y, model_z

# Function which predicts the forces on the atoms in a molecule for a certain number of time steps
#_____Inputs______
# file - String with the name of the xyz file containing information on the molecule
# input - Positive integer with the number of data points used as input in training the OLS VAR models
# pred_step - Positive integer which decides the number of time steps predicted by the OLS VAR models
# order - Positive integer which decides the numper of points in the past is used to predict the next point
#
#_____Output______
# Returns a 3D np array with the predicted forces. The array as the following shape [N, 3, p], with N being the number of atoms in the molecule
# and p being the number of time steps predicted
def predict_forces(file=file, input=input, pred_step=pred_step, order=2):
    #mol = read(file, index=slice(-input, None))
    mol = read(file, index=':8')
    n_atoms = mol[0].get_global_number_of_atoms()
    predicted_forces = np.zeros((n_atoms, 3, pred_step))
    forces = np.zeros((n_atoms, 3, input))

    for i in range(len(mol)):
        forces[:,:,i] = mol[i].get_forces()
    for i in range(n_atoms):
        model_x, model_y, model_z = create_atom_arima_model(forces[i,:,:], order=order)
        forecast_forces = np.zeros((3, pred_step))
        forecast_forces[0,:] = model_x.predict(start=input, end=input+pred_step)
        forecast_forces[1,:] = model_y.predict(start=input, end=input+pred_step)
        forecast_forces[2,:] = model_z.predict(start=input, end=input+pred_step)
        predicted_forces[i,:,:] = forecast_forces
        
    return predicted_forces

convergence = {
    "energy": 1e-7,
    "density": 1e-7
}

mol = read("./EtOH_opt.traj")
# mol = repartition_masses(mol, factor=4)

calc = GPAW(
    occupations=FermiDirac(0.05),
    xc="PBE",
    mode="lcao",
    basis="PBE.sz",
    convergence=convergence,
    kpts=(1, 1, 1),
    txt="model_test.txt"
)
mol.set_calculator(calc)
mol.center(vacuum=10)
mol.set_pbc(True)

# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(mol, temperature_K=T)
 
# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn = VelocityVerlet(mol, dt)  
# 0.5 fs time step.




df = pd.DataFrame(columns=['md','C1(x)','C1(y)','C1(z)','C2(x)','C2(y)','C2(z)','O(x)','O(y)','O(z)','H1(x)','H1(y)','H1(z)','H2(x)','H2(y)','H2(z)','H3(x)','H3(y)','H3(z)','H4(x)','H4(y)','H4(z)','H5(x)','H5(y)','H5(z)','H6(x)','H6(y)','H6(z)'])
def print_forces(a=mol, md=True):
    # Prints the forces
    force = a.get_forces()
    row = []
    row.append(md)
    for i in range(len(a)):
        for j in range(3):
            row.append(force[i,j])
    df.loc[len(df)] = row
    df.to_csv("Moldyn_dataframe_for.csv")

traj = TrajectoryWriter("Model_test_traj.traj", "w", mol)


def write_to_xyz(a=mol):
    if world.rank == 0:
        write("EtOH_moldyn_for.xyz", a, append = True)

# for i in range(n_time_steps):
#     if i==0 or i%pred_step!=0:
#         md = True
#         dyn.run(1)
#     else:
#         md = False
#         predicted_forces = predict_forces()
#         for i in range(pred_step):
#             #Get masses for the atoms in the molecule
#             masses = mol.get_masses()[:, np.newaxis]

#             #Get the forces, momenta, and positions for the current step
#             #forces = mol[i].get_forces()
#             forces = predicted_forces[:,:,i] 
#             p = mol.get_momenta()
#             r = mol.get_positions()
            
#             #Calculate new momenta and positions
#             p += 0.5 * dt * forces
#             mol.set_positions(r + dt * p / masses)
            

#             #Was in ase.step. Unsure if needed
#             if mol.constraints:
#                 p = (mol.get_positions() - r) * masses / dt

#             #Momenta needs to be stored before possible calculations of forces
#             mol.set_momenta(p, apply_constraint=False)

#             #Forces for next step is found either using predicted forces or gpaw calculator
#             if i<pred_step-1:
#                 forces = predicted_forces[:,:,i+1]
#                 #forces = mol[i+1].get_forces()
#             else:
#                 #print("Get forces from calculator")
#                 forces = mol.get_forces(md=True)
            

#             #Calculate and set momenta for the next step
#             mol.set_momenta(mol.get_momenta() + 0.5 * dt * forces)

#     print_forces(md=md)
#     if world.rank ==0:
#         write(file, mol, append = True)
#     traj.write()
        
for i in range(6):
    dyn.run(1)
    print_forces()
    
    write(file, mol, append = True)
    traj.write()
file2 = "moldyn_test.xyz"
if world.rank == 0:
        write(file2, mol)

mol2 = read(file2, index='-1')
mol2.set_calculator(calc)
mol2.center(vacuum=10)
mol2.set_pbc(True)
# Set the momenta corresponding to T=300K
MaxwellBoltzmannDistribution(mol2, temperature_K=T)
 
# We want to run MD with constant energy using the VelocityVerlet algorithm.
dyn2 = VelocityVerlet(mol2, dt)  
traj2 = TrajectoryWriter("model_test_traj2.traj", "w", mol2)
predicted_forces = predict_forces()
for i in range(pred_step):
    #Get masses for the atoms in the molecule
    masses = mol.get_masses()[:, np.newaxis]

    #Get the forces, momenta, and positions for the current step
    #forces = mol[i].get_forces()
    forces = predicted_forces[:,:,i] 
    p = mol.get_momenta()
    r = mol.get_positions()
    
    #Calculate new momenta and positions
    p += 0.5 * dt * forces
    mol.set_positions(r + dt * p / masses)
    

    #Was in ase.step. Unsure if needed
    if mol.constraints:
        p = (mol.get_positions() - r) * masses / dt

    #Momenta needs to be stored before possible calculations of forces
    mol.set_momenta(p, apply_constraint=False)

    #Forces for next step is found either using predicted forces or gpaw calculator
    if i<pred_step-1:
        forces = predicted_forces[:,:,i+1]
        #forces = mol[i+1].get_forces()
    else:
        #print("Get forces from calculator")
        forces = mol.get_forces(md=True)
    

    #Calculate and set momenta for the next step
    mol.set_momenta(mol.get_momenta() + 0.5 * dt * forces)
    print_forces(md=False)
    if world.rank == 0:
        write(file, mol, append = True)


for i in range(6):
    dyn.run(1)
    print_forces()
    if world.rank == 0:
        write(file, mol, append = True)
    traj.write()

for i in range(6+pred_step):
    dyn2.run(1)
    if world.rank == 0:
        write(file2, mol2, append = True)
    traj2.write()
    

