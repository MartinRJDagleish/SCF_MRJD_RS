import psi4
import numpy as np 

np.set_printoptions(precision=5, linewidth=100, suppress=True)
psi4.set_memory("4 GB")
psi4.core.set_output_file('output.dat', False)
numpy_memory = 4

mol = psi4.geometry("""
0 1
H 0.0 0.0 0.0
H 0.0 0.0 1.4
units bohr
""")

psi4.set_options({"save_jk": "true", "guess": "CORE", "basis": "sto-3g", "DIIS": "false", "SCF_INITIAL_ACCELERATOR": "None", "CFOUR_SCF_MAXCYC": "500", "e_convergence": 1e-8})

wfn = psi4.core.Wavefunction.build(mol, psi4.core.get_global_option("BASIS"))
mints = psi4.core.MintsHelper(wfn.basisset())

S = np.asarray(mints.ao_overlap())
T = np.asarray(mints.ao_kinetic())
V = np.asarray(mints.ao_potential())
ERI = np.asarray(mints.ao_eri())

no_basis_funcs: int = S.shape[0]

