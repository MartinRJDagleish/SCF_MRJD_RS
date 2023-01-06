
My Ipython journey: 
```python
h2o = psi4.geometry("""
0 1
O   0.000000000000  -0.143225816552   0.000000000000
H   1.638036840407   1.136548822547  -0.000000000001
H  -1.638036840407   1.136548822547  -0.000000000001
units bohr
""")
  15: psi4.energy("scf", molecule=h2o)
  16: eng, wfn = psi4.energy("scf", molecule=h2o, return_wfn=True)
  17: E_nuc = h2o.nuclear_repulsion_energy()
  18: E_nuc
  19: wfn.jk()
  20: wfn.jk().J()
  21: wfn.jk().J()[0]
  22: wfn.jk().J()[0].to_array()
  23: wfn.jk().J()[0].to_array()[0]
  24: wfn.jk().J()[0].to_array()[0][0]
  25: J = np.array(wfn.jk().J()[0].to_array())
  26: J
  27: J = wfn.jk().J()[0].to_array()[0]
  28: J
  29: print(J)
  30: J = wfn.jk().J()[0].to_array()
  31: J
  32: J[0][0][0][5]
  33: J[6][6][6][6]
  34: JK = psi4.core.JK.build(bas)
  35: JK = psi4.core.JK.build("sto-3g")
  36:
# Create instance of MintsHelper using primary basis set
mints = psi4.core.MintsHelper(primary_basis)
# Compute one-electron AO overlap matrix
S = mints.ao_overlap()
# Compute core Hamiltonian matrix
T = mints.ao_kinetic()
V = mints.ao_potential()
H = T + V
# Compute two-electron integrals in AO basis in memory
I_ao = mints.ao_eri()
  37:
# Create instance of MintsHelper using primary basis set
mints = psi4.core.MintsHelper("sto-3g")
# Compute one-electron AO overlap matrix
S = mints.ao_overlap()
# Compute core Hamiltonian matrix
T = mints.ao_kinetic()
V = mints.ao_potential()
H = T + V
# Compute two-electron integrals in AO basis in memory
I_ao = mints.ao_eri()
  38:
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96, 104.5
""")
hf_e, hf_wfn = psi4.energy("HF/cc-pVDZ", molecule=mol, return_wfn=True)
  39:
mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
""")
hf_e, hf_wfn = psi4.energy("HF/cc-pVDZ", molecule=mol, return_wfn=True)
  40: docc = hf_wfn.ndocc()
  41: hf_wfn.alpha_orbital_space()
  42: hf_wfn.nalpha()
  43: hf_wfn.nbeta()
  44: mints = psi4.core.MintsHelper("sto-3g")
  45: mints = psi4.core.MintsHelper?
  46: psi4.core.MintsHelper?
  47: psi4.core.MintsHelper("sto-3g")
  48: psi4.core.MintsHelper(hf_wfn.basisset())
  49: mints = psi4.core.MintsHelper(hf_wfn.basisset())
  50:
# Compute one-electron AO overlap matrix
S = mints.ao_overlap()
# Compute core Hamiltonian matrix
T = mints.ao_kinetic()
V = mints.ao_potential()
H = T + V
# Compute two-electron integrals in AO basis in memory
I_ao = mints.ao_eri()
  51: T
```

This were my input which mostly worked. 

-> Link: 
https://github.com/psi4/psi4numpy

http://vergil.chemistry.gatech.edu/notes/permsymm/permsymm.pdf

