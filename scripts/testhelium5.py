'''
Created on 2 feb 2014

@author: David Lidberg
'''
from imports import *
import nhqm.solve
import nhqm.quantum_numbers
import nhqm.bases.momentum
from nhqm.problems import Helium5
from nhqm.helpers.quantum import absq
from nhqm.helpers.plotting import find_resonance_state

problem=Helium5()
basis_state_count = 20 # What does this do? Is this how we discretize k?
k_max = 2.5 # Determines how far out on the re(k) axix we go
l = 0
j = 0.5
x_peak=0.17  # Determines peak of the triangle
y_peak=-0.07
quantum_numbers = nhqm.quantum_numbers.QuantumNumbers(l, j)

'''
Triangular basis, so the contour contains resonance states. Is mom_tri a berggren-basis now?
'''
contour_tri = nhqm.bases.momentum.triangle_contour(x_peak, y_peak, k_max, basis_state_count, 5)
mom_tri = nhqm.bases.momentum.MomentumBasis(contour_tri) 



'''
Same values but with the normal contour
'''
contour_str = nhqm.bases.momentum.gauss_contour([0, k_max], basis_state_count)
mom_str = nhqm.bases.momentum.MomentumBasis(contour_str)




r=sp.linspace(0,50,500)

'''
Solve med de olika baserna. resonance tycks alltid vara samma som ground_state_tri[0]. Vet inte vad det betyder.
Hursomhelst, jag tror att triangelkonturen skapar en berggren-bas
'''
states_tri=nhqm.solve.solve(problem, quantum_numbers, mom_tri)
states_str=nhqm.solve.solve(problem, quantum_numbers, mom_str)


resonance_tri = find_resonance_state(states_tri) 
resonance_str = find_resonance_state(states_str)
ground_state_str=states_str[0]
ground_state_tri=states_tri[0]





print resonance_tri.energy
print resonance_str.energy
print ground_state_tri.energy
print ground_state_str.energy

plt.plot(r, absq(resonance_tri.wavefunction(r) * r), label=("triangelresonans"))
plt.plot(r, absq(resonance_str.wavefunction(r) * r), label=("vanlig resonans"))

plt.show()
plt.legend()

