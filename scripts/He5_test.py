from __future__ import division
from imports import * #noqa
from nhqm.solve import solve
from nhqm.quantum_numbers import QuantumNumbers
from nhqm.bases.momentum import MomentumBasis, gauss_contour, triangle_contour
from nhqm.problems import Helium5
from nhqm.helpers.quantum import absq
# from nhqm.helpers.plotting import find_resonance_state

# Solves the He5 problem with the momentum basis,
# then plots the ground state wavefunction

# Parameters
l = 1
j = 1.5
x_peak = 0.17
y_peak = -0.07
basis_state_count = 30
k_max = 30

problem = Helium5()
quantum_numbers = QuantumNumbers(l, j)

# Bases and contours
mom_contour = gauss_contour([0, k_max], basis_state_count)
berg_contour = triangle_contour(x_peak, y_peak, k_max, basis_state_count, 5)
mom = MomentumBasis(mom_contour)
berg = MomentumBasis(berg_contour)
bas = [mom, berg]

#  Solve, print and plot for each basis
r = sp.linspace(0, 30, 100)
for basis in bas:
    states = solve(problem, quantum_numbers, basis)
    ground_state = states[0]
    print basis.name
    print "Ground state energy", ground_state.energy, problem.energy_units
    plt.plot(r, absq(ground_state.wavefunction(r) * r), label=basis.name)
# plot(ground_state.wavefunction) #pseudo-code
plt.show()
