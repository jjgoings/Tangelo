import unittest
import numpy as np

from tangelo.toolboxes.qubit_mappings.mapping_transform import fermion_to_qubit_mapping
from tangelo.molecule_library import mol_H4_sto3g
from tangelo.linq import Circuit, Gate, Simulator
from tangelo.toolboxes.operators.trim_trivial_qubits import trim_trivial_qubits


class VQESolverTest(unittest.TestCase):

    def test_trim_trivial_qubits(self):
        sim = Simulator()

        # Define operator
        mol = mol_H4_sto3g
        qb_ham = fermion_to_qubit_mapping(
            mol.fermionic_hamiltonian, "JW", mol.n_active_sos, up_then_down=True)

        # Create circuit
        qubits = [1, 3, 5, 7]
        initial_gates = [Gate("X", 0), Gate("X", 1), Gate("X", 4), Gate("X", 5)]
        gen_gates = [Gate("RX", 1, parameter=np.pi/2), Gate("H", 3), Gate("H", 5), Gate("H", 7)]
        gen_gates += [Gate("CNOT",  qubits[i+1], qubits[i]) for i in range(len(qubits)-1)]
        param_gate = [Gate("RZ", qubits[-1], parameter=0.22999414822164746)]
        gen_inv = [gate.inverse() for gate in gen_gates[::-1]]
        circuit = Circuit(initial_gates + gen_gates + param_gate + gen_inv)

        # Trim circuit and operator
        trimmed_circ, trimmed_op = trim_trivial_qubits(qb_ham, circuit)
        # Test for expected energy
        energy = sim.get_expectation_value(trimmed_op, trimmed_circ)

        self.assertAlmostEqual(energy, -1.803987566489405, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
