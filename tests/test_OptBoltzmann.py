"""Tests for OptBoltzmann."""

import pathlib
import pickle
import unittest

import cobra
import numpy as np
import pandas as pd
import simplesbml
from bitarray import test

from OptBoltzmann.maximum_entropy import (
    flux_ent_opt,
    get_fixed_log_counts,
    get_initial_beta,
    get_nullspace,
    get_objective_reaction_identifiers,
    get_params,
    get_random_initial_fluxes,
    get_standard_change_in_gibbs_free_energy,
    get_stoichiometric_matrix,
    get_target_log_variable_counts,
)


class TestMaximumEntropy(unittest.TestCase):
    """Test that Maximum entropy methods work correctly."""

    def setUp(self):
        """Load test models"""
        self.sbmlfiles = ["../models/sbml/split_pathway.xml"]
        self.picklefiles = ["../models/01_Glycolysis_Example/gluconeogenesis_prot4.pkl",
                           ]
        self.cobramodels = [cobra.io.read_sbml_model(sbmlfile) for sbmlfile in self.sbmlfiles]
        self.sbmlmodels = [simplesbml.loadSBMLFile(sbmlfile) for sbmlfile in self.sbmlfiles]
        self.picklemodels = []
        for picklefile in self.picklefiles:
            with open(picklefile, "rb") as handle:
                self.picklemodels.append(pickle.load(handle))
        self.physical_params = dict(
            T=298.15, R=8.314e-03, N_avogadro=6.022140857e23, VolCell=1.0e-15
        )

    def test_flux_ent_opt(self):
        """Test flux entropy optimization."""
        T, N_avogadro, VolCell, R = (
            self.physical_params["T"],
            self.physical_params["N_avogadro"],
            self.physical_params["VolCell"],
            self.physical_params["R"],
        )
        for model in self.picklemodels:

            Stoich_matrix = pd.DataFrame(model["S"])
            active_reactions = pd.DataFrame(model["active_reactions"])
            metabolites = pd.DataFrame(model["metabolites"])
            Keq = pd.DataFrame({"Keq": model["Keq"]}, index=Stoich_matrix.index)
            Concentration2Count = N_avogadro * VolCell
            concentration_increment = 1 / (N_avogadro * VolCell)

            S = Stoich_matrix.values

            # number of variable metabolites
            nvar = len(metabolites[metabolites["Variable"] == True].values)

            fixed_concs = np.array(metabolites["Conc"].iloc[nvar:].values, dtype=np.float64)
            fixed_counts = fixed_concs * Concentration2Count
            f_log_counts = np.log(fixed_counts)
            """
            The Variable Metabolite Upper bounds
            """

            vcount_upper_bound = np.log(np.ones(nvar) * 1.0e-03 * Concentration2Count)

            Keq = model["Keq"]

            # Initial choice here as a single reaction we want max flux through
            react_for_maxflux = 2

            n_REACT = S.shape[0]
            obj_coefs = np.zeros(n_REACT)
            obj_coefs[react_for_maxflux] = 1

            y_sol, alpha_sol, n_sol = flux_ent_opt(
                obj_coefs, vcount_upper_bound, f_log_counts, S, Keq
            )

            y_expected = np.array(
                [
                    0.64710692,
                    0.64710692,
                    0.64710692,
                    0.64710692,
                    0.64710692,
                    1.29421384,
                    1.29421384,
                    1.29421384,
                    1.29421384,
                    1.29421384,
                    1.29421384,
                ]
            )
            alpha_expected = np.array(
                [
                    1.00000305,
                    1.00000296,
                    1.00000306,
                    1.00000321,
                    1.000003,
                    1.00000179,
                    1.00000161,
                    1.00000178,
                    1.00000172,
                    1.00000173,
                    0.00109527,
                ]
            )
            n_expected = np.array(
                [
                    -1.33042504,
                    13.30836837,
                    -2.02971018,
                    1.48078474,
                    -2.47739902,
                    -4.37722669,
                    -2.83495049,
                    3.57730507,
                    1.3218781,
                    2.39885926,
                ]
            )

            np.testing.assert_almost_equal(y_sol, y_expected)
            np.testing.assert_almost_equal(alpha_sol, alpha_expected)
            np.testing.assert_almost_equal(n_sol, n_expected)

    def test_maximum_entropy_pyomo_relaxed(self):
        """Test against analytic solution."""

    def test_get_nullspace(self):
        """Make sure the nullspace is computed correctly."""

    def test_get_stoichiometric_matrix(self):
        """Make sure the stoichiometric matrix is extracted from the sbml file correctly."""
        for cobramodel, sbmlmodel in zip(self.cobramodels, self.sbmlmodels):
            expected = cobra.util.array.create_stoichiometric_matrix(
                cobramodel, array_type="DataFrame"
            )
            actual = get_stoichiometric_matrix(sbmlmodel)
            self.assertTrue(expected.equals(actual))

    def test_get_random_initial_variable_concentrations(self):
        """Make sure the initial variable concentrations pass muster."""

    def test_get_random_initial_fluxes(self):
        """Make sure the fluxes are initialized appropriately."""

    def test_get_standard_change_in_gibbs_free_energy(self):
        """Ensure the gibbs free energy is extracted from the sbml file correctly."""

    def test_get_initial_beta(self):
        """Ensure the initial beta is consistent with the initial fluxes."""

    def test_get_target_log_variable_counts(self):
        """Ensure the log variable counts are extracted from the sbml file correctly."""

    def test_get_fixed_log_counts(self):
        """Ensure the fixed log counts are extracted from the sbml file correctly."""

    def test_get_objective_reaction_identifiers(self):
        """Ensure the flux objective coefficients are extracted from the sbml file correctly."""

    def test_get_params(self):
        """Ensure all parameters are accepted by maximum entropy pyomo."""
