import unittest
from SBS_functions import *

# unit testing a selection of functions from SBS_functions.py

class TestGeneratingHamsFunctions(unittest.TestCase):
    
    def test_perm(self):
        """Test perm() with valid and invalid inputs"""
        good_inputs = [1,2,3] # inputs must be positive integers
        for input in good_inputs:
            permutation = perm(input)
            self.assertIsInstance(permutation, list) # check output is a list
            self.assertEqual(len(permutation), input) # check length of list == no. of envs
        bad_inputs = [0, 2.5, -1]
        for input in bad_inputs:
            with self.assertRaises(ValueError):  # check invalid inputs raise ValueError
                perm(input)
                
    def test_tensor_prod_env_hams_GUE(self):
        """Test tensor_prod_env_hams_GUE(number_of_environments, env_dim, var=1) with valid and invalid inputs"""
        # good input test:
        ham = tensor_prod_env_hams_GUE(2,2,2)
        self.assertIsInstance(ham, Qobj) # check output is a Qobj object (Qutip)
        self.assertEqual(ham.shape, (2*2,2*2)) # check output has correct dimension
        self.assertEqual(ham.isherm, True) # check output is Hermitian
        
        # bad input test
        with self.assertRaises(ValueError):
            tensor_prod_env_hams_GUE(2.5,2,2) # number_of_environments is not an integer
        with self.assertRaises(ValueError):
            tensor_prod_env_hams_GUE(2,1,2) # env_dim is not an integer >= 2
        with self.assertRaises(ValueError):
            tensor_prod_env_hams_GUE(2,5.5,2) # env_dim is not an integer

            
class TestGeneratingDensityMatricesFunctions(unittest.TestCase):  
    
    def test_initial_state(self):
        """Test initial_state(system_dim, number_of_environments, env_dim, env_initial_state=0, sys_probs=0) with valid and invalid inputs"""
        # good input test
        state = initial_state(2, 3, 4)
        self.assertIsInstance(state, Qobj) # check output is a Qobj object (Qutip)
        self.assertEqual(state.shape, (2*4**3,2*4**3)) # check output has correct dimension
        self.assertAlmostEqual(state.tr(), 1) # check trace(rho)=1
        # bad input tests
        with self.assertRaises(ValueError):
            initial_state(2.5,2,2) # system_dim is not an integer
        with self.assertRaises(ValueError):
            initial_state(2,1.2,2) # number_of_environments is not an integer 
        with self.assertRaises(ValueError):
            initial_state(2,5,1) # env_dim is not an integer >= 2
        
        