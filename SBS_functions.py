import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.linalg import sqrtm
from qutip import *
from itertools import permutations
import h5py
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import unitary_group
from scipy.linalg import block_diag
import scipy
import pickle
from random import random

###########################################################################
# Saving data
###########################################################################

def pickle_data(filename, data):
    """
    filename is a string
    data can be any object
    """
    # open a file, where you want to store the data
    file = open(filename, 'wb')
    # dump information to that file
    pickle.dump(data, file)
    # close the file
    file.close()

###########################################################################
# Functions to generate random hamiltonians
###########################################################################

def perm(number_of_environments):
    if not isinstance(number_of_environments, int) or number_of_environments <= 0:
        raise ValueError("number_of_environments must be a positive integer")
    x = [0]*number_of_environments
    x[0] = 1
    permu = set(permutations(x, number_of_environments))
    return list(permu)

def ham(dim):
    ham = (2 * np.random.rand(dim, dim) - 1) + 1j * (2 * np.random.rand(dim, dim) - 1)
    return (ham + ham.T.conj())
    
def tensor_prod_env_hams(number_of_environments, env_dim):
    '''
    only works for env that are all the same dimension
    gives a block of environment hamiltonians
    '''
    p = perm(number_of_environments)
    for i in range(len(p)):
        p[i] = list(p[i])
        for j in range(number_of_environments):
            if p[i][j]==1:
                hammy = Qobj(ham(env_dim))
                p[i][j] = hammy
                #plot_complex_Qutip(hammy)
            else:
                p[i][j] = Qobj(np.eye(env_dim))
    total_tensor_list = [0]*number_of_environments
    for i in range(number_of_environments):
        total_tensor_list[i] = tensor(p[i])
    total_tensor = sum(total_tensor_list)
    return total_tensor

def tot_ham_no_perm(system_dim, number_of_environments, env_dim):
    blocks = [0]*system_dim
    sys = [0]*system_dim
    tensors = [0]*system_dim
    for i in range(system_dim):
        blocks[i] = tensor([Qobj(ham(env_dim))]*number_of_environments)
        sys[i] = basis(system_dim, i)*basis(system_dim, i).dag()
        tensors[i] = tensor(sys[i], blocks[i])
    return sum(tensors) 

def total_hamiltonian(system_dim, number_of_environments, env_dim):
    blocks = [0]*system_dim
    sys = [0]*system_dim
    tensors = [0]*system_dim
    for i in range(system_dim):
        blocks[i] = tensor_prod_env_hams(number_of_environments, env_dim)
        sys[i] = basis(system_dim, i)*basis(system_dim, i).dag()
        tensors[i] = tensor(sys[i], blocks[i])
    return sum(tensors) 


##### GUE  Hamiltonian #####################################################

def total_hamiltonian_GUE(system_dim, number_of_environments, env_dim, var=1):
    blocks = [0]*system_dim
    sys = [0]*system_dim
    tensors = [0]*system_dim
    for i in range(system_dim):
        blocks[i] = tensor_prod_env_hams_GUE(number_of_environments, env_dim, var)
        sys[i] = basis(system_dim, i)*basis(system_dim, i).dag()
        tensors[i] = tensor(sys[i], blocks[i])
    return sum(tensors) 

def tensor_prod_env_hams_GUE(number_of_environments, env_dim, var=1):
    '''
    only works for env that are all the same dimension
    gives a block of environment hamiltonians
    returns a Qobj
    '''
    if not isinstance(env_dim, int) or env_dim <= 1:
        raise ValueError("environment dimension must be a positive integer >= 2")
    p = perm(number_of_environments)
    for i in range(len(p)):
        p[i] = list(p[i])
        for j in range(number_of_environments):
            if p[i][j]==1:
                hammy = Qobj(GUE(env_dim, var))
                p[i][j] = hammy
                #plot_complex_Qutip(hammy)
            else:
                p[i][j] = Qobj(np.eye(env_dim))
    total_tensor_list = [0]*number_of_environments
    for i in range(number_of_environments):
        total_tensor_list[i] = tensor(p[i])
    total_tensor = sum(total_tensor_list)
    return total_tensor

# tenpy - GUE

def standard_normal_complex(size):
    """return ``(R + 1.j*I)`` for independent `R` and `I` from np.random.standard_normal."""
    return np.random.standard_normal(size) + 1.j * np.random.standard_normal(size)

def GUE(N, variance=1):
    r"""Gaussian unitary ensemble (GUE).
    Parameters
    ----------
    size : tuple
        ``(n, n)``, where `n` is the dimension of the output matrix.
    Returns
    -------
    H : ndarray
        Hermitian (complex) numpy matrix drawn from the GUE, i.e.
        :math:`p(H) = 1/Z exp(-n/4 tr(H^2))`.
    """
    size = (N,N)
    A = variance * standard_normal_complex(size)
    return (A + A.T.conj()) * 0.5

###########################################################################
# generating CNOT Hamiltonians
###########################################################################

def total_cnot_hamiltonian(number_of_environments, coupling_coefficients):
    """
    see max and manus example hamiltonian for structure
    system is a qubit
    envs are all qubits
    we can have any number of environments
    coupling_coefficients is a list number_of_environments long 
    it gives the g_k's in the Hamiltonian
    """
    system_dim = 2
    env_dim = 2
    blocks = [0]*system_dim
    sys = [0]*system_dim
    tensors = [0]*system_dim
    blocks[0] = Qobj(np.eye(env_dim**number_of_environments), dims=[[env_dim]*number_of_environments,[env_dim]*number_of_environments])
    blocks[1] = tensor_cnot_hams(number_of_environments, coupling_coefficients)
    for i in range(system_dim):
        sys[i] = basis(system_dim, i)*basis(system_dim, i).dag()
        tensors[i] = tensor(sys[i], blocks[i])
    return sum(tensors) 

def tensor_cnot_hams(number_of_environments, coupling_coefficients):
    '''
    only works for env_dim =2 
    gives a block of environment cnot hamiltonians
    '''
    env_dim = 2
    sigma_x = np.array([[0,1],[1,0]])
    p = perm(number_of_environments)
    for i in range(len(p)):
        p[i] = list(p[i])
        for j in range(number_of_environments):
            if p[i][j]==1:
                p[i][j] = Qobj(coupling_coefficients[j]*sigma_x)
            else:
                p[i][j] = Qobj(np.eye(env_dim))
    total_tensor_list = [0]*number_of_environments
    for i in range(number_of_environments):
        total_tensor_list[i] = tensor(p[i])
    total_tensor = sum(total_tensor_list)
    return total_tensor
    
def total_cnot_hamiltonian_high_dim(number_of_environments, coupling_coefficients, env_dim):
    """
    see my notes for hamiltonian structure
    system is a qubit
    envs are all qudits
    we can have any number of environments
    they all have matching dimension given by env_dim
    coupling_coefficients is a list number_of_environments long 
    it gives the g_k's in the Hamiltonian
    """
    system_dim = 2
    blocks = [0]*system_dim
    sys = [0]*system_dim
    tensors = [0]*system_dim
    blocks[0] = Qobj(np.eye(env_dim**number_of_environments), dims=[[env_dim]*number_of_environments,[env_dim]*number_of_environments])
    blocks[1] = tensor_cnot_hams_high_dim(number_of_environments, coupling_coefficients, env_dim)
    for i in range(system_dim):
        sys[i] = basis(system_dim, i)*basis(system_dim, i).dag()
        tensors[i] = tensor(sys[i], blocks[i])
    return sum(tensors) 

def tensor_cnot_hams_high_dim(number_of_environments, coupling_coefficients, env_dim):
    '''
    gives a block of environment cnot hamiltonians
    '''
    sigma_x = generalised_sigma_x(env_dim)
    p = perm(number_of_environments)
    for i in range(len(p)):
        p[i] = list(p[i])
        for j in range(number_of_environments):
            if p[i][j]==1:
                p[i][j] = Qobj(coupling_coefficients[j]*sigma_x)
            else:
                p[i][j] = Qobj(np.eye(env_dim))
    total_tensor_list = [0]*number_of_environments
    for i in range(number_of_environments):
        total_tensor_list[i] = tensor(p[i])
    total_tensor = sum(total_tensor_list)
    return total_tensor

def generalised_sigma_x(dim):
    #from gellmann_matrices import gellmann
    sigma_x = Qobj(np.zeros((dim,dim)))
    for d in range(dim):
        sigma_x += basis(dim,(d+1)%dim )*basis(dim,d).dag()
    
    #x = np.eye(dim)
    #sigma_x = np.rot90(x)
    #sigma_x = gellmann(2,1,dim)
    
    return sigma_x #+ sigma_x.dag()

###########################################################################
# generating states, evolving, time averaging, finding fidelities for random hamiltonians
###########################################################################

def initial_state(system_dim, number_of_environments, env_dim, env_initial_state=0, sys_probs=0):
    '''
    generates the system in a given superposition 
    sys_probs is a list of sys probs of length system_dim
    probs must sum to one
    if sys_probs is not defined, generates system in equal superposition
    all environmnets initialised in the ground state if env_initial_state=0
    if env_initial_state=n then env is excited in nth mode
    '''
    if not isinstance(number_of_environments, int) or number_of_environments <= 0:
        raise ValueError("number_of_environments must be a positive integer")
    if not isinstance(system_dim, int) or system_dim <= 1:
        raise ValueError("system dimension must be a positive integer >= 2")
    if not isinstance(env_dim, int) or env_dim <= 1:
        raise ValueError("environment dimension must be a positive integer >= 2")
    if sys_probs == 0:
        sys_probs = np.ones(system_dim)/system_dim
    if np.sum(sys_probs) != 1:
        raise Exception("Error: Probabilities do not sum to 1")
    ket = Qobj(np.sum([ np.sqrt(sys_probs[i])*basis(system_dim, [i]) for i in range(system_dim) ], axis=0))
    sys = ket * ket.dag()
    env = Qobj(np.diag([1 if i == env_initial_state else 0 for i in range(env_dim)]))
    return tensor([sys, *[env]*number_of_environments])

def initial_state_mixed(system_dim, number_of_environments, env_dim):
    '''
    generates the system in an equal superposition 
    all environmnets initialised in maximally mixed states
    '''
    sys = Qobj(np.ones((system_dim, system_dim))/system_dim)
    env = Qobj(np.eye(env_dim)/env_dim)
    return tensor([sys, *[env]*number_of_environments])

def evolution(t, initial_state, hamiltonian):
    ''' Hamiltonian and initial must be a Qobj'''
    #h_bar = 6.62607015e-34
    H = -1j * t * hamiltonian #/ h_bar
    unitary = H.expm()
    return unitary * initial_state * unitary.dag()

def time_average_state(system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0, sys_probs=0):
    number_of_environments = len(environment_dims)
    state = initial_state(system_dim, number_of_environments, env_dim, sys_probs=sys_probs)
    H = total_hamiltonian(system_dim, number_of_environments, env_dim)
    state_average = tensor( [Qobj(np.zeros((system_dim, system_dim))), *[Qobj(np.zeros((env_dim,           env_dim)))]*number_of_environments] )
    if number_to_average == 1:
        time = [max_time]
    else:
        time = np.linspace(0, max_time, number_to_average)
    for t in time:
        state_average = state_average + evolution(t, state, H)
    state_average = state_average / number_to_average
    return state_average

def infinite_time_average_state(system_dim, env_dim, environment_dims, sys_probs=0):
    number_of_environments = len(environment_dims)
    initial = initial_state(system_dim, number_of_environments, env_dim, sys_probs=sys_probs)
    H = total_hamiltonian(system_dim, number_of_environments, env_dim)
    state_average = tensor( [Qobj(np.zeros((system_dim, system_dim))), *[Qobj(np.zeros((env_dim,env_dim)))]*number_of_environments] )
    eigvals, eigvecs = scipy.linalg.eigh(H)
    eigvecs = [ eigvecs[:,i].reshape((np.shape(H)[0],1)) for i in range(np.shape(H)[0]) ]
    projs = [ eigvecs[i] @ eigvecs[i].T.conj() for i in range(np.shape(H)[0]) ]
    time_av = [ projs[i]@ initial.full() @ projs[i] for i in range(np.shape(H)[0]) ]
    time_av = np.sum(time_av, axis=0)
    time_av = time_av / np.trace(time_av)
    return time_av

def branches_of_one_environment(system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0):
    size = np.prod(environment_dims)
    state_average = time_average_state(system_dim, env_dim,  environment_dims, number_to_average, max_time)
    state_average = state_average.full()
    branches = state_average.reshape(system_dim,size,system_dim,size).diagonal(axis1=0, axis2=2) # this line pulls out the blocks along the diagonal 
    branches = branches / np.einsum("iij", branches)[np.newaxis, np.newaxis, :] # divide each block by its trace (renormalise) 
    fidelities = np.array([[fidelity_sophie(branches[:,:,i], branches[:,:,j]) for j in range(system_dim)] for i in range(system_dim)]) 
    # assuming one env so can just find fidelitys between all the blocks (number of blocks = system_dim)
    return fidelities[0,1].real


def branches_many_envs_2d_system(env_dim, number_of_environments, number_to_average=100, max_time=100000.0):
    '''
    All environments must be the same dimension
    system is fixed at 2D'''
    system_dim = 2
    environment_dims = [env_dim]*number_of_environments
    state = initial_state(system_dim, number_of_environments, env_dim)
    time_av_state = time_average_state(system_dim, env_dim,  environment_dims, number_to_average, max_time)
    systems_to_keep = [ list(range(0,i+2,1)) for i in range(number_of_environments)]
    fraction_sizes = range(1,number_of_environments+1)
    fractions = [ time_av_state.ptrace(i).full() for i in systems_to_keep ]
    frac_dims = [  int(i.shape[0] / system_dim) for i in fractions ]
    diags = [ fractions[i].reshape(system_dim,frac_dims[i],system_dim,frac_dims[i]).diagonal(axis1=0, axis2=2) for i in range(len(fractions)) ]
    diags_norm = [ diags[i] / np.einsum("iij", diags[i])[np.newaxis, np.newaxis, :] for i in range(len(diags)) ]
    blocks = [[ diags_norm[j][:,:,i] for i in range(system_dim) ] for j in range(len(diags_norm)) ]  
    fidelities = [ np.array([[fidelity_sophie(blocks[k][i], blocks[k][j]) for j in range(system_dim)] for i in range(system_dim)])[0,1].real for k in range(len(blocks)) ] 
    return fraction_sizes, fidelities

def fidelity_sophie(A, B): return ( np.trace(sqrtm(sqrtm(A) @ B @ sqrtm(A))) )**2

def trace_norm_sophie(A, B): return 0.5 * np.trace(sqrtm( (A-B).T.conj() @ (A-b) ) )

def branches_of_one_environment_av_over_many_hams(system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0):
    total_fid = 0
    size = np.prod(environment_dims)
    for _ in range(number_to_average):
        system_state = np.ones((system_dim, system_dim))/system_dim
        initial = initial_state(system_dim, 1, env_dim)
        hamiltonian = total_hamiltonian(system_dim, 1, env_dim)
        state_evolved = evolution(max_time, initial, hamiltonian).full()
        branches = state_evolved.reshape(system_dim,size,system_dim,size).diagonal(axis1=0, axis2=2)
        branches = branches / np.einsum("iij", branches)[np.newaxis, np.newaxis, :]
        fidelities = np.array([[fidelity_sophie(branches[:,:,i], branches[:,:,j]) for j in range(system_dim)] for i in range(system_dim)])
        total_fid += fidelities[0,1].real
    total_fid /= number_to_average
    return total_fid


def time_average_state_given_H(initial, Ham, system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0):
    number_of_environments = len(environment_dims)
    state_average = tensor( [Qobj(np.zeros((system_dim, system_dim))), *[Qobj(np.zeros((env_dim, env_dim)))]*number_of_environments] )
    if number_to_average == 1:
        time = [max_time]
    else:
        time = np.linspace(0, max_time, number_to_average)
    for t in time:
        state_average = state_average + evolution(t, initial, Ham)
    state_average = state_average / state_average.tr()
    return state_average


def infinite_time_average_state_given_H(initial, H, system_dim, env_dim, environment_dims):
    number_of_environments = len(environment_dims)
    eigvals, eigvecs = scipy.linalg.eig(H)
    eigvecs = [ eigvecs[:,i].reshape((np.shape(H)[0],1)) for i in range(np.shape(H)[0]) ]
    projs = [ eigvecs[i] @ eigvecs[i].T.conj() for i in range(np.shape(H)[0]) ]
    time_av = [ projs[i]@ initial.full() @ projs[i] for i in range(np.shape(H)[0]) ]
    time_av = np.sum(time_av, axis=0)
    return time_av


def branches_given_H(initial, hamiltonian, system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0):
    size = np.prod(environment_dims)
    state_average = time_average_state_given_H(initial, hamiltonian, system_dim, env_dim,  environment_dims, number_to_average, max_time)
    state_average = state_average.full()
    branches = state_average.reshape(system_dim,size,system_dim,size).diagonal(axis1=0, axis2=2) # this line pulls out the blocks along the diagonal 
    branches = branches / np.einsum("iij", branches)[np.newaxis, np.newaxis, :] # divide each block by its trace (renormalise) 
    fidelities = np.array([[fidelity_sophie(branches[:,:,i], branches[:,:,j]) for j in range(system_dim)] for i in range(system_dim)]) 
    # assuming one env so can just find fidelitys between all the blocks (number of blocks = system_dim)
    return fidelities[0,1].real


def fidelity_by_number_of_environments(number_of_environments, comparison_state, env_dim, max_time=100000.0):
    system_dim = comparison_state.shape[0]
    environment_dims = [env_dim]*number_of_environments
    initial = initial_state(system_dim, number_of_environments, env_dim)
    hamiltonian = total_hamiltonian(system_dim, number_of_environments, env_dim)
    state_evolved = evolution(max_time, initial, hamiltonian).full()
    assert np.isclose(1.0, state_evolved.trace()), f"The evolved state isn't close:\n{state_evolved.trace()}"
    traced = np.einsum("abcb->ac", state_evolved.reshape((system_dim, np.prod(environment_dims), system_dim, np.prod(environment_dims))))
    assert np.isclose(1.0, traced.trace()), f"THE TRACE ISN'T CLOSE:\n {traced},\n {traced.trace()}"
    return fidelity_sophie(comparison_state, traced).real

def average_fidelity(n_iterations, number_of_environments, comparison_state, environment_dim, max_time=100000.0):
    total = 0
    for _ in range(n_iterations):
        total += fidelity_by_number_of_environments(number_of_environments, comparison_state, environment_dim, max_time)
    return total/n_iterations

def generalised_overlap(A,B):
    """this is the sqrt of the fidelity"""
    return np.trace(sqrtm( sqrtm(A) @ B @ sqrtm(A) ))



####################################################################################################
# Plotting functions
####################################################################################################

def plot_complex_Qutip(complex_matrix, figsize=(10, 5), vmin=0, vmax=None, colour='magma_r'):
    complex_matrix = complex_matrix.full()
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].axis('off')
    axs[1].axis('off')
    real = axs[0].imshow(np.abs(complex_matrix), cmap=colour, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(real, cax=cax, orientation='vertical')
    cbar1.set_label('amplitude', rotation=90)
    im = axs[1].imshow(np.angle(complex_matrix), vmin=-np.pi, vmax=np.pi, cmap=colour)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im, cax=cax, orientation='vertical');
    cbar2.set_label('phase (rad)', rotation=90)
    axs[0].set_title(r'Amplitude')
    axs[1].set_title('Phase')
    return fig

def plot_complex(complex_matrix, figsize=(10, 10), vmin=0, vmax=None, colour='magma_r'):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].axis('off')
    axs[1].axis('off')
    real = axs[0].imshow(np.abs(complex_matrix), cmap=colour, vmin=vmin, vmax=vmax)
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar1 = fig.colorbar(real, cax=cax, orientation='vertical')
    cbar1.set_label('amplitude', rotation=90)
    
    im = axs[1].imshow(np.angle(complex_matrix), vmin=-np.pi, vmax=np.pi, cmap=colour)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar2 = fig.colorbar(im, cax=cax, orientation='vertical');
    cbar2.set_label('phase (rad)', rotation=90)
    
    axs[0].set_title(r'Amplitude')
    axs[1].set_title('Phase')
    return fig


# Max Tyler's colourise function
import matplotlib as mpl
from matplotlib.colors import hsv_to_rgb
def colourise(a: np.ndarray) -> np.ndarray:
    """Turn an array a with dimensions d into an HSV array
    Args:
        a (np.ndarray): The input array
    Returns:
        np.ndarray: The coloured array, ready to display with matplotlib
    """
    def map_value(v: float, min_v: float, max_v: float, min_o: float,
                  max_o: float) -> float:
        """Map the values v from the initial min and max vs to min and max os
        """
        ratio = (max_o - min_o) / (max_v - min_v)
        return (v - min_v) * ratio + min_o

    abs_a = np.abs(a).astype(np.float64)

    return hsv_to_rgb(
        np.moveaxis(
            np.array([
                map_value(np.angle(a), -np.pi, np.pi, 0, 1),
                np.ones_like(a),
                map_value(abs_a, 0, np.max(abs_a), 0, 1)
            ]), 0, -1).astype(np.float64))


## Vatshals colour map function
def Complex2HSV(z, hue_start=90):
    # get amplidude of z and limit to [rmin, rmax]
    rmin = min(np.abs(z).ravel())
    rmax = max(np.abs(z).ravel())
    amp = np.abs(z)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=1) + hue_start
    # HSV are values in range [0,1]
    h = (ph % 360 ) /360
    s = 0.85 * np.ones_like(h)
    v = (amp - rmin) / (rmax - rmin)
    return hsv_to_rgb(np.dstack((h,s,v)))



##########################################################################
#    Distance Measure (convex optimisation using cvx) 
###########################################################################

import cvxpy as cp

def find_projectors_single_env(rho, dim_sys=2):
    '''
    rho is the density matrix of 2 dim system and n dim env
    dim_sys=2 for now (need to generalise)
    cvx finds projectors on env that give
    max prob of success for distinguishing sys state
    '''
    dim_E = int(rho.shape[0]/2)

    block1 = np.array(rho[0:dim_E, 0:dim_E])
    block4 = np.array(rho[dim_E:dim_E*dim_sys, dim_E:dim_E*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)
    
    rho_envs = [ np.array((1/Probs[0])*block1), np.array((1/Probs[1])*block4) ]
    #print(rho_envs)
    
    # variables are projectors on the environment
    Proj_0 = cp.Variable((dim_E,dim_E))#, hermitian=True)
    Proj_1 = cp.Variable((dim_E,dim_E))#, hermitian=True)
    
    Projs = [Proj_0, Proj_1]
    
    success = cp.real(Probs[0]*cp.trace(rho_envs[0] @ Proj_0)+Probs[1]*cp.trace(rho_envs[1] @ Proj_1))

    # constraints:
    constraints = [Projs[i] >> 0 for i in range(dim_sys)]
    constraints += [ Projs[0] + Projs[1] == np.eye(dim_E) ] 
    constraints += [ Projs[i] << np.eye(dim_E) for i in range(dim_sys)]  
    
    prob = cp.Problem(cp.Maximize(success),constraints)
    prob.solve()
    
    return Proj_0.value, Proj_1.value

def generate_candidate_SBS_state_one_env(rho, Projs, dim_sys=2):
    """
    finds clostest SBS state to rho, given optimal projectors Projs
    """
    dim_E = int(rho.shape[0]/2)
    block1 = np.array(rho[0:dim_E, 0:dim_E])
    block4 = np.array(rho[dim_E:dim_E*dim_sys, dim_E:dim_E*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)
    
    rho_envs = [ np.array((1/Probs[0])*block1), np.array((1/Probs[1])*block4) ]
    
    q = [ (Probs[i]*np.trace(rho_envs[i]@Projs[i]))/( Probs[0]*np.trace(rho_envs[0]@Projs[0])+ Probs[1]*np.trace(rho_envs[1]@Projs[1]) )  for i in range(dim_sys) ]
   
    rho_env_tilde = [ (Projs[i]@rho_envs[i]@Projs[i])/(np.trace(rho_envs[i]@Projs[i])) for i in range(dim_sys) ]
    
    ground_state = Qobj(np.array([[1,0],[0,0]]))
    excited_state = Qobj(np.array([[0,0],[0,1]]))

    rho_SBS = q[0]*tensor( ground_state, Qobj(rho_env_tilde[0]) ) + q[1]*tensor( excited_state, Qobj(rho_env_tilde[1]) )
    
    return rho_SBS.full(), q



def trace_distance(A, B):
    return np.real(0.5 * np.trace(sqrtm( (A-B).conj().T @ (A-B) )))


def find_projectors(rho, dim_envs, dim_sys=2):
    '''
    rho is the density matrix of 2 dim system and n dim env
    dim_sys=2 for now (need to generalise)
    dim_envs = list of environment dimensions in order
    
    cvx finds projectors on env that give
    max prob of success for distinguishing sys state   
    
    returns projectors_list - this is dim_env long
    the kth element of the list contains a list of projectors
    for kth env for each i dim of system, i.e
    Projs[k] = [Pi_0^k, Pi_1^k] if dim_sys=2
    Projs[k][i] = Pi_{k}^{i}
    '''
    number_of_environments = len(dim_envs)
    dim_E_tot = int(rho.shape[0]/2)
    block1 = np.array(rho[0:dim_E_tot, 0:dim_E_tot])
    block4 = np.array(rho[dim_E_tot:dim_E_tot*dim_sys, dim_E_tot:dim_E_tot*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)
    
    # this density matrix is a tensor prod of all the envs
    rho_envs = [ np.array((1/Probs[0])*block1), np.array((1/Probs[1])*block4) ]
    
    # to get each env, we need to partially trace out everything else
    rho_envs = [ Qobj(rho_envs[i], dims=[dim_envs, dim_envs]) for i in range(dim_sys) ]
    # rho_envs_list is a list dim_sys long where the ith item is a list of the
    # env density matrices rho_i for all k envs
    rho_envs_list=[0]*dim_sys
    for i in range(dim_sys):
        rho_envs_list[i] = [ rho_envs[i].ptrace(k) for k in range(number_of_environments) ]
    
    # now we need to find optimal projectors for each environment separately
    # put all the projectors in a list 
    projectors_list = [0]*number_of_environments
    successes = [0]*number_of_environments
    for k in range(number_of_environments):
        # variables are projectors on the environment
        Proj_0 = cp.Variable((dim_envs[k],dim_envs[k]))#, hermitian=True)
        Proj_1 = cp.Variable((dim_envs[k],dim_envs[k]))#, hermitian=True)
    
        Projs = [Proj_0, Proj_1]
    
        success = cp.real(Probs[0]*cp.trace(rho_envs_list[0][k] @ Proj_0)+Probs[1]*cp.trace(rho_envs_list[1][k] @ Proj_1))
        #success = np.sum( [ cp.real(Probs[i]*cp.trace(rho_envs_list[i][k] @ Projs[i])) for i in range(dim_sys) ])

        # constraints:
        constraints = [Projs[i] >> 0 for i in range(dim_sys)]
        constraints += [ Projs[0] + Projs[1] == np.eye(dim_envs[k]) ] 
        constraints += [ Projs[i] << np.eye(dim_envs[k]) for i in range(dim_sys)]  
        #constraints += [ Projs[i]@Projs[i] == Projs[i] for i in range(dim_sys)]
        #constraints += [Projs[0]@Projs[1] == np.zeros((dim_envs[k],dim_envs[k])) ]# not necessary
    
        prob = cp.Problem(cp.Maximize(success),constraints)
        result = prob.solve()
        
        projectors_list[k] = [ Proj_0.value, Proj_1.value ]
        successes[k] = result
    
    return projectors_list #, successes


def generate_candidate_SBS_state(rho, projectors_list, dim_envs, dim_sys=2):
    """
    finds clostest SBS state to rho, given optimal projectors
    list - projectors_list, where
    projectors_list[k][i] = rho_E_{k}^i
    dim_envs = [ d_1, d_2, ... d_N ]
    Currently only works for a qubit system
    """
    number_of_environments = len(dim_envs)
    dim_E_tot = int(rho.shape[0]/2)
    block1 = np.array(rho[0:dim_E_tot, 0:dim_E_tot])
    block4 = np.array(rho[dim_E_tot:dim_E_tot*dim_sys, dim_E_tot:dim_E_tot*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)
    
    # this density matrix is a tensor prod of all the envs
    rho_envs = [ np.array((1/Probs[0])*block1), np.array((1/Probs[1])*block4) ]
    
    # to get each env, we need to partially trace out everything else
    rho_envs = [ Qobj(rho_envs[i], dims=[dim_envs, dim_envs]) for i in range(dim_sys) ]
    # rho_envs_list is a list dim_sys long where the ith item is a list of the
    # env density matrices rho_i for all k envs
    rho_envs_list=[0]*dim_sys
    for i in range(dim_sys):
        rho_envs_list[i] = [ np.array(rho_envs[i].ptrace(k)) for k in range(number_of_environments) ]
    
    rho_envs_tilde_list=[0]*dim_sys
    for i in range(dim_sys):
        rho_envs_tilde_list[i] = [ (projectors_list[k][i]@rho_envs_list[i][k]@projectors_list[k][i])/(np.trace(rho_envs_list[i][k]@projectors_list[k][i])) for k in range(number_of_environments) ]

    numerator0 = Probs[0]*np.prod([np.trace(rho_envs_list[0][k]@projectors_list[k][0]) for k in range(number_of_environments) ])
    denominator0 = np.sum([ Probs[i]*np.prod([np.trace(rho_envs_list[i][k]@projectors_list[k][i]) for k in range(number_of_environments) ]) for i in range(dim_sys) ])
    q0 = numerator0 / denominator0
    
    numerator1 = Probs[1]*np.prod([np.trace(rho_envs_list[1][k]@projectors_list[k][1]) for k in range(number_of_environments) ])
    denominator1 = np.sum([ Probs[i]*np.prod([np.trace(rho_envs_list[i][k]@projectors_list[k][i]) for k in range(number_of_environments) ]) for i in range(dim_sys) ])
    q1 = numerator1 / denominator1
    
    q = [q0,q1]    

    ground_state = Qobj(np.array([[1,0],[0,0]]))
    excited_state = Qobj(np.array([[0,0],[0,1]]))
    
    density_matrix_list = [0]*dim_sys
    density_matrix_list[0] = [ground_state]
    for k in range(number_of_environments):
        density_matrix_list[0].append(Qobj(rho_envs_tilde_list[0][k]))
        
    density_matrix_list[1] = [excited_state]
    for k in range(number_of_environments):
        density_matrix_list[1].append(Qobj(rho_envs_tilde_list[1][k]))

    rho_SBS = q[0]*tensor(density_matrix_list[0]) + q[1]*tensor(density_matrix_list[1])

    return rho_SBS.full(), q 

def generate_candidate_SBS_state_preserve_probs(rho, projectors_list, dim_envs, dim_sys=2):
    """
    finds a close SBS state to rho, given optimal projectors
    while preserving measurement statistics on system
    list - projectors_list, where
    projectors_list[k][i] = rho_E_{k}^i
    dim_envs = [ d_1, d_2, ... d_N ]
    Currently only works for a qubit system
    """
    number_of_environments = len(dim_envs)
    dim_E_tot = int(rho.shape[0]/2)
    block1 = np.array(rho[0:dim_E_tot, 0:dim_E_tot])
    block4 = np.array(rho[dim_E_tot:dim_E_tot*dim_sys, dim_E_tot:dim_E_tot*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)
    
    # this density matrix is a tensor prod of all the envs
    rho_envs = [ np.array((1/Probs[0])*block1), np.array((1/Probs[1])*block4) ]
    
    # to get each env, we need to partially trace out everything else
    rho_envs = [ Qobj(rho_envs[i], dims=[dim_envs, dim_envs]) for i in range(dim_sys) ]
    # rho_envs_list is a list dim_sys long where the ith item is a list of the
    # env density matrices rho_i for all k envs
    rho_envs_list=[0]*dim_sys
    for i in range(dim_sys):
        rho_envs_list[i] = [ np.array(rho_envs[i].ptrace(k)) for k in range(number_of_environments) ]
    
    rho_envs_tilde_list=[0]*dim_sys
    for i in range(dim_sys):
        rho_envs_tilde_list[i] = [ (projectors_list[k][i])/(np.trace(projectors_list[k][i])) for k in range(number_of_environments) ]

    ground_state = Qobj(np.array([[1,0],[0,0]]))
    excited_state = Qobj(np.array([[0,0],[0,1]]))
    
    density_matrix_list = [0]*dim_sys
    density_matrix_list[0] = [ground_state]
    for k in range(number_of_environments):
        density_matrix_list[0].append(Qobj(rho_envs_tilde_list[0][k]))
        
    density_matrix_list[1] = [excited_state]
    for k in range(number_of_environments):
        density_matrix_list[1].append(Qobj(rho_envs_tilde_list[1][k]))

    rho_SBS = Probs[0]*tensor(density_matrix_list[0]) + Probs[1]*tensor(density_matrix_list[1])

    return rho_SBS.full()


def distance_given_H(initial, hamiltonian, system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0):
    size = np.prod(environment_dims)
    state_average = time_average_state_given_H(initial, hamiltonian, system_dim, env_dim,  environment_dims, number_to_average, max_time)
    if len(environment_dims) == 1:
        rho_SBS, q = generate_candidate_SBS_state_one_env(state_average, find_projectors_single_env(state_average))
    else:
        projs = find_projectors(state_average, environment_dims, dim_sys=2)
        rho_SBS, q = generate_candidate_SBS_state(state_average, projs, environment_dims, system_dim)
    return trace_distance(state_average.full(), rho_SBS)


def distance_given_H_infinite(initial, hamiltonian, system_dim, env_dim, environment_dims, number_to_average=100, max_time=100000.0):
    size = np.prod(environment_dims)
    state_average = infinite_time_average_state_given_H(initial, hamiltonian, system_dim, env_dim,  environment_dims, number_to_average, max_time)
    if len(environment_dims) == 1:
        rho_SBS, q = generate_candidate_SBS_state_one_env(state_average, find_projectors_single_env(state_average))
    else:
        projs = find_projectors(state_average, environment_dims, dim_sys=2)
        rho_SBS, q = generate_candidate_SBS_state(state_average, projs, environment_dims, system_dim)
    return trace_distance(state_average, rho_SBS)


def find_rho_envs(rho, dim_envs, dim_sys=2):
    '''
    rho is the density matrix of 2 dim system and n dim env
    dim_sys=2 for now (need to generalise)
    dim_envs = list of macro environment dimensions in order

    returns rho_envs_list - this is number_of_environments long list
    the ith element of the list contains a list of rho_envs
    for all k envs corresponding to sys state i, i.e
    rho_envs_list[i][k] = rho_i^k
    
    '''
    number_of_environments = len(dim_envs)
    dim_E_tot = int(rho.shape[0]/2)
    block1 = np.array(rho[0:dim_E_tot, 0:dim_E_tot])
    block4 = np.array(rho[dim_E_tot:dim_E_tot*dim_sys, dim_E_tot:dim_E_tot*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)
    
    # this density matrix is a tensor prod of all the envs
    rho_envs = [ np.array((1/Probs[0])*block1), np.array((1/Probs[1])*block4) ]
    
    # to get each env, we need to partially trace out everything else
    rho_envs = [ Qobj(rho_envs[i], dims=[dim_envs, dim_envs]) for i in range(dim_sys) ]
    # rho_envs_list is a list dim_sys long where the ith item is a list of the
    # env density matrices rho_i for all k envs
    rho_envs_list=[0]*dim_sys
    for i in range(dim_sys):
        rho_envs_list[i] = [ rho_envs[i].ptrace(k) for k in range(number_of_environments) ]
        
    return rho_envs_list

def find_probs(rho, dim_envs, dim_sys=2):
    '''
    rho is the density matrix of 2 dim system and n dim env
    dim_sys=2 for now (need to generalise)
    dim_envs = list of macro environment dimensions in order

    returns Probs
    Probs[i] = p_{i}
    '''
    number_of_environments = len(dim_envs)
    dim_E_tot = int(rho.shape[0]/2)
    block1 = np.array(rho[0:dim_E_tot, 0:dim_E_tot])
    block4 = np.array(rho[dim_E_tot:dim_E_tot*dim_sys, dim_E_tot:dim_E_tot*dim_sys])
    
    Probs = [np.trace(block1) , np.trace(block4) ]
    Probs = np.real(Probs)

    return Probs


###########################################################################
# Equilibration functions 
###########################################################################

def deff(state,Ham):
    """
    Gives the effective dimension of a density matrix given a Hamiltonian 'Ham', in the presence of degeneracies
    """
    Hameig=Ham.eigenstates(sort='low')    
    unique,counts = np.unique(np.around(Hameig[0],8), return_counts=True)    
    counting=np.insert(np.cumsum(counts),0,0)
    projectorlist=[]
    for m in range(len(counting)-1):
        projectorhere= Hameig[1][0].proj()*0.0
        for n in range(counting[m],counting[m+1]):
            projectorhere=projectorhere+Hameig[1][n].proj()
        projectorlist.append(projectorhere)
    
    deffy=0.0
    for n in range(len(counting)-1): deffy = deffy + ((projectorlist[n]*state).tr())**2

    deffy = 1.0/deffy
    del Hameig
    return deffy


def inf_time_ave_degen(state,Ham):
    """
    Gives the infinite time average of a state as a pinching map with Hamiltonian 'Ham', in the presence of degeneracies
    """
    Hameig=Ham.eigenstates(sort='low')  
    rho_infdegen= state*0.0
    projectorlist=[]
    unique, counts = np.unique(np.around(Hameig[0],8), return_counts=True)
    counting=np.insert(np.cumsum(counts),0,0)
    degendict=dict(zip(unique, counts))
    for m in range(len(counting)-1):
        projectorhere= Hameig[1][0].proj()*0.0
        for n in range(counting[m],counting[m+1]):
            projectorhere=projectorhere+Hameig[1][n].proj()
        projectorlist.append(projectorhere)
    for n in range(len(projectorlist)):
        rho_infdegen=rho_infdegen+(projectorlist[n]*state*projectorlist[n])
    del Hameig
    return rho_infdegen


def tracedist2(A, B):
    """
    An alternate function defining the trace distance between two numpy matrices A and B
    """
    return np.real(np.abs(np.trace(A-B))**2)

def opnorm(op):
    """
    An alternate function defining the operator norm of an operator 'op'
    this should be equivalent to np.linalg.norm(op, 2)
    """
    opnorm=np.max(np.sqrt(np.maximum((op.dag()*op).eigenenergies(),0.0)))
    return opnorm

def random_variable_X(d_env, N):
    """
    X = \sum_{n_k} | <E_{n_k}|0> |^4 
    """
    X_rand_var = []
    for n in range(N):
        H = GUE(d_env)
        eigvals, eigvecs = np.linalg.eig(H)
        X = 0
        for nk in range(d_env):
            # sum over eigvecs in single env, i.e. sum_{nk}
            X = X +  np.abs(np.dot(eigvecs[:,nk] , basis(d_env,0)))**4

        X_rand_var.append( X ) 

    X_rand_var = np.array(X_rand_var).reshape(N)
    
    return X_rand_var
