import jax.numpy as jnp
import jax.random as jrnd
import jax.numpy.linalg as LA
from mpi4py import MPI
import sys
from hardness_spin_dynamics.xx_hamiltonian_solver import create_basis_u1, create_hamiltonian_xx
import numpy as np

comm = MPI.COMM_WORLD
mpi_rank = comm.Get_rank()
mpi_size = comm.Get_size()

if __name__ == '__main__':
    N = int(sys.argv[1])

    rnd_key = jrnd.PRNGKey(1557 + mpi_rank)

    basis_vectors = create_basis_u1(2*N, N)

    np.save("BasisJz.npy", np.array(basis_vectors))

    x_n_over_2_states = []
    half_basis_vectors = create_basis_u1(N, N//2)

    for v1 in half_basis_vectors:
        for v2 in half_basis_vectors:
            x_n_over_2_states.append((v1 << N) | v2)

    x_n_over_2_idices = jnp.searchsorted(basis_vectors, jnp.array(x_n_over_2_states))

    init_st = jnp.searchsorted(basis_vectors, (1 << N) - 1)

    TOTAL_ITER = 1024
    
    ts = []

    for i in range(5):
        ts.append(i * np.log(N))

    for iter in range(mpi_rank, TOTAL_ITER, mpi_size):
        print(f"Run iter={iter} from mpi_rank={mpi_rank}", flush=True)

        rnd_key, new_key = jrnd.split(rnd_key)
        J = jrnd.normal(new_key, (N, N), dtype=jnp.float64)

        ham = create_hamiltonian_xx(J, basis_vectors)

        evals, evecs = LA.eigh(ham)

        for tidx, t in enumerate(ts):
            overlap = evecs @ jnp.diag(jnp.exp(-1j*evals*t)) @ evecs[init_st, :].T
            probs = overlap.real ** 2 + overlap.imag ** 2

            np.save(f"PROBS_XX_N{N}_LOGT{tidx}_{iter}.npy", probs)
