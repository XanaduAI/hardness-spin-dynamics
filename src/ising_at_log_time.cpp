#include "external/BasisJz.hpp"
#include "external/npy.hpp"

#include <StateVectorKokkos.hpp>

#include <mpi.h>
#include <Eigen/Dense>

#include <string>
#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>

using namespace Pennylane;

void single_trotter(const size_t N, StateVectorKokkos<double>& sv, const std::vector<double>& Js, double dt) {
	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			sv.applyOperation("IsingXX", {i, j+N}, false, {-2.0*Js[i*N + j]*dt/N});
		}
	}
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	Kokkos::initialize(argc, argv);

	{
		int mpi_size;
		int mpi_rank;

		size_t N = std::stoul(argv[1]);
		if((N % 2 != 0) || (N > 10)) {
			fprintf(stderr, "Given N = %lu is not supported.\n", N);
			return 1;
		}

		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

		const size_t total_iter = 1024;
		//const size_t total_iter = mpi_size;
		const double dt = std::log(N);
		const size_t total_time_steps = 5;

		std::vector<Kokkos::complex<double>> sv_h(1u << (2*N));

		const std::vector<uint32_t> x_indices = [N](){
			edlib::BasisJz<uint32_t> basis(2*N, N);
			return std::vector<uint32_t>(basis.begin(), basis.end());
		}();

		if (mpi_rank == 0){ // save x_indices
			npy::npy_data_ptr<uint32_t> d;
			d.data_ptr = x_indices.data();
			d.shape = {static_cast<unsigned long>(x_indices.size())};
			char filename[255];
			sprintf(filename, "BasisJz_N%lu.npy", N);
			write_npy(filename, d);
		}

		MPI_Barrier(MPI_COMM_WORLD);

		std::random_device rd;
		std::mt19937_64 re{rd() + mpi_rank};
		std::normal_distribution<double> ndist(0.0, 1.0);

		for(uint32_t iter = mpi_rank; iter < total_iter; iter += mpi_size) {

			// Set Hamiltonian parameters
			std::vector<double> Js;
			Js.reserve(N*N);
			for(size_t i = 0; i < N*N; i++) {
				Js[i] = ndist(re);
			}

			// Initialize statevector
			StateVectorKokkos<double> sv(2*N);
			sv.setBasisState((1u << N) - 1);

			for(size_t tidx = 0; tidx < total_time_steps; tidx++) {
				const double t = dt*tidx;
				{ // record
					std::cout << "Processing time t = " << t << " at mpi_rank = " << mpi_rank << std::endl;
					sv.DeviceToHost(sv_h.data(), sv_h.size());

					Eigen::VectorXd probs_at_t(x_indices.size());
					for(size_t i = 0; i < x_indices.size(); i++) {
						auto val = sv_h[x_indices[i]];
						probs_at_t(i) = std::pow(val.real(), 2) + std::pow(val.imag(), 2);
					}

					char filename[255];
					sprintf(filename, "PROBS_TIME_AT_N%lu_LOGT%lu_ITER%04u.npy", N, tidx, iter);

					npy::npy_data_ptr<double> d;
					d.data_ptr = probs_at_t.data();
					d.shape = {static_cast<unsigned long>(probs_at_t.size())};
					write_npy(filename, d);
				}
				single_trotter(N, sv, Js, dt);
			}
		}
	}

	Kokkos::finalize();
	MPI_Finalize();
	return 0;
}
