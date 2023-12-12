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
		const double dt = 0.1;
		const size_t total_time_steps = 101;

		std::vector<Kokkos::complex<double>> sv_h(1u << (2*N));

		const std::vector<uint32_t> x_n_over_2_indices = [N](){
			edlib::BasisJz<uint32_t> basis(N, N/2);
			std::vector<uint32_t> basis_vec(basis.begin(), basis.end());
			
			std::vector<uint32_t> res;
			for(auto i: basis_vec) {
				for(auto j: basis_vec) {
					res.push_back((i << N) | j);
				}
			}
			return res;
		}();

		MPI_Barrier(MPI_COMM_WORLD);

		Eigen::MatrixXd prob_means(total_time_steps, total_iter / mpi_size);

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

					double prob_mean = 0.0;
					for(auto idx : x_n_over_2_indices) {
						double p = std::pow(sv_h[idx].real(), 2) + std::pow(sv_h[idx].imag(), 2);
						prob_mean += p;
					}

					prob_mean /= x_n_over_2_indices.size();
					prob_means(tidx, iter / mpi_size) = prob_mean;
				}
					
				single_trotter(N, sv, Js, dt);
			}
		}

		{ // save prob_means
			std::ostringstream filename;
			filename << "PROB_MEAN_N" << N << "_" << mpi_rank << ".dat";
			
			auto filename_str = filename.str();

			std::ofstream fout(filename_str);
			fout << prob_means;
			fout.close();
		}
	}

	Kokkos::finalize();
	MPI_Finalize();
	return 0;
}
