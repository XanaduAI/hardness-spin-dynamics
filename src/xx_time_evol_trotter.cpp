#include "external/BasisJz.hpp"
#include "external/npy.hpp"

#include <StateVectorKokkos.hpp>

#include <mpi.h>
#include <Eigen/Dense>

#include <cstdlib>
#include <vector>
#include <random>
#include <fstream>

using namespace Pennylane;

void single_trotter(const size_t N, StateVectorKokkos<double>& sv, const std::vector<double>& Js, double dt) {
	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			sv.applyOperation("IsingXY", {i, j+N}, false, {-Js[i*N + j]*dt/N});
		}
	}
	for(int64_t i = N-1; i >= 0; i--) {
		for(int64_t j = N-1; j >= 0; j--) {
			sv.applyOperation("IsingXY", {static_cast<size_t>(i), static_cast<size_t>(j)+N}, false, {-Js[i*N + j]*dt/N});
		}
	}
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	Kokkos::initialize(argc, argv);

	{
		int mpi_size;
		int mpi_rank;

		MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
		MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

		//const size_t total_iter = 1024;
		const size_t total_iter = mpi_size;
		const size_t N = 10;
		const double dt = 1e-6;
		const double max_t = 10.0;
		const uint32_t t_step_max = static_cast<uint32_t>((max_t / dt) + 0.5);
		const uint32_t record_t_per = static_cast<uint32_t>(0.1/dt+0.5);

		std::vector<Kokkos::complex<double>> sv_h(1u << (2*N));

		const std::vector<uint32_t> x_indices = [](){
			edlib::BasisJz<uint32_t> basis(2*N, N);
			return std::vector<uint32_t>(basis.begin(), basis.end());
		}();

		if (mpi_rank == 0){ // save x_indices
			npy::npy_data_ptr<uint32_t> d;
			d.data_ptr = x_indices.data();
			d.shape = {static_cast<unsigned long>(x_indices.size())};
			write_npy("BasisJz.npy", d);
		}
		MPI_Barrier(MPI_COMM_WORLD);

		Eigen::MatrixXd prob_means(t_step_max/record_t_per + 1, total_iter / mpi_size);
		Eigen::MatrixXd prob_sqr_means(t_step_max/record_t_per + 1, total_iter / mpi_size);

		std::random_device rd;
		std::mt19937_64 re{rd() + mpi_rank};
		std::normal_distribution<double> ndist(0.0, 1.0);

		std::vector<uint32_t> record_probs_tindices;
		for(int i = 0; i <= 4; i++) {
			double t = i * std::log(N);
			record_probs_tindices.emplace_back(static_cast<uint32_t>(t / dt + 0.5));
		}

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

			uint32_t tidx = 0;

			while(tidx <= t_step_max) {
				double t = dt * tidx;

				if(std::find(record_probs_tindices.begin(), record_probs_tindices.end(), tidx) != record_probs_tindices.end()) {
					Eigen::VectorXd probs_at_t(x_indices.size());
					for(size_t i = 0; i < x_indices.size(); i++) {
						auto val = sv_h[x_indices[i]];
						probs_at_t(i) = std::pow(val.real(), 2) + std::pow(val.imag(), 2);
					}

					char filename[255];
					sprintf(filename, "PROBS_TIME_AT_N%lu_T%03d_ITER%04u.npy", N, int(t*10+0.5), iter);

					npy::npy_data_ptr<double> d;
					d.data_ptr = probs_at_t.data();
					d.shape = {static_cast<unsigned long>(probs_at_t.size())};
					write_npy(filename, d);
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
		{ // save prob_mean_sqr
			std::ostringstream filename;
			filename << "PROB_MEAN_SQR_N" << N << "_" << mpi_rank << ".dat";
			
			auto filename_str = filename.str();

			std::ofstream fout(filename_str);
			fout << prob_sqr_means;
			fout.close();
		}
	}

	Kokkos::finalize();
	MPI_Finalize();
	return 0;
}
