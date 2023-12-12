#include "ham_utils.hpp"

#include <edlib/EDP/ConstructSparseMat.hpp>
#include <edlib/EDP/LocalHamiltonian.hpp>
#include <edlib/Basis/BasisJz.hpp>

#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>

#include <mpi.h>

#include <random>
#include <iostream>
#include <fstream>


template<typename RandomEngine>
Eigen::MatrixXd randomMatrix(RandomEngine& re, size_t row, size_t col) {
	std::normal_distribution urd(0.0, 1.0);

	Eigen::MatrixXd m(row, col);
	for(size_t i = 0; i < row; i++) {
		for(size_t j = 0; j < col; j++) {
			m(i, j) = urd(re);
		}
	}
	return m;
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	int mpi_size;
	int mpi_rank;

	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

	std::random_device rd;
	// std::mt19937_64 re{1557 + mpi_rank};
	std::mt19937_64 re{rd() + mpi_rank};

	const size_t N = std::stoi(argv[1]);
	const size_t total_iter = 1024;

	const std::vector<size_t> basis_vectors = [=]() {
		edlib::BasisJz<uint32_t> full_basis(2*N, N);
		return std::vector<size_t>(full_basis.begin(), full_basis.end());
	}();

	std::vector<size_t> x_n_over_two_indices;
	{
		edlib::BasisJz<uint32_t> basis(N, N / 2);
		std::vector<size_t> local_state(basis.begin(), basis.end());

		x_n_over_two_indices.reserve(basis.size()*basis.size());
		for(size_t i = 0; i < local_state.size(); i++) {
			for(size_t j = 0; j < local_state.size(); j++) {
				auto iter = std::find(basis_vectors.begin(), basis_vectors.end(),
						(local_state[i] << N) | local_state[j]);

				if (iter == basis_vectors.end()) {
					std::cerr << "Error";
					return 1;
				}
				x_n_over_two_indices.push_back(std::distance(basis_vectors.begin(), iter));
			}
		}
	}

	const std::vector<double> ts = [](){
		std::vector<double> res;
		for(size_t tidx = 0; tidx <= 100; tidx++) {
			res.push_back(tidx*0.1);
		}
		return res;
	}();
	Eigen::MatrixXd prob_means(ts.size(), total_iter / mpi_size);

	size_t init_st = std::distance(basis_vectors.begin(),
			std::find(basis_vectors.begin(), basis_vectors.end(), (1u<<N)-1));

	for(size_t iter = mpi_rank; iter < total_iter; iter += mpi_size) {
		std::cout << "Processing iter=" << iter << " at mpi_rank=" << mpi_rank << std::endl;

		Eigen::MatrixXd J = randomMatrix(re, N, N);
		edp::LocalHamiltonian<double> lh(2*N, 2);
		for(size_t i = 0; i < N; i++) {
			for(size_t j = 0; j < N; j++) {
				lh.addTwoSiteTerm(std::make_pair(i, N+j), J(i, j) * XXYY() / (2.0 * N));
			}
		}

		Eigen::MatrixXd ham = edp::constructSubspaceMat<double>(lh, basis_vectors);
		auto solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(ham);
		solver.compute(ham);
		const Eigen::VectorXd evals = solver.eigenvalues();
		const Eigen::MatrixXd evecs = solver.eigenvectors();

		for(size_t tidx = 0; tidx < ts.size(); tidx++) {
			const double t = ts[tidx];
			Eigen::VectorXcd exp_vals = Eigen::VectorXcd::Zero(evals.size());
			exp_vals.imag() = evals * (-t);
			exp_vals.array() = exp_vals.array().exp();

			Eigen::VectorXcd overlap = evecs * exp_vals.asDiagonal() * evecs.row(init_st).transpose();
			Eigen::VectorXd probs = overlap.cwiseAbs2();

			double prob_mean = 0.0;
			for(auto idx : x_n_over_two_indices) {
				prob_mean += probs[idx];
			}

			prob_mean /= x_n_over_two_indices.size();

			prob_means(tidx, iter / mpi_size) = prob_mean;
		}
	}

	std::ostringstream filename;
	filename << "PROB_MEAN_N" << N << "_" << mpi_rank << ".dat";
	
	auto filename_str = filename.str();

	std::ofstream fout(filename_str);
	fout << prob_means;
	fout.close();

	MPI_Finalize();

	return 0;
}
