#include "ham_utils.hpp"

#include <edlib/EDP/ConstructSparseMat.hpp>
#include <edlib/EDP/LocalHamiltonian.hpp>
#include <edlib/Basis/BasisJz.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <concepts>
#include <span>

namespace py = pybind11;

template<std::unsigned_integral UINT>
static UINT binom_coeff(UINT n, UINT k) {
	auto res = static_cast<UINT>(1U);
	// Since C(n, k) = C(n, n-k)
	if(k > n - k)
	{
		k = n - k;
	}

	// Calculate value of
	// [n * (n-1) *---* (n-k+1)] / [k * (k-1) *----* 1]
	for(UINT i = 0U; i < k; ++i)
	{
		res *= (n - i);
		res /= (i + 1);
	}
	return res;
}

py::array_t<uint32_t> createBasisU1(uint32_t n, uint32_t n_up) {
	edlib::BasisJz<uint32_t> basis(n, n_up);

	auto res = py::array_t<uint32_t>({basis.size()});
	py::buffer_info res_buffer = res.request();

	std::span res_sp(static_cast<uint32_t*>(res_buffer.ptr), res_buffer.size);
	std::copy(basis.begin(), basis.end(), res_sp.begin());

	return res;
}

py::array_t<std::complex<double>> createHamiltonianXX(py::array_t<double> Js,
		py::array_t<uint32_t> basis_vectors) {

	py::buffer_info js_buffer = Js.request();
	py::buffer_info basis_vectors_buffer = basis_vectors.request();

	if (js_buffer.ndim != 2) {
		throw std::invalid_argument("Js must be a two-dimensional array");
	}

	if(js_buffer.shape[0] != js_buffer.shape[1]) {
		throw std::invalid_argument("Js must be a square matrix");
	}

	if(basis_vectors_buffer.ndim != 1) {
		throw std::invalid_argument("basis_vector must be a one-dimensional array");
	}

	const uint32_t N = js_buffer.shape[0];

	const uint32_t mat_dim = basis_vectors.size();

	auto res = py::array_t<double>({mat_dim, mat_dim});
	py::buffer_info res_buffer = res.request();

	Eigen::Map<Eigen::MatrixXd> J(static_cast<double*>(js_buffer.ptr), N, N);
	Eigen::Map<Eigen::MatrixXd> ham(static_cast<double*>(res_buffer.ptr), mat_dim, mat_dim);

	edp::LocalHamiltonian<double> lh(2*N, 2);
	for(size_t i = 0; i < N; i++) {
		for(size_t j = 0; j < N; j++) {
			lh.addTwoSiteTerm(std::make_pair(i, N+j), J(i, j) * XXYY() / (2.0 * N));
		}
	}

	ham = edp::constructSubspaceMat<double>(lh, 
			std::span(static_cast<uint32_t*>(basis_vectors_buffer.ptr), basis_vectors_buffer.size));

	return res;
}


PYBIND11_MODULE(xx_hamiltonian_solver, m) {
	m.def("create_basis_u1", createBasisU1)
	 .def("create_hamiltonian_xx", createHamiltonianXX);
}
