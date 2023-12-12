#pragma once

#include <Eigen/Sparse>

Eigen::SparseMatrix<double> XXYY() {
	// XX + YY
	Eigen::SparseMatrix<double> m(4,4);
	m.coeffRef(1, 2) = 2.0;
	m.coeffRef(2, 1) = 2.0;
	m.makeCompressed();
	return m;
}
Eigen::SparseMatrix<double> XX() {
	// XX 
	Eigen::SparseMatrix<double> m(4,4);
	m.coeffRef(3, 0) = 1.0;
	m.coeffRef(2, 1) = 1.0;
	m.coeffRef(1, 2) = 1.0;
	m.coeffRef(0, 3) = 1.0;
	m.makeCompressed();
	return m;
}

Eigen::SparseMatrix<double> pauliX() {
	Eigen::SparseMatrix<double> m(2,2);
	m.coeffRef(1, 0) = 1.0;
	m.coeffRef(0, 1) = 1.0;
	m.makeCompressed();
	return m;
}
