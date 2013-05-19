// This header file contains various useful matrices and functions.

// Included header libraries.
#include <complex>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;
using namespace boost::numeric::ublas;

// Some type definitions for convenience.
typedef boost::numeric::ublas::matrix<complex<double> > matrix_type;
typedef boost::numeric::ublas::vector<complex<double> > state_type;

// Function prototypes.
matrix_type Kronecker  (const matrix_type &A, const matrix_type &B);
state_type  Diag       (const matrix_type &A);
matrix_type LeftOp     (const matrix_type &A);
matrix_type RightOp    (const matrix_type &A);
matrix_type Commutator (const matrix_type &A);
matrix_type Dissipator (const double gamma, const matrix_type &L);
state_type  Vec        (const matrix_type &A);
void initialise_operators (void);

// Memory for subsystem and system matrices.
matrix_type sigma_x(2, 2, 0);
matrix_type sigma_y(2, 2, 0);
matrix_type sigma_z(2, 2, 0);
matrix_type sigma_p(2 ,2, 0);
matrix_type sigma_m(2, 2, 0);

// Imaginary number.
complex<double> img(0, 1);

// Fill operators. Can't do this at the same time as allocation, so it's here.
// Since the operators are initialised with zeros, only the non-zero elements
// are changed here.
void initialise_operators(void) {
	sigma_x(0, 1) = 1;
	sigma_x(1, 0) = 1;

	sigma_y(0, 1) = -img;
	sigma_y(1, 0) = img;

	sigma_z(0, 0) = 1; 
	sigma_z(1, 1) = -1;

	sigma_p(0, 1) = 1;

	sigma_m(1, 0) = 1;
}

// Tensor product. Probably a better way of doing this.
matrix_type Kronecker(const matrix_type &A, const matrix_type &B) {
	matrix_type C(A.size1() * B.size1(), A.size2() * B.size2());

	for (int i = 0; i < A.size1(); i++) {
		for (int j = 0; j < A.size2(); j++) {
			for (int k = 0; k < B.size1(); k++) {
				for (int l = 0; l < B.size2(); l++) {
					C(i * B.size1() + k, j * B.size2() + l) = A(i, j) * B(k, l);
				}
			}
		}
	}

	return C;
}

// Vectorizes (stacks columns of) an input matrix.
state_type Vec(const matrix_type &A) {
	state_type vectorized(A.size1() * A.size2());

	for (int n = 0; n < A.size1(); n++) {
		for (int m = 0; m < A.size2(); m++) {
			vectorized(n * A.size1() + m) = A(m, n);
		}
	}

	return vectorized;
}

// Left multiplying operator to superoperator.
matrix_type LeftOp(const matrix_type &A) {
	return Kronecker(identity_matrix< complex<double> >(A.size1()), A);
}

// Right multiplying operator to superoperator.
matrix_type RightOp(const matrix_type &A) {
	return Kronecker(trans(A), identity_matrix< complex<double> >(A.size1()));
}

// Commutator to superoperator.
matrix_type Commutator(const matrix_type &A) {
	return LeftOp(A) - RightOp(A);
}

// Superoperator containing dissipative terms of the master equation.
matrix_type Dissipator(const double gamma, const matrix_type &L) {
	matrix_type dissipator = gamma * prod(LeftOp(L), RightOp(herm(L)));
	dissipator -= 0.5 * gamma * prod(LeftOp(herm(L)), LeftOp(L));
	dissipator -= 0.5 * gamma * prod(RightOp(L), RightOp(herm(L)));

	return dissipator;
}
