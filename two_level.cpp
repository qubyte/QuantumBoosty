// # A two level dissipative quantum system utilising C++ functors

/** 
 * This program is designed to test the convenience of C++ with the
 * Boost header library. In addition, it uses odeint, which is a
 * in the Boost sandbox. You'll need to download and place the
 * unpacked odeint library inside the numeric folder of Boost.
 *
 * This program makes use of special classes called functors.
 * A functor is instatiated with some variables, and the functor
 * instance then acts like a function with these variables baked in.
 * This is particularly useful behaviour for defining generic
 * Schrödinger and master equations which are handed a Hamiltonian
 * and perhaps a dissipator.
 *
 * Mark S Everitt - 26/April/2012
 * mark.s.everitt@gmail.com
 * This  file is licensed under the Creative Commons Attribution Non-Commercial
 * Schare Alike license (cc by-nc-nd)
 *
 * http://creativecommons.org/licenses/by-nc-sa/3.0/
 *
 * As such, I require derivative works to include my name and email address.
 * This file or any derivative thereof may not be used for commercial purposes
 * without my explicit written permission.
 */

// Turn off debugging mode.
#define NDEBUG

// Include header libraries.
#include <iostream>
#include <boost/numeric/odeint.hpp>
#include "quantum.hpp"

using namespace std;
using namespace boost::numeric::odeint;

typedef controlled_runge_kutta< runge_kutta_cash_karp54<state_type> > stepper_type;

// Hamiltonian functor. This class overloads function brackets for instances derived from it. This
// allows variables to be baked in when an instance is made, and the instance behave as a function
// with a single argument (time). Conveniently, in this case, nothing needs to be done in the
// constructor as there is sufficient information for the initialisers to do the work when an
// instance is made.
template <class optype> 
class hamiltonian {
private:
	const double nu;
	const optype H0;
	const optype drive;

public:
	hamiltonian (double Omega, double nu, double Amp)
		: nu(nu), H0(Omega * sigma_z * 0.5), drive(Amp * sigma_x * 0.5) {};

	optype operator () (double t) const {
		return H0 + cos(nu * t) * drive;
	}
};

// The Liouville functor. This functor takes a dissipation superoperator and a Hamiltonian functor
// by reference when an instance is made. The result is a function object that takes two vectors and
// a time as arguments, which is compatible with odeint.
template <class optype, class vectype>
class liouville {
private:
	const hamiltonian<optype> &H;
	const optype &L0;

public:
	liouville (const hamiltonian<optype> &H, optype &L0) : H(H), L0(L0) {};

	void operator () (const vectype &rho, vectype &drhodt, double t) const {
		drhodt = prod(-img * Commutator(H(t)) + L0, rho);
	}
};

// Schrodinger equation functor. Like the Liouville functor, this functor takes a reference to a
// Hamiltonian instance.
template <class optype, class vectype>
class schrodinger {
private:
	const hamiltonian<optype> &H;

public:
	schrodinger (const hamiltonian<optype> &H) : H(H) {};

	// Function to hand to odeint.
	void operator () (const vectype &psi, vectype &dpsidt, double t) const {
		dpsidt = -img * axpy_prod(H(t), psi);
	}
};

// Vector observer functor. Used to print result of integrator. This is implemented as a functor so
// that the output can be directed to an arbitrary destination. This function breaks up the real and
// imaginary parts of a state vector into separate columns in a tab delimited format. The zeroth
// column is the time of the snapshot represeted by a row.
class vector_observer {
public:
	std::ostream &m_out;

	vector_observer(std::ostream &out) : m_out(out) {}

	void operator() (const state_type psit, double t) const {
		m_out << t;		

		for (size_t i = 0; i < psit.size(); ++i) {
			m_out << "\t" << psit(i).real() << "\t" << psit(i).imag();
		}

		m_out << endl;
	}
};

// Matrix observer functor. Used to print result of integrator. Like the vector observer it is a
// functor to allow an arbitrary output desination. This extracts the diagonal elements of the
// density matrix and outputs the time, diagonal elements, and a sum of the diagonals so that the
// normalisation can be checked.
class matrix_observer {
public:
	std::ostream &m_out;

	matrix_observer(std::ostream &out) : m_out(out) {}

	void operator () (const state_type rhot, double t) const {
		// Find dimension of density matrix from vector.
		int side = int(sqrt(float(rhot.size())));

		// Grab elements corresponding to diagonals in density matrix.
		const vector_slice<const state_type> diags(rhot, slice(0, side + 1, side));

		// Print diagonal elements.
		m_out << t;

		for(size_t n = 0; n < diags.size(); n++) {
			m_out << "\t" << diags[n].real();
		}

		m_out << "\t" << sum(diags).real();
		m_out << endl;
	}
};

int main(int argc, char **argv) {
	// Fill global matrices.
	initialise_operators();

	// Parameters.
	const double Omega = 1.0; // Gap between upper and lower levels.
	const double nu	= 1.0; // Drive frequency.
	const double amp = 1.0; // Drive amplitude.

	// Decay rates
	const double gSm = 1.0 / 1e2; // Rate of depopulation.
	const double gSz = 1.0 / 1e2; // Rate of (pure) dephasing.

	// Parameters for nuclear drive.
	const double tend = 100.0; // Plot over a reasonable time to see gate behaviour.

	// Time independent parts of the Lindblad equation.
	matrix_type L0(Dissipator(gSm, sigma_m) + Dissipator(gSz, sigma_z));

	// Initialise psi and rho.
	state_type psiv(2); psiv <<= 1, 0;
	state_type rhov(Vec(outer_prod(psiv, psiv)));

	hamiltonian<matrix_type> H(Omega, nu, amp);

	// Uncomment the block function call below to try out pure evolution.
	/*
	integrate_const(
		stepper_type(default_error_checker<double>(1e-9, 1e-9)),
		schrodinger<matrix_type,state_type>(H),
		psiv,
		0.0,
		tend,
		abs(tend/2000.0),
		vector_observer(cout)
	);
	*/

	// The following should be uncommented for dissipative evolution.
	integrate_const(
		stepper_type(default_error_checker<double>(1e-6, 1e-6)), // Stepper with error checking.
		liouville<matrix_type,state_type>(H,L0),                 // Integrate the master equation.
		rhov,                                                    // Initial state.
		0.0,                                                     // Start time.
		tend,                                                    // Stop time.
		abs(tend/500.0),                                         // Step time.
		matrix_observer(cout)                                    // Output at each step.
	);

	return 0;
}
