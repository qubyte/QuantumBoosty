// # A two level dissipative quantum system utilising C++11 lambdas

/**
 * This program is designed to test the convenience of C++ with the Boost
 * header library. In addition, it uses odeint, which recently graduated from
 * the Boost sandbox. If you're using a Boost version prior to 1.53, then you
 * will need to download and place the unpacked odeint library inside the
 * numeric folder of Boost.
 *
 * This version of the program removes functors and instead uses lambdas, which
 * were introduced with C++11. These are written inline, and behave like functor
 * singletons. The ability to inline, along with to the reduction in lines of
 * code when compared with functors, makes them extremely convenient. The cost
 * of this approach is that it is somewhat less general than the functor
 * approach.
 *  
 * Mark S Everitt - 11/May/2013
 * mark.s.everitt@gmail.com
 * This file is licensed under the Creative Commons Attribution Non-Commercial
 * Share Alike license (cc by-nc-nd)
 * 
 * http://creativecommons.org/licenses/by-nc-sa/3.0/
 * 
 * As such, I require derivative works to include my name and email address.
 * This file or any derivative thereof may not be used for commercial purposes
 * without my explicit written permission.
 */

// Turn off debugging mode.
#define NDEBUG

// Included header libraries.
#include <iostream>
#include <boost/numeric/odeint.hpp>
#include "quantum.hpp"

using namespace std;
using namespace boost::numeric::odeint;

typedef controlled_runge_kutta<runge_kutta_cash_karp54<state_type>> stepper_type;

// For this version of the program, most of the important stuff happens inside
// main.
int main(int argc, char **argv) {
	// Fill global matrices.
	initialise_operators();

	// Parameters.
	const double Omega = 1.0; // Gap between upper and lower levels.
	const double nu	= 1.0; // Drive frequency.
	const double amp   = 1.0; // Drive amplitude.

	// Decay rates
	const double gSm = 1.0 / 1e2; // Rate of depopulation.
	const double gSz = 1.0 / 1e2; // Rate of (pure) dephasing.

	// Plot over a reasonable time to see behaviour.
	const double tend = 100.0;

	// The time independent part of the Hamiltonian.
	const auto H0 = matrix_type(Omega * sigma_z * 0.5);

	// This lambda is an equation that returns the Hamilton matrix at a time t.
	// It captures the time independent part from the surrounding scope by
	// reference, and the drive frequency and amplitude by value. By doing this,
	// it avoids a lot of repeated calculations. Note the alternative return
	// syntax used by C++11 lambdas.
	auto hamiltonian = [&H0, nu, amp] (double t) -> matrix_type {
		return H0 + amp * cos(nu * t) * sigma_x * 0.5;
	};

	// This lambda is a function that operates on the arguments it is given. It
	// is essentially the Schrodinger equation, expressed in a way that odeint
	// likes.
	auto schrodinger = [&hamiltonian] (const state_type &psi, state_type &dpsidt, double t) {
		dpsidt = -img * prod(hamiltonian(t), psi);
	};

	// Time independent parts of the Lindblad equation.
	auto L0 = matrix_type(Dissipator(gSm, sigma_m) + Dissipator(gSz, sigma_z));

	// Like the Hamiltonian lambda, this lambda captures what it needs from the
	// surrounding scope to avoid more calculation than needed. Note that like
	// the Schrodinger lambda, the arguments are two vectors and a double. This
	// means that the density matrix must be vectorised. This is convenient,
	// because in the vectorised form, the time independent part of the Lindblad
	// equation can be pre-calculated.
	auto liouville = [&hamiltonian, &L0] (const state_type &rho, state_type &drhodt, double t) {
		matrix_type Ht(hamiltonian(t));
		drhodt = prod(-img * Commutator(Ht) + L0, rho);
	};

	// This lambda is like the matrix observer, but used for unitary evolution.
	// Again, it is an example, and can be customised as long as the arguments
	// remain the same. Note the range based for to iterate over the vector
    // elements.
	auto vector_observer = [] (const state_type psit, double t) {
		cout << t;

		for (const auto& element: psit) {
			cout << "\t" << element.real() << "\t" << element.imag();
		}

		cout << endl;
	};

	// This lambda is used for observing the state of a dissipative system
	// whilst it is being calculated. It can be customised, so long as the
	// arguments remain unchanged. As it is, no variables are captured, but if
	// you wanted to, for example, get the fidelity of the integrated system
	// with the prediction from some model, then variable capture would be a
	// good approach.
	auto matrix_observer = [] (const state_type rhot, double t) {
		int side = int(sqrt(float(rhot.size())));

		const vector_slice<const state_type> diags(rhot, slice(0, side + 1, side));

		cout << t;

		for (const auto& element: diags) {
			cout << "\t" << element.real();
		}

		cout << "\t" << sum(diags).real();
		cout << endl;
	};

	// Initialise psi and rho.
	state_type psiv(2, 0); psiv <<= 1,0;
	state_type rhov(Vec(outer_prod(psiv, psiv)));

	// The stepper for odeint. The values given are absolute and relative
	// tolerances, a lot like MATLAB ODE solvers. Tweak these! Note the use of
	// `auto`. We let the compiler handle the type.
	auto stepper = stepper_type(default_error_checker<double>(1e-6, 1e-6));

	// The step time between observations.
	double stepTime = abs(tend / 500.0);

	// Uncomment the block function call below to try out pure evolution.
	/*
	integrate_const(
		stepper,
		schrodinger,
		psiv,
		0.0,
		tend,
		stepTime,
		vector_observer
	);
	*/

	// The following should be uncommented for dissipative evolution.
	integrate_const(
		stepper,        // Stepper with error checking.
		liouville,      // The master equation.
		rhov,           // Initial state.
		0.0,			// Start time.
		tend,           // Stop time.
		stepTime        // Step time.
		matrix_observer // Output at each step.
	);

	return 0;
}
