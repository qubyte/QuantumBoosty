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
#include <complex>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;

// Some type definitions for convenience.
typedef boost::numeric::ublas::matrix<complex<double>> matrix_type;
typedef boost::numeric::ublas::vector<complex<double>> state_type;
typedef controlled_runge_kutta<runge_kutta_cash_karp54<state_type>> stepper_type;

// Function prototypes.
matrix_type Kronecker            (const matrix_type &A, const matrix_type &B);
state_type  Diag                 (const matrix_type &A);
matrix_type LeftOp               (const matrix_type &A);
matrix_type RightOp              (const matrix_type &A);
matrix_type Commutator           (const matrix_type &A);
matrix_type Dissipator           (const double gamma, const matrix_type &L);
state_type  Vec                  (const matrix_type &A);
void        initialise_operators (void);

// Memory for subsystem and system matrices.
matrix_type sigma_x(2, 2, 0);
matrix_type sigma_y(2, 2, 0);
matrix_type sigma_z(2, 2, 0);
matrix_type sigma_p(2 ,2, 0);
matrix_type sigma_m(2, 2, 0);

// Imaginary number.
complex<double> img(0, 1);

// For this version of the program, most of the important stuff happens inside
// main.
int main(int argc, char **argv) {
    // Fill global matrices.
    initialise_operators();

    // Parameters.
    const double Omega = 1.0; // Gap between upper and lower levels.
    const double nu    = 1.0; // Drive frequency.
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
    // remain the same.
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
        0.0,            // Start time.
        tend,           // Stop time.
        stepTime,       // Step time.
        matrix_observer // Output at each step.
    );
    
    return 0;
}

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
    matrix_type C(A.size1()*B.size1(), A.size2()*B.size2());
    
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
    auto vectorized = state_type(A.size1() * A.size2());
    
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

