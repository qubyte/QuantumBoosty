/** This program is designed to test the convenience of C++ with the
 *  Boost header library. In addition, it uses odeint, which is a
 *  in the Boost sandbox. You'll need to download and place the
 *  unpacked odeint library inside the numeric folder of Boost.
 *  
 *  This program makes use of special classes called functors.
 *  A functor is instatiated with some variables, and the functor
 *  instance then acts like a function with these variables baked in.
 *  This is particularly useful behaviour for defining generic
 *  Schrödinger and master equations which are handed a Hamiltonian
 *  and perhaps a dissipator.
 *  
 *  Mark S Everitt - 26/April/2012
 *  mark.s.everitt@gmail.com
 *  This  file is licensed under the Creative Commons Attribution
 *  Non-Commercial Schare Alike license (cc by-nc-nd)
 *  http://creativecommons.org/licenses/by-nc-sa/3.0/
 *  As such, I require derivative works to include my name and email
 *  address. This file or any derivative thereof may not be used for
 *  commercial purposes without my explicit written permission.
 */
#include <iostream>
#include <complex>
#include <boost/numeric/ublas/blas.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/assignment.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/odeint.hpp>

#define NDEBUG

using namespace std;
using namespace boost::numeric::odeint;
using namespace boost::numeric::ublas;

typedef boost::numeric::ublas::matrix< complex<double> > matrix_type;
typedef boost::numeric::ublas::vector< complex<double> > state_type;
typedef boost::numeric::odeint::controlled_runge_kutta< runge_kutta_cash_karp54<state_type> > stepper_type;

// Function prototypes.
matrix_type Kronecker(const matrix_type &A, const matrix_type &B);
state_type Diag(const matrix_type &A);
matrix_type LeftOp(const matrix_type &A);
matrix_type RightOp(const matrix_type &A);
matrix_type Commutator(matrix_type &A);
matrix_type Dissipator(const double gamma, const matrix_type &L);
state_type Vec(const matrix_type &A);
void initialise_operators(void);

// Memory for subsystem and system matrices.
matrix_type sigma_x(2,2,0), sigma_y(2,2,0), sigma_z(2,2,0), sigma_p(2,2,0), sigma_m(2,2,0);

// Imaginary number.
complex<double> img(0.0,1.0);

// Hamiltonian functor.
template <class optype> 
class hamiltonian
{
private:
    const double Omega, nu, Amp;
    optype H0;
    
public:
    hamiltonian (double Omega, double nu, double Amp) : Omega(Omega), nu(nu), Amp(Amp)
    {
        H0 = Omega*sigma_z*0.5;
    };
    
    optype operator() (double t) const
    {
        optype H(2,2);
        
        H = H0 + Amp*cos(nu*t)*sigma_x*0.5;
        return H;
    }
};

template <class optype, class vectype>
class liouville
{
private:
    // Initialise with a pointer to Hamiltonian function.
    const hamiltonian<optype> &H;
    const optype &L0;
    
public:
    liouville (const hamiltonian<optype> &_H, optype &_L0) : H(_H), L0(_L0) {};
    
    void operator () (const vectype &rho, vectype &drhodt, double t) const
    {
        optype Ht(H(t));
        drhodt = prod(-img*Commutator(Ht) + L0,rho);
    }
};

// Schrodinger equation functor.
template <class optype, class vectype>
class schrodinger
{
private:
    // schrodinger is initialised with a function pointer.
    const hamiltonian<optype> &H;
    
public:
    // Grab the input function pointer H and assign it to Hamiltonian.
    schrodinger (const hamiltonian<optype> &_H) : H(_H) {};
    
    // Function to hand to odeint.
    void operator() (const vectype &psi, vectype &dpsidt, double t) const
    {
        dpsidt = -img*prod(H(t),psi);
    }
};

// Vector observer functor. Used to print result of integrator.
class vector_observer
{
public:
    std::ostream &m_out;
    
    vector_observer(std::ostream &out) : m_out(out) {}
    
    void operator() (const state_type psit, double t) const
    {
        // If the above line doesn't work for you, sub in the one below.
        m_out << t;        
        for (size_t i=0; i < psit.size(); ++i) m_out << "\t" << psit(i).real() << "\t" << psit(i).imag();
        m_out << "\n";
    }
};

// Matrix observer functor. Used to print result of integrator.
class matrix_observer
{
public:
    std::ostream &m_out;
    
    matrix_observer(std::ostream &out) : m_out(out) {}
    
    void operator() (const state_type rhot, double t) const
    {
        // Find dimension of density matric from vector.
        int side = int(sqrt(float(rhot.size())));
        
        // Grab elements corresponding to diagonals in density matrix.
        const vector_slice<const state_type> diags(rhot, slice(0,side+1,side));
        
        // Print diagonal elements.
        m_out << t;
        for(size_t n = 0; n < diags.size(); n++) m_out << "\t" << diags[n].real();
        m_out << "\t" << sum(diags).real();
        m_out << endl;
    }
};

int main(int argc, char **argv)
{
    // Fill global matrices.
    initialise_operators();
    
    // Parameters.
    const double Omega = 1.0; // Gap between upper and lower levels.
    const double nu    = 1.0; // Drive frequency.
    const double amp   = 1.0; // Drive amplitude.
    
    // Decay rates
    const double gSm = 1/1e2; // Rate of depopulation.
    const double gSz = 1/1e2; // Rate of (pure) dephasing.
    
    // Parameters for nuclear drive.
    const double tend = 100.0; // Plot over a reasonable time to see gate behaviour.
        
    // Time independent parts of the Lindblad equation.
    matrix_type L0(Dissipator(gSm, sigma_m) + Dissipator(gSz, sigma_z));
    
    // Initialise psi and rho.
    state_type psiv(2); psiv <<= 1,0;
    state_type rhov(Vec(outer_prod(psiv, psiv)));
    
    hamiltonian<matrix_type> H(Omega,nu,amp);
    
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
    time_t start, end;
    time(&start);
    integrate_const(
                    stepper_type(default_error_checker<double>(1e-6, 1e-6)), // Stepper type with error checking.
                    liouville<matrix_type,state_type>(H,L0),                 // Integrate the master equation.
                    rhov,                                                    // Initial state.
                    0.0,                                                     // Start time.
                    tend,                                                    // Stop time.
                    abs(tend/500.0),                                         // Step time.
                    matrix_observer(cout)                                    // Output at each step.
                    );
    time(&end);
    
    return 0;
}

// Fill the operators. Can't do this at the same time as assignment, so it's in here.
void initialise_operators(void)
{    
    // Fill global spin matrices. These matrices were initialised to zero, so we need only
    // change the non-zero elements.
    sigma_x(0,1) = complex<double>(1.0,0.0);  sigma_x(1,0) = complex<double>(1.0,0.0);
    sigma_y(0,1) = complex<double>(0.0,-1.0); sigma_y(1,0) = complex<double>(0.0,1.0);
    sigma_z(0,0) = complex<double>(1.0,0.0);  sigma_z(1,1) = complex<double>(-1.0,0.0);
    sigma_p(0,1) = complex<double>(1.0,0.0);
    sigma_m(1,0) = complex<double>(1.0,0.0);
}

// Tensor product. Probably a better way of doing this.
matrix_type Kronecker(const matrix_type &A, const matrix_type &B)
{
    matrix_type C(A.size1()*B.size1(), A.size2()*B.size2());
    
    for (int i=0; i < A.size1(); i++) {
        for (int j=0; j < A.size2(); j++) {
            for (int k=0; k < B.size1(); k++) {
                for (int l=0; l < B.size2(); l++) {
                    C(i*B.size1()+k, j*B.size2()+l) = A(i,j)*B(k,l);
                }
            }
        }
    }
    
    return C;
}

// Vectorizes (stacks columns of) an input matrix.
state_type Vec(const matrix_type &A)
{
    state_type vectorized(A.size1()*A.size2());
    
    for (int n = 0; n < A.size1(); n++) {
        for (int m = 0; m < A.size2(); m++) {
            vectorized(n*A.size1()+m) = A(n,m);
        }
    }
    
    return vectorized;
}

// Left multiplying operator to superoperator.
matrix_type LeftOp(const matrix_type &A)
{
    return Kronecker(identity_matrix< complex<double> >(A.size1()),A);
}

// Right multiplying operator to superoperator.
matrix_type RightOp(const matrix_type &A)
{
    return Kronecker(trans(A), identity_matrix< complex<double> >(A.size1()));
}

// Commutator to superoperator.
matrix_type Commutator(matrix_type &A)
{
    return LeftOp(A) - RightOp(A);
}

// Superoperator containing dissipative terms of the master equation.
matrix_type Dissipator(const double gamma, const matrix_type &L)
{
    return gamma*prod(LeftOp(L),RightOp(herm(L))) - 0.5*gamma*prod(LeftOp(herm(L)),LeftOp(L)) - 0.5*gamma*prod(RightOp(L),RightOp(herm(L)));
}

