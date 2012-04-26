# QuantumBoosty

A few files to demonstrate some useful C++ features for integrating Schrödinger and master equations. These files use ublas (from [Boost](www.boost.org)) and the [odeint](www.odeint.com) library, which is currently in the Boost sandbox. Users may also find the [expm.hpp](https://www.dbtsai.com/blog/2008-11-25-matrix-exponential/) implementation of the general matrix exponential helpful for time independent systems.

## Background

One of the largest problems I came up against when working with C or Fortran was the lack of useful integrators. Obviously integrators are available, but I found myself jumping through hoops. In the case of C, I was using the GSL library, which both defines its own complex type (C has had its own since C99) and then fails to use it for the integrator routines, necessitating the doubling of the size of the system. That's not all that difficult in practise, but it's a an additional layer in the code that doesn't help readability. Worse, the routines rely on arcane drivers which means cracking out the manual, which is also unhelpful. In the case of Fortran, the VODE integrators are good, but require your program to be arranged in a particular way. Not so good for plug and play ODE solving.

Recently I came across the C++ [odeint](www.odeint.com) library. I was so impressed by its flexibility that I actually started using C++ (having had no prior experience) just for the convenience of it. I'm glad that I did this, because C++ has some other very useful features that make solving quantum mechanical systems simpler. Probably the most important feature that I use (perhaps abuse) is functors, a kind class with which instances can be used as functions.

## Some explanation

The code may be unfamiliar to a C coder in places. In this section I attempt to explain some of the features that C++ allows.

### Functors
C++ allows _operator overloading_. Usually it's best to avoid this feature. It allows operators in the language to be extended to new classes. For example, you could define your own matrix class, and overload the + operator to allow matrix addition in the way you would expect. There's no need to do this though, since several libraries already implement matrices for C++, and I use ublas from Boost.

It may surprise you to learn (it surprised me) that the brackets of a function can be overloaded too. This means that you can make a instance of a class behave like a function when it has brackets next to it. The cool thing is that some of the guts of the function can be determined upon instantiation.

In the case of the code two_level.cpp, I've used a simple Hamiltonian representing a driven two level system. The Hamiltonian is instantiated with the level splitting, the frequency of the drive, and the amplitude of the drive:

    hamiltonian H(Omega,nu,Amp);

Now we can use this instance to give us the Hamiltonian at any time:
    
    matrix_type Ht = H(t);
    
Looking closer at the functor:
    
    template <class optype> 
    class hamiltonian
    {
    private:
        const double Omega, nu, Amp;
        optype H0;
    
    public:
        hamiltonian (double _Omega, double _nu, double _Amp) : Omega(_Omega), nu(_nu), Amp(Amp)
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

Hopefully much of this is understandable. A C++ class is like a C structure, but it can have internal functions called _member functions_ or _methods_. The addition of these functions means that it makes sense for the class to keep some information (think book keeping) to itself, so both member functions and data can be declared as public (anyone can take a look) or private (for the class only). This is massively over simplifying, but either you know this or you can wiki it.

One member function takes the name of the class. This is the function called when an instance of the class is produced. The prototype tells us that the instance will take three double precision floats as arguments, and these get assigned to private variables. Underscores here are used to differentiate between arguments and member data. The content of the function generates the static part of the Hamiltonian, which remains unchanged after the instance is produced. This is important! You can get lots of stuff like matrix multiplication out of the way like this.

The other function in public is the overloaded function brackets. When you call an instance like a function, it's this that gets executed.

The code in two_level.cpp goes a step further. If it's used to integrate a master equation, then a reference of an instance of the Hamiltonian functor is handed to the Liouville functor (functors in functors!).

### Superoperators
A master equation on paper is a matrix valued differential equation. In computational terms, this is a real pain to deal with. On the face of it, it seems like even the time independent terms need to be formulated as functions. A trick can be used to avoid this, with the cost of increasing the size of your system. By vectorising the density matrix (stack the columns, with the left-most on top) all operators, whether they are on the left, right or both sides of the density matrix. There's a very handy page on wikipedia on [vectorisation](http://en.wikipedia.org/wiki/Vectorization_\(mathematics\)) which shows how $n\times n$ operator matrices can be turned into $n^2\times n^2$ superoperator matrices always acting on the left.

Once this is done, the situation is very similar to solving the Schrödinger equation by integration. The master equation can be summed up into a single matrix (which may be time dependent of course) acting on the left of a vector representing the density operator.

To construct superoperators additional functions were needed. In two_level.cpp you'll find a function for performing the vectorisation of a matrix, functions to transform operators acting on the left or right into superoperators and some additional convenience functions for constructing terms in a master equation.