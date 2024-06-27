
#include <stdio.h>
#include <iostream> 
#include <cvode/cvode.h>            /* prototypes for CVODE fcts., consts.  */
#include <nvector/nvector_serial.h> /* access to serial N_Vector            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sundials/sundials_types.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sundials/sundials_context.h>
#include "headers/functions.h"
namespace py = pybind11;
using namespace std;
#if defined(SUNDIALS_EXTENDED_PRECISION)
#define GSYM "Lg"
#define ESYM "Le"
#define FSYM "Lf"
#else
#define GSYM "g"
#define ESYM "e"
#define FSYM "f"
#endif
#ifdef SUNDIALS_HAS_SUNREALTYPE
// newer SUNDIALS versions
#else
// older SUNDIALS versions
typedef realtype sunrealtype;
#endif


#define Ith(v, i) NV_Ith_S(v, i - 1) /* i-th vector component i=1..NEQ */
#define IJth(A, i, j) \
  SM_ELEMENT_D(A, i - 1, j - 1) /* (i,j)-th matrix component i,j=1..NEQ */

/* Problem Constants */


/* Functions Called by the Solver */




/* Private function to check function return values */

static int check_retval(void* returnvalue, const char* funcname, int opt);



/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */




py::object ODEsimulate(std::vector<double> times, std::unordered_map<std::string, double> params){
    #define NEQ   2               /* number of equations  */
    #define RTOL  SUN_RCONST(1.0e-5) /* scalar relative tolerance            */
    #define ATOL1 SUN_RCONST(1.0e-6) /* vector absolute tolerance components */
    #define ATOL2 SUN_RCONST(1.0e-6)
    #define ATOL3 SUN_RCONST(1.0e-5)
    #define ZERO SUN_RCONST(0.0)

    sundials::SUNContext sunctx;
    sunrealtype t, tout;
    N_Vector y;
    N_Vector abstol;
    SUNMatrix A;
    SUNLinearSolver LS;
    void* cvode_mem;
    int retval, iout;
    int retvalr;
    FILE* FID;
    std::vector<vector<double>> state_variables;
    int num_times=times.size();
    state_variables.resize(NEQ+1, std::vector<double>(num_times));
    py::object return_val=py::cast(state_variables);

    y         = NULL;
    abstol    = NULL;
    A         = NULL;
    LS        = NULL;
    cvode_mem = NULL;

    #ifdef SUNDIALS_VERSION_7
        retval = SUNContext_Create(SUN_COMM_NULL, &sunctx);
    #else
        retval = SUNContext_Create(NULL, &sunctx);
    #endif
    if (check_retval(&retval, "SUNContext_Create", 1)) { return (return_val); }

    /* Initial conditions */
    y = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)y, "N_VNew_Serial", 0)) { return (return_val); }

    /* Initialize y */
    //Current
    Ith(y, 1)=params["Cdl"]*mono_dE(params, 0, params["phase"]);
    
    Ith(y, 2)=0;
    for (int j=2; j<NEQ;j++){
        Ith(y, j+1) = 0;
    }
    

    /* Set the vector absolute tolerance */
    abstol = N_VNew_Serial(NEQ, sunctx);
    if (check_retval((void*)abstol, "N_VNew_Serial", 0)) { return (return_val); }

    Ith(abstol, 1) = ATOL1;
    Ith(abstol, 2) = ATOL2;
    Ith(abstol, 3) = ATOL3;

    /* Call CVodeCreate to create the solver memory and specify the
    * Backward Differentiation Formula */
    cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if (check_retval((void*)cvode_mem, "CVodeCreate", 0)) { return (return_val); }

    /* Call CVodeInit to initialize the integrator memory and specify the
    * user's right hand side function in y'=f(t,y), the initial time T0, and
    * the initial dependent variable vector y. */
    retval = CVodeInit(cvode_mem, single_e, 0, y);
    if (check_retval(&retval, "CVodeInit", 1)) { return (return_val); }
    CVodeSetUserData(cvode_mem, &params);
    /* Call CVodeSVtolerances to specify the scalar relative tolerance
    * and vector absolute tolerances */
    retval = CVodeSVtolerances(cvode_mem, RTOL, abstol);
    if (check_retval(&retval, "CVodeSVtolerances", 1)) { return (return_val); }
    


    /* Create dense SUNMatrix for use in linear solves */
    A = SUNDenseMatrix(NEQ, NEQ, sunctx);
    if (check_retval((void*)A, "SUNDenseMatrix", 0)) { return (return_val); }
    
    /* Create dense SUNLinearSolver object for use by CVode */
    LS = SUNLinSol_Dense(y, A, sunctx);
    if (check_retval((void*)LS, "SUNLinSol_Dense", 0)) { return (return_val); }

    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (check_retval(&retval, "CVodeSetLinearSolver", 1)) { return (return_val); }

    /* Set the user-supplied Jacobian routine Jac */
    retval = CVodeSetJacFn(cvode_mem, Jac);
    if (check_retval(&retval, "CVodeSetJacFn", 1)) { return (return_val); }

    for (iout=1; iout<num_times; iout++)
    {
        retval = CVode(cvode_mem, SUN_RCONST(times[iout]), y, &t, CV_NORMAL);
        
        state_variables[0][iout]=Ith(y, 1);
        state_variables[1][iout]=Ith(y, 2);
        for (int j=2; j<NEQ;j++){
            state_variables[j][iout]=Ith(y, j+1);
            
        }

        
        
        if (check_retval(&retval, "CVode", 1)) { break; }
        
        }


    /* Free memory */
    N_VDestroy(y);            /* Free y vector */
    N_VDestroy(abstol);       /* Free abstol vector */
    CVodeFree(&cvode_mem);    /* Free CVODE memory */
    SUNLinSolFree(LS);        /* Free the linear solver memory */
    SUNMatDestroy(A);         /* Free the matrix memory */
    SUNContext_Free(&sunctx); /* Free the SUNDIALS context */
    return(py::cast(state_variables));
    }





    /*
    * Check function return value...
    *   opt == 0 means SUNDIALS function allocates memory so check if
    *            returned NULL pointer
    *   opt == 1 means SUNDIALS function returns an integer value so check if
    *            retval < 0
    *   opt == 2 means function allocates memory so check if returned
    *            NULL pointer
    */

    static int check_retval(void* returnvalue, const char* funcname, int opt)
    {
    int* retval;

    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if (opt == 0 && returnvalue == NULL)
    {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
    }

    /* Check if retval < 0 */
    else if (opt == 1)
    {
    retval = (int*)returnvalue;
    if (*retval < 0)
    {
        fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
                funcname, *retval);
        return (1);
    }
    }

    /* Check if function returned NULL pointer - no memory allocated */
    else if (opt == 2 && returnvalue == NULL)
    {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return (1);
    }

    return (0);
    }


    PYBIND11_MODULE(SurfaceODESolver, m) {
    m.def("ODEsimulate", &ODEsimulate, "solve for I");
    m.def("mono_E", &mono_E, "potential_function");
    m.def("potential", &potential, "Get the full list of potential values");
    }
