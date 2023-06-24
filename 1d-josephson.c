/*
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   SLEPc - Scalable Library for Eigenvalue Problem Computations
   Copyright (c) 2002-, Universitat Politecnica de Valencia, Spain

   This file is part of SLEPc.
   SLEPc is distributed under a 2-clause BSD license (see LICENSE).
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/

static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 1 dimension.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            H;           /* BdG Hamiltonian */
  EPS            eps;         /* eigenproblem solver context */
  EPSType        type;
  PetscReal      error,tol;
  PetscScalar    kr,ki;
  PetscScalar gap, sc_gap = 0.1, mu = 1;
  PetscInt       i,nev,maxit,its,nconv;
  PetscInt       N_sites = 1000;
  
  PetscMPIInt mpi_size;
  
  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N_sites,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCheck(mpi_size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Josephson junction\n"));

  PetscPrintf(PETSC_COMM_WORLD,"sizeof(petsc scalar): %lu, sizeof(petsc real): %lu\n", sizeof(PetscScalar), sizeof(PetscReal));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, H_{BdG}Φ = EΦ
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(PETSC_COMM_WORLD,&H));
  PetscCall(MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,2*N_sites,2*N_sites));
  PetscCall(MatSetFromOptions(H));
  PetscCall(MatSetUp(H));

  for (i=0; i < N_sites; ++i) {
    // on-site
    // electron
    PetscCall(MatSetValue(H,2*i,2*i, 2.0 - mu, INSERT_VALUES));

    // hole
    PetscCall(MatSetValue(H,2*i+1,2*i+1, -2.0 + mu, INSERT_VALUES));

    
    // SC gap parameter
   
    
    PetscCall(MatSetValue(H,2*i,2*i+1, 0, INSERT_VALUES));
    PetscCall(MatSetValue(H,2*i+1,2*i, 0, INSERT_VALUES));
    // hoppings
    if (i>0) {
      //electron
      PetscCall(MatSetValue(H,2*i,2*(i-1),-1.0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,2*i+1,2*(i-1)+1,1.0,INSERT_VALUES));
    }
    
    if (i<N_sites-1) {
      //electron
      PetscCall(MatSetValue(H,2*i,2*(i+1),-1.0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,2*i+1,2*(i+1)+1,1.0,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create the eigensolver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
    Create eigensolver context
  */
  PetscCall(EPSCreate(PETSC_COMM_WORLD,&eps));

  /*
    Set operators. In this case, it is a standard eigenvalue problem
  */
  PetscCall(EPSSetProblemType(eps,EPS_HEP));

  /*
    Set solver parameters at runtime
  */
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (PetscReal Phi = -3.141; Phi < 3.141; Phi += 0.1) {
    for (i=0; i < N_sites; ++i) {
      gap =  i > N_sites/2 ? sc_gap * PetscExpComplex(PETSC_i * Phi) : sc_gap;
      PetscCall(MatSetValue(H,2*i,2*i+1, gap, INSERT_VALUES));
      PetscCall(MatSetValue(H,2*i+1,2*i, PetscConjComplex(gap), INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
   
    PetscCall(EPSSetOperators(eps,H,NULL));
    /* PetscCall(EPSSetUp(eps)); */
    PetscCall(EPSSolve(eps));
    
    /*
      Optional: Get some information from the solver and display it
    */
    PetscCall(EPSGetIterationNumber(eps,&its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));
    PetscCall(EPSGetType(eps,&type));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Solution method: %s\n\n",type));
    PetscCall(EPSGetDimensions(eps,&nev,NULL,NULL));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of requested eigenvalues: %" PetscInt_FMT "\n",nev));
    PetscCall(EPSGetTolerances(eps,&tol,&maxit));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Stopping condition: tol=%.4g, maxit=%" PetscInt_FMT "\n",(double)tol,maxit));

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Display solution and clean up
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*
      Get number of converged approximate eigenpairs
    */
    PetscCall(EPSGetConverged(eps,&nconv));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %" PetscInt_FMT "\n\n",nconv));

    if (nconv>0) {
      /*
        Display eigenvalues and relative errors
      */
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "           k          ||Ax-kx||/||kx||\n"
                            "   ----------------- ------------------\n"));

      for (i=0;i<nconv;i++) {
        /*
          Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
          ki (imaginary part)
        */
        PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
        /*
          Compute the relative error associated to each eigenpair
        */
        PetscCall(EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error));

        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g\n",(double)kr,(double)error));
      }
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));
    }
  }
  /*
     Free work space
  */
  /* PetscCall(EPSDestroy(&eps)); */
  /* PetscCall(MatDestroy(&H)); */
  /* PetscCall(VecDestroy(&xr)); */
  /* PetscCall(VecDestroy(&xi)); */
  /* PetscCall(SlepcFinalize()); */
  return 0;
}
