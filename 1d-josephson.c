static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 1 dimension.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

static PetscReal const_hbar = 1.0545718176461565e-34;
static PetscReal const_e = 1.602176634e-19;
static PetscReal const_m_e = 9.1093837015e-31;


#include <slepceps.h>

int main(int argc,char **argv)
{
  Mat            H;           /* BdG Hamiltonian */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  PetscScalar    kr,ki;
  PetscScalar gap, sc_gap = 0.1, mu = 1.234;
  PetscInt       i,its,nconv;
  PetscInt       N_evs = 40, N_sites = 1000;
  FILE *file;
  PetscMPIInt mpi_size;
  
  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N_sites,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCheck(mpi_size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Josephson junction\n"));

  PetscPrintf(PETSC_COMM_WORLD,"sizeof(petsc scalar): %lu, sizeof(petsc real): %lu\n", sizeof(PetscScalar), sizeof(PetscReal));

  /* Open file */
  PetscCall(PetscFOpen(PETSC_COMM_WORLD, "output.dat", "w" , &file));
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
   
    
    PetscCall(MatSetValue(H,2*i,2*i+1, sc_gap, INSERT_VALUES));
    PetscCall(MatSetValue(H,2*i+1,2*i, sc_gap, INSERT_VALUES));
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
  PetscCall(EPSGetST(eps,&st));
  PetscCall(STSetType(st,STSINVERT));
  PetscCall(EPSSetDimensions(eps, N_evs, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(EPSSetTarget(eps, 0));
  PetscCall(EPSSetTolerances(eps, 1e-2, 1000));  
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (PetscReal Phi = -2*3.141; Phi < 2*3.141; Phi += 0.1) {
    for (i=N_sites/2; i < N_sites; ++i) {
      gap = sc_gap * PetscExpComplex(PETSC_i * Phi);
      PetscCall(MatSetValue(H,2*i,2*i+1, gap, INSERT_VALUES));
      PetscCall(MatSetValue(H,2*i+1,2*i, PetscConjComplex(gap), INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
   
    PetscCall(EPSSetOperators(eps,H,NULL));
    PetscCall(EPSSolve(eps));
    
    /*
      Optional: Get some information from the solver and display it
    */
    PetscCall(EPSGetIterationNumber(eps,&its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of iterations of the method: %" PetscInt_FMT "\n",its));

    PetscCall(EPSGetConverged(eps,&nconv));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %" PetscInt_FMT "\n\n",nconv));

    PetscCheck(nconv >= N_evs, PETSC_COMM_WORLD, 1, "did not converge");
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "%.5g\t", Phi));
    for (i = 0; i < N_evs; ++i) {
      PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "%.5g\t", (double ) kr));
    }
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "\n"));
  }

  return 0;
}
