#include <slepceps.h>

static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 1 dimension.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

static PetscReal const_hbar = 1.0545718176461565e-34;
static PetscReal const_e = 1.602176634e-19;
static PetscReal const_m_e = 9.1093837015e-31;
static PetscReal const_mu_B = 9.2740100783e-24;
static PetscReal const_pi = 3.141592;

static int allocate_matrix(Mat *h, PetscInt N_sites_leads, PetscInt N_sites_JJ) {
  PetscInt N_sites = 2*N_sites_leads + N_sites_JJ;

  PetscCall(MatCreate(PETSC_COMM_WORLD,h));
  Mat H = *h;
  PetscCall(MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,4*N_sites,4*N_sites));
  PetscCall(MatSetFromOptions(H));
  PetscCall(MatSetUp(H));
  for (PetscInt i=0; i < N_sites; ++i) {
    // on-site
    // electron
    PetscCall(MatSetValue(H,4*i,4*i, 0, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+1,4*i+1,0, INSERT_VALUES));
  
    // Hole
    PetscCall(MatSetValue(H,4*i+2,4*i+2, 0, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+3,4*i+3, 0, INSERT_VALUES));

    // excHange coupling terms
      PetscCall(MatSetValue(H,4*i  ,4*i+1, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i  , 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+2,4*i+3, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+2, 0, INSERT_VALUES));
    
    // SC gap parameter
   
    if (i < N_sites_leads || i > N_sites_leads + N_sites_JJ) {
      PetscCall(MatSetValue(H,4*i  ,4*i+2, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i+3, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+2,4*i, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+1, 0, INSERT_VALUES));
    }
    // Hoppings
    if (i>0) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i-4,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i-3,0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i-2,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i-1,0,INSERT_VALUES));

      // for SOC (k x σ)
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i-3,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i-4,0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i-1,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i-2,0,INSERT_VALUES));
    }
    
    if (i<N_sites-1) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i+4,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i+5,0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i+6,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+7,0,INSERT_VALUES));
      
      // for SOC (k x σ)
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i+5,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i+4,0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i+7,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+6,0,INSERT_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  return 0;
}

static int set_normal_hamiltonian(Mat H, PetscInt N_sites_leads, PetscInt N_sites_JJ, PetscReal sc_gap, PetscReal mu, PetscReal t_hopping) {
  PetscInt N_sites = 2*N_sites_leads + N_sites_JJ;
  
  mu /= sc_gap;
  t_hopping /= sc_gap;
  
  for (int i=0; i < N_sites; ++i) {
    // on-site
    // electron
    PetscCall(MatSetValue(H,4*i,4*i, 2*t_hopping - mu, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+1,4*i+1, 2*t_hopping - mu, INSERT_VALUES));
  
    // hole
    PetscCall(MatSetValue(H,4*i+2,4*i+2, -(2*t_hopping - mu), INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+3,4*i+3, -(2*t_hopping - mu), INSERT_VALUES));

    // hoppings
    if (i>0) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i-4,-t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i-3,-t_hopping,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i-2,t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i-1,t_hopping,INSERT_VALUES));

    }
    
    if (i<N_sites-1) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i+4,-t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i+5,-t_hopping,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i+6,t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+7,t_hopping,INSERT_VALUES));
      
    }
  }
  return 0;
}

static int set_pairing(Mat H, PetscInt N_sites_leads, PetscInt N_sites_JJ, PetscReal Phi) {
  // need to assemble matrix after call
  // assume that H is scaled with 1/|Δ|

  // left lead
  for (PetscInt i=0; i < N_sites_leads; ++i) {
    PetscCall(MatSetValue(H,4*i  ,4*i+2, 1, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+1,4*i+3, 1, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+2,4*i, 1, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+3,4*i+1, 1, INSERT_VALUES));
  }

  // right lead
  PetscScalar gap = PetscExpComplex(PETSC_i * Phi); 
  for (PetscInt i=N_sites_JJ + N_sites_leads; i < (2*N_sites_leads + N_sites_JJ); ++i) {
    PetscCall(MatSetValue(H,4*i  ,4*i+2, gap, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+1,4*i+3, gap, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+2,4*i, PetscConjComplex(gap), INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+3,4*i+1, PetscConjComplex(gap), INSERT_VALUES));
  }
  return 0;
}

static int set_spin(Mat H, PetscInt N_sites, PetscReal spacing, PetscReal sc_gap, PetscReal gfactor, PetscReal B_x, PetscReal B_y, PetscReal k_y, PetscReal alpha_rashba) {
  // need to assemble matrix after call
  // assume that H is scaled with 1/|Δ|
  // H_Z = 0.5 g* μ_B * (B_x σ_x + B_y σ_y)
  // σ_x = [[0, 1], [1, 0]] , σ_y = [[0,-i], [i, 0]]
  PetscScalar E_z = 0.5 * gfactor * const_mu_B * (B_x - PETSC_i * B_y) / sc_gap;
  PetscReal SOC_term = alpha_rashba  / (2*spacing * sc_gap);

  // Rashba SOC gives onsite term α k_y σ_x / Δ
  PetscReal SOC_term_ky = alpha_rashba * k_y / sc_gap;
  
  for (PetscInt i=0; i < N_sites; ++i) {
    // onsite terms
    // electron
    PetscCall(MatSetValue(H,4*i  ,4*i+1,
                          E_z + SOC_term_ky, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+1,4*i,
                          PetscConjComplex(E_z)+SOC_term_ky, INSERT_VALUES));
    // hole
    PetscCall(MatSetValue(H,4*i+2,4*i+3,
                          E_z-SOC_term_ky, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+3,4*i+2,
                          PetscConjComplex(E_z)-SOC_term_ky, INSERT_VALUES));

    // hoppings -αk_xσ_y -> (α hbar / a) * [[0,1],[-1, 0]]
    if (i > 0) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i-3,-SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i-4,+SOC_term,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i-1,+SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i-2,-SOC_term,INSERT_VALUES));      

    }
    if (i<N_sites-1) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i+5,SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i+4,-SOC_term,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i+7,-SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+6,SOC_term,INSERT_VALUES));
    }
  }

  
  return 0;
}


int main(int argc,char **argv)
{
  Mat            H;           /* BdG Hamiltonian */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  PetscScalar    kr,ki;
  PetscReal m_eff = 0.03 * const_m_e;
  PetscReal sc_gap = 100e-6*const_e;
  PetscReal mu = 50e-3 * const_e;
  PetscReal k_F = 1/const_hbar * sqrt(2 * m_eff * mu);
  PetscReal v_F = const_hbar * k_F / m_eff;
  PetscReal xi_0 = const_hbar * v_F / (const_pi * sc_gap);
  
  PetscReal lambda_F = 2*const_pi / k_F;
  PetscReal spacing = lambda_F / 10;

  PetscReal t_hopping = const_hbar*const_hbar / (2 * m_eff * spacing*spacing);
  PetscInt       i,its,nconv;
  PetscReal JJ_length = 500e-9;
  PetscInt       N_evs = 32, N_sites, N_sites_JJ, N_sites_leads;
  FILE *file;
  PetscMPIInt mpi_size;

  
  
  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  //  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&N_sites,NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCheck(mpi_size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Josephson junction with spin\n"));
  printf("spacing = %.2g\n", spacing);
  printf("t / Δ = %.2g\n",  t_hopping / sc_gap);
  printf("λ_F = %.2g\n", lambda_F);
  printf("λ_F / a = %.2g\n", lambda_F / spacing);
  printf("ξ_0 = %.2g\n", xi_0);
  N_sites_leads = 3*xi_0 / spacing;
  N_sites_JJ = JJ_length / spacing;
  N_sites = 2*N_sites_leads + N_sites_JJ;
  printf("N_sites = %d, N_sites_JJ = %d\n", N_sites, N_sites_JJ);
  printf("L_electrode = %.2g\n", N_sites_leads * spacing);
  
  printf("ξ_0 / L_electrode = %.2g\n", xi_0 / (N_sites_leads * spacing));
  
  PetscPrintf(PETSC_COMM_WORLD,"sizeof(petsc scalar): %lu, sizeof(petsc real): %lu\n", sizeof(PetscScalar), sizeof(PetscReal));

  /* Open file */
  PetscCall(PetscFOpen(PETSC_COMM_WORLD, "output-spin.dat", "w" , &file));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, H_{BdG}Φ = EΦ
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 
  allocate_matrix(&H, N_sites_leads, N_sites_JJ);
  set_normal_hamiltonian(H,  N_sites_leads, N_sites_JJ, sc_gap, mu, t_hopping);
 
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
  PetscCall(EPSSetTolerances(eps, 1e-1, 1000));  
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  for (PetscReal k_y = 0; k_y < k_F; k_y += k_F / 10) {
    for (PetscReal Phi = -1.1*const_pi; Phi < 1.1*const_pi; Phi += 0.1) {
      printf("\n-------------------\nk_y / k_F = %.3g  Phi = %.3g pi\n", k_y / k_F, Phi / const_pi);
      set_pairing(H, N_sites_leads, N_sites_JJ, Phi);
      set_spin(H, N_sites, spacing, sc_gap,
               10,                  // g-factor
               0,                   // B_x
               0.1,                 // B_y
               k_y,                   // k_y
               10 *1e-3 * const_e * 1e-9); // α

    
      PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
      //    PetscCall(MatView(H, PETSC_VIEWER_STDOUT_SELF)); 
    
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
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "%.5g\t%.5g\t", k_y, Phi));
      for (i = 0; i < N_evs; ++i) {
        PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
        // printf("ev = %.3g\n", (double) kr);
        PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "%.5g\t", (double ) kr));
      }
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "\n"));
    }
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "\n"));
  }
  return 0;
}
