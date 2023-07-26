#include <slepceps.h>
#include <sys/stat.h> // for mkdir
#include <time.h>
static char help[] = "Standard symmetric eigenproblem corresponding to the Laplacian operator in 1 dimension.\n\n"
  "The command line options are:\n"
  "  -n <n>, where <n> = number of grid subdivisions = matrix dimension.\n\n";

static PetscReal const_hbar = 1.0545718176461565e-34;
static PetscReal const_e = 1.602176634e-19;
static PetscReal const_m_e = 9.1093837015e-31;
static PetscReal const_mu_B = 9.2740100783e-24;
static PetscReal const_pi = 3.141592;

static PetscReal spacing;
static PetscReal t_hopping;
static PetscReal B_y = 0.5;
static PetscReal B_x = 0;
static PetscReal gfactor = 15;



static  PetscReal sc_gap = 100e-6*const_e;


static PetscReal alpha_rashba =  30; // meV nm

static  PetscInt N_sites, N_sites_JJ, N_sites_leads;
static Mat H;

static int allocate_matrix() {
  PetscCall(MatCreate(PETSC_COMM_WORLD,&H));
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

static int set_normal_hamiltonian(PetscReal mu, PetscReal t_hopping, PetscReal sc_gap, PetscReal junction_potential) {
  // mu is effective potential after removing hbar**2 k_y**2 / (2m*)
  mu /= sc_gap;
  t_hopping /= sc_gap;
  
  for (int i=0; i < N_sites; ++i) {
    // on-site
    // electron
    PetscReal site_potential = junction_potential * exp(-pow(((double) i - (double) N_sites/2) / N_sites_JJ,2)) / sc_gap;
    // printf("i = %d, site_potential = %.2g\n", i, site_potential);
    PetscCall(MatSetValue(H,4*i,4*i, 2*t_hopping - mu + site_potential, INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+1,4*i+1, 2*t_hopping - mu + site_potential, INSERT_VALUES));
  
    // hole
    PetscCall(MatSetValue(H,4*i+2,4*i+2, -(2*t_hopping - mu + site_potential), INSERT_VALUES));
    PetscCall(MatSetValue(H,4*i+3,4*i+3, -(2*t_hopping - mu + site_potential), INSERT_VALUES));

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

static int set_pairing(PetscReal Phi) {
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

static int set_spin(PetscReal k_y) {
  // need to assemble matrix after call
  // assume that H is scaled with 1/|Δ|
  // H_Z = 0.5 g* μ_B * (B_x σ_x + B_y σ_y)
  // σ_x = [[0, 1], [1, 0]] , σ_y = [[0,-i], [i, 0]]
   // Rashba SOC gives onsite term α k_y σ_x / Δ
  
  for (PetscInt i=0; i < N_sites; ++i) {
    // onsite terms
    PetscScalar E_z = 0;
    PetscReal SOC_term_ky = 0;
    PetscReal SOC_term = 0;
    if (i > N_sites_leads-1 && i < N_sites_leads + N_sites_JJ) {
      // inside JJ
      E_z = 0.5 * gfactor * const_mu_B * (B_x - PETSC_i * B_y) / sc_gap;
      SOC_term_ky = alpha_rashba * k_y / sc_gap;
      SOC_term = alpha_rashba  / (2*spacing * sc_gap);
    }
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
    // Have Rashba SOC only in normal region
    if (i > N_sites_leads-1 && i < N_sites_leads + N_sites_JJ) {
      //electron
      PetscCall(MatSetValue(H,4*i  ,4*i+5,SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+1,4*i+4,-SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+4  ,4*i+1,-SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+5,4*i,+SOC_term,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*i+2,4*i+7,-SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+3,4*i+6,SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+6,4*i+3,+SOC_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*i+7,4*i+2,-SOC_term,INSERT_VALUES)); 
    }
  }

  
  return 0;
}


int main(int argc,char **argv)
{

  PetscMPIInt mpi_size;
  
  PetscFunctionBeginUser;
  PetscCall(SlepcInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &mpi_size));
  PetscCheck(mpi_size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n1-D Josephson junction with spin\n"));

  PetscReal mu = 10; // meV
  PetscReal JJ_potential = 5; // meV
  PetscReal JJ_length = 100;
  
  PetscInt N_evs = 20;
  
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-pot", &JJ_potential,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-alpha",&alpha_rashba,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-length",&JJ_length,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-B_y",&B_y,NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL,"-Nevs", &N_evs, NULL));
  // create_output_string_here
  //
  //
  
  mu *= 1e-3 * const_e;
  JJ_potential *= 1e-3 * const_e;
  alpha_rashba *= 1e-3 * const_e * 1e-9;
  JJ_length *= 1e-9;
  
  
  Mat            H;           /* BdG Hamiltonian */
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  PetscScalar    kr,ki;
  PetscReal m_eff = 0.036 * const_m_e;
  


  PetscReal k_F = 1/const_hbar * sqrt(2 * m_eff * mu);
  PetscReal v_F = const_hbar * k_F / m_eff;
  PetscReal xi_0 = const_hbar * v_F / (const_pi * sc_gap);
  
  PetscReal lambda_F = 2*const_pi / k_F;
  spacing = lambda_F / 10;

  PetscReal t_hopping = const_hbar*const_hbar / (2 * m_eff * spacing*spacing);
  PetscInt       i,its,nconv;
  

  char output_dir[200], output_file[300];
  FILE *file;


  

  
  printf("t / Δ = %.2g\n",  t_hopping / sc_gap);
  printf("λ_F = %.2g\n", lambda_F);
  printf("λ_F / a = %.2g\n", lambda_F / spacing);
  printf("ξ_0 = %.2g\n", xi_0);
  N_sites_leads = 2*xi_0 / spacing;
  N_sites_JJ = JJ_length / spacing;
  N_sites = 2*N_sites_leads + N_sites_JJ;
  printf("N_sites = %d, N_sites_JJ = %d\n", N_sites, N_sites_JJ);
  printf("L_electrode = %.2g\n", N_sites_leads * spacing);
  
  printf("ξ_0 / L_electrode = %.2g\n", xi_0 / (N_sites_leads * spacing));
  

  /* create output directory */
  time_t t = time(NULL);
  struct tm tm = *localtime(&t);
  snprintf(output_dir, sizeof(output_dir), "%d-%02d-%02d_%02d-%02d-%02d_mu=%.2gmeV_By=%.2gT_LJJ=%.2gnm_JJpotential=%.2gmeV_alpha=%.2gmeVnm", tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, mu / const_e * 1e3, B_y, JJ_length * 1e9, JJ_potential / const_e * 1e3, alpha_rashba / const_e * 1e12);
  mkdir(output_dir, 0777);

  snprintf(output_file, sizeof(output_file), "%s/output-spin.dat", output_dir);
  printf("output file: %s\n", output_file);
  file = fopen(output_file, "w");
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, H_{BdG}Φ = EΦ
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 
  allocate_matrix(&H, N_sites_leads, N_sites_JJ);
 
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
  PetscCall(EPSSetTolerances(eps, 1e-5, 1000));  
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "# k_y phi evs ...\n"));
  for (PetscReal k_y = 0; k_y < 1.2 * k_F; k_y += k_F / 50) {
    for (PetscReal Phi = -1.1*const_pi; Phi < 1.1*const_pi; Phi += 0.02 * const_pi) {
      printf("\n-------------------\nk_y / k_F = %.3g, φ = %.3g π\n", k_y / k_F, Phi / const_pi);

      set_normal_hamiltonian(mu - pow(k_y*const_hbar,2) / (2*m_eff), JJ_potential);
      set_pairing(Phi);
      set_spin(H, N_sites, spacing, sc_gap,
               -10,                  // g-factor
               0,                   // B_x
               B_y,                 // B_y
               k_y,                   // k_y
               alpha_rashba); // α
      
    
      PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
      // PetscCall(MatView(H, PETSC_VIEWER_STDOUT_SELF)); 
    
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
