#include <slepceps.h>
#include <sys/stat.h> // for mkdir
#include <petscmat.h>
#include <time.h>
static char help[] = "Bogoliuboev de-Gennes eigenvalue solver for SNS junctions with arbitrary disorder.\n\n";

static PetscReal const_hbar = 1.0545718176461565e-34;
static PetscReal const_e = 1.602176634e-19;
static PetscReal const_m_e = 9.1093837015e-31;
static PetscReal const_pi = 3.141592;



static  PetscInt N_sites_x, N_sites_y, N_sites_JJ, N_sites_leads;
static Mat H;

static int allocate_matrix() {
  PetscCall(MatCreate(PETSC_COMM_WORLD,&H));
  PetscCall(MatSetSizes(H,PETSC_DECIDE,PETSC_DECIDE,4*N_sites_x * N_sites_y,4*N_sites_x*N_sites_y));
  PetscCall(MatSetFromOptions(H));
  PetscCall(MatSetUp(H));
  // Fill each block (on-site terms + hoppings in x-direction)
  for (PetscInt iy = 0; iy < N_sites_y; ++iy) {
    for (PetscInt ix=0; ix < N_sites_x; ++ix) {
      PetscInt block_start = 4 * N_sites_x * iy;
      // on-site
      // electron
      PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix+1, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix+1, 0, INSERT_VALUES));
      
      // Hole
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+2, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+3, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+2, 0, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+3, 0, INSERT_VALUES));
      // SC gap parameter
   
      if (ix < N_sites_leads || ix > N_sites_leads + N_sites_JJ-1) {
        PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix+2, 0, INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix, 0, INSERT_VALUES));
        
        PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix+3, 0, INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+1, 0, INSERT_VALUES));
      }
      // Hoppings

      if (ix<N_sites_x-1) {
        //electron
        PetscCall(MatSetValue(H,block_start + 4*ix  ,block_start + 4*ix+4,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+4  ,block_start + 4*ix,0,INSERT_VALUES));

        PetscCall(MatSetValue(H,block_start + 4*ix+1  ,block_start + 4*ix+5,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+5  ,block_start + 4*ix+1,0,INSERT_VALUES));

        // SOC
        PetscCall(MatSetValue(H,block_start + 4*ix  ,block_start + 4*ix+5,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+5  ,block_start + 4*ix,0,INSERT_VALUES));

        PetscCall(MatSetValue(H,block_start + 4*ix+1  ,block_start + 4*ix+4,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+4  ,block_start + 4*ix+1,0,INSERT_VALUES));
        //hole
        PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+6,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+6,block_start + 4*ix+2,0,INSERT_VALUES));

        PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+7,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+7,block_start + 4*ix+3,0,INSERT_VALUES));

        // SOC
        PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+7,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+7,block_start + 4*ix+2,0,INSERT_VALUES));

        PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+6,0,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+6,block_start + 4*ix+3,0,INSERT_VALUES));
      }
    }
  }

  // Fill diagonal blocks (hoppings in y-direction)
  for (PetscInt iy = 0; iy < N_sites_y-1; ++iy) {
    for (PetscInt ix=0; ix < N_sites_x; ++ix) {
      //electron
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix,4*N_sites_x*(iy+1)+4*ix ,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+1,4*N_sites_x*(iy+1)+4*ix+1 ,0,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix,4*N_sites_x*iy+4*ix ,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+1,4*N_sites_x*iy+4*ix+1 ,0,INSERT_VALUES));
      // electron-SOC
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix,4*N_sites_x*(iy+1)+4*ix+1 ,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+1,4*N_sites_x*(iy+1)+4*ix ,0,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+1,4*N_sites_x*iy+4*ix ,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix,4*N_sites_x*iy+4*ix+1 ,0,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+2,4*N_sites_x*(iy+1)+4*ix+2 ,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+3,4*N_sites_x*(iy+1)+4*ix+3 ,0,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+2,4*N_sites_x*iy+4*ix+2,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+3,4*N_sites_x*iy+4*ix+3 ,0,INSERT_VALUES));

      //hole-SOC
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+2,4*N_sites_x*(iy+1)+4*ix+3 ,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+3,4*N_sites_x*(iy+1)+4*ix+2 ,0,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+3,4*N_sites_x*iy+4*ix+2,0,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+2,4*N_sites_x*iy+4*ix+3 ,0,INSERT_VALUES));

    }
  }
  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  return 0;
}

static int set_normal_hamiltonian(PetscReal sc_gap,  PetscReal t_hopping, PetscReal mu, PetscReal disorder_potential, unsigned int seed) {
  // normalize hamiltonian with sc_gap
  mu /= sc_gap;
  t_hopping /= sc_gap;
  srand(seed);
  for (int iy = 0; iy < N_sites_y; ++iy) {
    PetscInt block_start = 4 * N_sites_x * iy;
    for (int ix=0; ix < N_sites_x; ++ix) {
      int neighbors = 4;
      if (ix == 0 || ix == N_sites_x - 1)
        neighbors -= 1;
      if (iy == 0 || iy == N_sites_y - 1)
        neighbors -= 1;
      // on-site
      PetscReal site_potential = ((((float) rand()) / RAND_MAX) - 0.5) * disorder_potential / sc_gap;
      /* // printf("i = %d, site_potential = %.2g\n", i, site_potential); */

      // electron
      PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix, neighbors*t_hopping - mu + site_potential, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix+1, neighbors*t_hopping - mu + site_potential, INSERT_VALUES));
      
      // Hole
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+2,  -(neighbors*t_hopping - mu + site_potential), INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+3,  -(neighbors*t_hopping - mu + site_potential), INSERT_VALUES));
      
      // hoppings
    
      if (ix<N_sites_x-1) {
        //electron
        PetscCall(MatSetValue(H,block_start + 4*ix  ,block_start + 4*ix+4,-t_hopping,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+4  ,block_start + 4*ix,-t_hopping,INSERT_VALUES));

        PetscCall(MatSetValue(H,block_start + 4*ix+1  ,block_start + 4*ix+5,-t_hopping,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+5  ,block_start + 4*ix+1,-t_hopping,INSERT_VALUES));
        //hole
        PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+6,t_hopping,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+6,block_start + 4*ix+2,t_hopping,INSERT_VALUES));

        PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+7,t_hopping,INSERT_VALUES));
        PetscCall(MatSetValue(H,block_start + 4*ix+7,block_start + 4*ix+3,t_hopping,INSERT_VALUES));
      }
    }
  }
  // Fill diagonal blocks (hoppings in y-direction)
  for (PetscInt iy = 0; iy < N_sites_y-1; ++iy) {
    for (PetscInt ix=0; ix < N_sites_x; ++ix) {
      //electron
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix,4*N_sites_x*(iy+1)+4*ix ,-t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+1,4*N_sites_x*(iy+1)+4*ix+1 ,-t_hopping,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix,4*N_sites_x*iy+4*ix ,-t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+1,4*N_sites_x*iy+4*ix+1 ,-t_hopping,INSERT_VALUES));
      //hole
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+2,4*N_sites_x*(iy+1)+4*ix+2 ,t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+3,4*N_sites_x*(iy+1)+4*ix+3 ,t_hopping,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+2,4*N_sites_x*iy+4*ix+2,t_hopping,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+3,4*N_sites_x*iy+4*ix+3 ,t_hopping,INSERT_VALUES));
      
    }
  }
  return 0;
}

static int set_pairing(PetscReal Phi) {
  // need to assemble matrix after call
  // assume that H is scaled with 1/|Δ|
  for (int iy = 0; iy < N_sites_y; ++iy) {
    PetscInt block_start = 4*N_sites_x * iy;
    // left lead
    for (int ix=0; ix < N_sites_leads; ++ix) {
      PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix+2, 1, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix, 1, INSERT_VALUES));
        
      PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix+3, 1, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+1, 1, INSERT_VALUES));
    }

    // right lead
    PetscScalar gap = PetscExpComplex(PETSC_i * Phi);
    for (int ix =N_sites_JJ + N_sites_leads; ix < N_sites_x; ++ix) {
      PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix+2, gap, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix, PetscConjComplex(gap), INSERT_VALUES));
        
      PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix+3, gap, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+1, PetscConjComplex(gap), INSERT_VALUES));
    }
  }
  return 0;
}

static int set_zeeman(PetscReal EZX, PetscReal EZY, PetscBool zeeman_in_leads) {
  for (int iy = 0; iy < N_sites_y; ++iy) {
    PetscInt block_start = 4 * N_sites_x * iy;
    for (int ix=0; ix < N_sites_x; ++ix) {
      if (!zeeman_in_leads && (ix < N_sites_leads || ix >= N_sites_leads+N_sites_JJ))
        continue;
      // electron
      PetscCall(MatSetValue(H,block_start + 4*ix,block_start + 4*ix+1, EZX -  PETSC_i*EZY, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+1,block_start + 4*ix, EZX +  PETSC_i*EZY, INSERT_VALUES));
      
      // Hole
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+3, EZX -  PETSC_i*EZY, INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+2, EZX +  PETSC_i*EZY, INSERT_VALUES));
    }
  }
  return 0;
}


static int set_rasbha(PetscReal alpha, PetscReal sc_gap,  PetscReal spacing) {
  PetscReal soc_term = alpha / (2*spacing * sc_gap);
  printf("soc_term = %g\n", soc_term);
  for (int iy = 0; iy < N_sites_y; ++iy) {
    PetscInt block_start = 4 * N_sites_x * iy;
    for (int ix=0; ix < N_sites_x-1; ++ix) {
      // electron-SOC
      PetscCall(MatSetValue(H,block_start + 4*ix  ,block_start + 4*ix+5,soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+5  ,block_start + 4*ix,soc_term,INSERT_VALUES));

      PetscCall(MatSetValue(H,block_start + 4*ix+1  ,block_start + 4*ix+4,-soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+4  ,block_start + 4*ix+1,-soc_term,INSERT_VALUES));

      // hole-SOC
      PetscCall(MatSetValue(H,block_start + 4*ix+2,block_start + 4*ix+7,-soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+7,block_start + 4*ix+2,-soc_term,INSERT_VALUES));

      PetscCall(MatSetValue(H,block_start + 4*ix+3,block_start + 4*ix+6,soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,block_start + 4*ix+6,block_start + 4*ix+3,soc_term,INSERT_VALUES));
    }
  }

  // Fill diagonal blocks (hoppings in y-direction)
  for (PetscInt iy = 0; iy < N_sites_y-1; ++iy) {
    for (PetscInt ix=0; ix < N_sites_x; ++ix) {
       // electron-SOC
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix,4*N_sites_x*(iy+1)+4*ix+1 ,-PETSC_i*soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+1,4*N_sites_x*(iy+1)+4*ix ,-PETSC_i*soc_term,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+1,4*N_sites_x*iy+4*ix ,PETSC_i*soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix,4*N_sites_x*iy+4*ix+1 ,PETSC_i*soc_term,INSERT_VALUES));
      //hole-SOC
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+2,4*N_sites_x*(iy+1)+4*ix+3 ,PETSC_i*soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*iy+4*ix+3,4*N_sites_x*(iy+1)+4*ix+2 ,PETSC_i*soc_term,INSERT_VALUES));

      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+3,4*N_sites_x*iy+4*ix+2,-PETSC_i*soc_term,INSERT_VALUES));
      PetscCall(MatSetValue(H,4*N_sites_x*(iy+1)+4*ix+2,4*N_sites_x*iy+4*ix+3 ,-PETSC_i*soc_term,INSERT_VALUES));
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

  struct timespec  t_seed;
  clock_gettime(CLOCK_REALTIME, &t_seed);
  uint seed_pairing = (uint) t_seed.tv_nsec;
  printf("pairing seed: %u\n", seed_pairing);

  clock_gettime(CLOCK_REALTIME, &t_seed);
  uint seed_potential = (uint) t_seed.tv_nsec;
  printf("potential seed: %u\n", seed_potential);
  
  
  char filename[1024] = "output.dat";
  PetscReal mu = 10; // chemical potential (meV)
  PetscReal disorder_potential = 0; // relative to chemical potential mu
  PetscReal EZX = 0;
  PetscReal EZY = 0;
  PetscReal alpha = 0; // (meV nm)
  PetscReal spectrum_range = 4; // calculate spectrum up to N_ABS_bound_states * spectrum_range
  //PetscReal pairing_density = 1;
  PetscBool zeeman_in_leads = 0;
  
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-dis", &disorder_potential,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-leadlength",&N_sites_leads,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-JJlength",&N_sites_JJ,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-JJwidth",&N_sites_y,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&mu,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-EZX",&EZX,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-EZY",&EZY,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-alpha",&alpha,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-spectrum",&spectrum_range,NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zeeman_in_leads", &zeeman_in_leads, NULL));
  //PetscCall(PetscOptionsGetReal(NULL,NULL,"-pairing_density",&pairing_density,NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-output", filename, sizeof(filename), NULL));

  
  mu *= 1e-3 * const_e;
  alpha *= 1e-3 * const_e * 1e-9;
  
  double m_eff = 0.036 * const_m_e;
  double sc_gap = 200e-6*const_e;
  double k_F = sqrt(2 * m_eff * mu) / const_hbar;
  double v_F = const_hbar * k_F / m_eff;
  double xi_0 = const_hbar * v_F / (const_pi * sc_gap);
  double lambda_F = 2*const_pi / k_F;
  double spacing = lambda_F / 10;
  // need to convert to double, as hbar**2 is zero in single precision math
  double t_hopping = ((double) const_hbar)*const_hbar / (2 * m_eff * spacing*spacing);

  disorder_potential *= mu;
  N_sites_x = 2*N_sites_leads + N_sites_JJ;

  printf("mu = %g\n", mu);
  printf("L_electrode = %.2g\n", N_sites_leads * spacing);
  printf("ξ_0 / L_electrode = %.2g\n", xi_0 / (N_sites_leads * spacing));
  printf("sc_gap = %g, m_eff = %g, hbar = %g\n", sc_gap, m_eff, const_hbar);
  printf("N_sites_x = %d, N_sites_y = %d, N_sites_JJ = %d\n", N_sites_x, N_sites_y, N_sites_JJ);
  printf("t / Δ = %.2g\n",  t_hopping / sc_gap);
  printf("λ_F = %.2g\n", lambda_F);
  printf("λ_F / a = %.2g\n", lambda_F / spacing);
  printf("ξ_0 = %.2g\n", xi_0);
  printf("Zeeman in leads: %d\n", zeeman_in_leads);
  //  printf("pairing density = %g\n", pairing_density);
  EPS            eps;         /* eigenproblem solver context */
  ST             st;          /* spectral transformation context */
  PetscScalar    kr,ki;
  PetscInt N_evs = (int ) (2 * spectrum_range *(0.4 * N_sites_y + 4));
  PetscInt       i,its,nconv;
  FILE *file;
  PetscCall(PetscFOpen(PETSC_COMM_WORLD, filename, "a", &file)); // append to file, if it already exists
  // setvbuf(file, NULL, _IONBF, 0); // always flush output data
  
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Compute the operator matrix that defines the eigensystem, H_{BdG}Φ = EΦ
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
 

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
  int ncv = 1.5 * N_evs + 20;
  printf("requested eigenvalues: %d, subspace dimension: %d\n", N_evs, ncv);
  PetscCall(EPSSetDimensions(eps, N_evs, ncv, PETSC_DECIDE));
  PetscCall(EPSSetTarget(eps, 0));

  PetscCall(EPSSetTolerances(eps, 1e-10, 1000));

  // force the computation of the true residual
  //  PetscCall(EPSSetTrueResidual(eps, 0));
  PetscCall(EPSSetFromOptions(eps));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the eigensystem
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "# phi/π evs ...\n"));
  allocate_matrix();
  CHKMEMQ;

  set_normal_hamiltonian(sc_gap, t_hopping, mu, disorder_potential, seed_potential);
  CHKMEMQ;
  set_zeeman(EZX, EZY, zeeman_in_leads);
  CHKMEMQ;
  set_rasbha(alpha, sc_gap, spacing);
    
  for (double Phi = -1.04*const_pi; Phi <= 1.041*const_pi; Phi += 0.01 * const_pi) {
    printf("\n-------------------\ndisorder / mu = %.3g, φ = %.3g π\n", disorder_potential / mu, Phi / const_pi);

    struct timespec  t_start, t_end;
    clock_gettime(CLOCK_REALTIME, &t_start);
    set_pairing(Phi);
      
    
    PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
    //    PetscCall(MatView(H, PETSC_VIEWER_STDOUT_SELF));
    //exit(1);
    
    PetscCall(EPSSetOperators(eps,H,NULL));
    PetscCall(EPSSolve(eps));
    clock_gettime(CLOCK_REALTIME, &t_end);
    float duration = t_end.tv_sec - t_start.tv_sec + 1e-9 * (t_end.tv_nsec - t_start.tv_nsec);
    printf("Duration for EPSSolve: %g s\n\n" ,duration);

    PetscCall(EPSGetIterationNumber(eps,&its));
    printf(" Number of iterations of the method: %" PetscInt_FMT "\n",its);
      
    PetscCall(EPSGetConverged(eps,&nconv));
    printf(" Number of converged eigenpairs: %" PetscInt_FMT "\n\n",nconv);

    PetscCheck(nconv >= N_evs, PETSC_COMM_WORLD, 1, "did not converge");
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "%.10g\t", Phi / const_pi));

    for (i = 0; i < N_evs; ++i) {
      PetscCall(EPSGetEigenvalue(eps, i, &kr, &ki));
      PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "%.10g\t", (double ) kr));
    }
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "\n"));
    PetscCall(PetscFFlush(file));
  }
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, file, "\n"));
  PetscCall(PetscFFlush(file));
  
  return 0;
}
