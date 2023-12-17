# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>

# define OUT 0

// Include MPI header
# include "omp.h"

// Function definitions
int main ( int argc, char *argv[] );
double boundary_condition ( double x, double time );
double initial_condition ( double x, double time );
double source ( double x, double time );
void runSolver( int n);



/*-------------------------------------------------------------
  Purpose: Compute number of primes from 1 to N with naive way
 -------------------------------------------------------------*/
// This function is fully implemented for you!!!!!!
// usage: mpirun -n 4 heat1d N
// N    : Number of nodes per processor
int main ( int argc, char *argv[] ){
  int rank, size;
  double wtime;

  // get number of nodes per processor
  int N = strtol(argv[1], NULL, 10);

  omp_set_num_threads(strtol(argv[2],NULL,10));

  // Solve and update the solution in time
  runSolver(N);

  return 0;
}

/*-------------------------------------------------------------
  Purpose: computes the solution of the heat equation.
 -------------------------------------------------------------*/
void runSolver( int n){
  // CFL Condition is fixed
  double cfl = 0.5; 
  // Domain boundaries are fixed
  double x_min=0.0, x_max=1.0;
  // Diffusion coefficient is fixed
  double k   = 0.002;
  // Start time and end time are fixed
  double tstart = 0.0, tend = 10.0;  

  // Storage for node coordinates, solution field and next time level values
  double *x, *q, *qn;
  // Set the x coordinates of the n nodes padded with +2 ghost nodes. 
  x  = ( double*)malloc((n+2)*sizeof(double));
  q  = ( double*)malloc((n+2)*sizeof(double));
  qn = ( double*)malloc((n+2)*sizeof(double));

  // Write solution field to text file if size==1 only
  FILE *qfile, *xfile, *q_tstep;

  // uniform grid spacing
  double dx = ( x_max - x_min ) / ( double ) ( n - 1 );

  // Set time step size dt <= CFL*h^2/k
  // and then modify dt to get integer number of steps for tend
  double dt  = cfl*dx*dx/k; 
  int Nsteps = ceil(( tend - tstart )/dt);
  dt =  ( tend - tstart )/(( double )(Nsteps)); 

  double time, time_new, wtime;
  
  // find the coordinates for uniform spacing 
    for ( int i = 0; i <= n + 1; i++ )
    {
        // COMPLETE THIS PART
        // any position of nodes= starting position of processor + number of intervals*interval size
        x[i] = x_min+(i-1)*dx;
    }

    // Set the values of q at the initial time.
    time = tstart; q[0] = 0.0; q[n+1] = 0.0;

    wtime=omp_get_wtime();
    int i;
    #pragma omp parallel for default(none) private(i) shared(q,x,time,n) 
      for (int i = 1; i <= n; i++ ){
        q[i] = initial_condition(x[i],time);
      }


  // Compute the values of H at the next time, based on current data.
 
  for ( int step = 1; step <= Nsteps; step++ ){

    time_new = time + step*dt; 


    // UPDATE the solution based on central differantiation.
    // qn[i] = q[i] + dt*rhs(q,t)
    // For OpenMP make this loop parallel also

    

     #pragma omp parallel for default(none) private(i) shared(dt,qn, q, k,dx,x,time,n)
    for ( int i = 1; i <= n; i++ ){
      // COMPLETE THIS PART

      qn[i]=q[i]+dt*((k/(dx*dx))*(q[i-1]+2*q[i]+q[i+1])+source(x[i],time));
    }
   

  
    // q at the extreme left and right boundaries was incorrectly computed
    // using the differential equation.  
    // Replace that calculation by the boundary conditions.
    // global left endpoint 
    
    qn[1] = boundary_condition ( x[1], time_new );
    
    // global right endpoint 
    
    qn[n] = boundary_condition ( x[n], time_new );
    

  // Update time and field.
    time = time_new;
    // For OpenMP make this loop parallel also

    #pragma omp parallel for default(none) private(i) shared(q,n,qn)
    for ( int i = 1; i <= n; i++ ){
      q[i] = qn[i];
    }
  }
  
  wtime=omp_get_wtime()-wtime;

  printf ( "  Wall clock elapsed seconds = %f\n", wtime );      


  free(q); free(qn); free(x);

  return;
}
/*-----------------------------------------------------------*/
double boundary_condition ( double x, double time ){
  double value;

  // Left condition:
  if ( x < 0.5 ){
    value = 100.0 + 10.0 * sin ( time );
  }else{
    value = 75.0;
  }
  return value;
}
/*-----------------------------------------------------------*/
double initial_condition ( double x, double time ){
  double value;
  value = 95.0;

  return value;
}
/*-----------------------------------------------------------*/
double source ( double x, double time ){
  double value;

  value = 0.0;

  return value;
}