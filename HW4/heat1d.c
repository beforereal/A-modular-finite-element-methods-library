# include <math.h>
# include <stdlib.h>
# include <stdio.h>
# include <time.h>
#include <mpi.h>

# define OUT 0

// Include MPI header
# include "mpi.h"

// Function definitions
int main ( int argc, char *argv[] );
double boundary_condition ( double x, double time );
double initial_condition ( double x, double time );
double source ( double x, double time );
void runSolver( int n, int rank, int size );



/*-------------------------------------------------------------
  Purpose: Compute number of primes from 1 to N with naive way
 -------------------------------------------------------------*/
// This function is fully implemented for you!!!!!!
// usage: mpirun -n 4 heat1d N
// N    : Number of nodes per processor
int main ( int argc, char *argv[] ){
  int rank, size;
  double wtime;

  // Initialize MPI, get size and rank
  MPI_Init ( &argc, &argv );
  MPI_Comm_rank ( MPI_COMM_WORLD, &rank );
  MPI_Comm_size ( MPI_COMM_WORLD, &size );

  // get number of nodes per processor
  int N = strtol(argv[1], NULL, 10);

  // Solve and update the solution in time
  runSolver(N, rank, size);

  // Terminate MPI.
  MPI_Finalize ( );
  // Terminate.
  return 0;
}

/*-------------------------------------------------------------
  Purpose: computes the solution of the heat equation.
 -------------------------------------------------------------*/
void runSolver( int n, int rank, int size ){
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
  FILE *qfile, *xfile;

  // uniform grid spacing
  double dx = ( x_max - x_min ) / ( double ) ( size * n - 1 );

  // Set time step size dt <= CFL*h^2/k
  // and then modify dt to get integer number of steps for tend
  double dt  = cfl*dx*dx/k; 
  int Nsteps = ceil(( tend - tstart )/dt);
  dt =  ( tend - tstart )/(( double )(Nsteps)); 

  int tag;
  MPI_Status status;
  double time, time_new, wtime;  

  // starting position of each position
  int starting_node= rank*n;
  double starting_position= x_min+starting_node*dx;

  // find the coordinates for uniform spacing 
  for ( int i = 0; i <= n + 1; i++ ){
    // COMPLETE THIS PART
    // any position of nodes= starting position of processor + number of intervals*interval size
    x[i] = starting_position+(i-1)*dx;
  }

  // Set the values of q at the initial time.
  time = tstart; q[0] = 0.0; q[n+1] = 0.0;
  for (int i = 1; i <= n; i++ ){
    q[i] = initial_condition(x[i],time);
  }


  // In single processor mode
  if (size == 1 && OUT==1){
    // write out the x coordinates for display.
    xfile = fopen ( "x_data.txt", "w" );
    for (int i = 1; i<(n+1); i++ ){
      fprintf ( xfile, "  %f", x[i] );
    }
    fprintf ( xfile, "\n" );
    fclose ( xfile );
    // write out the initial solution for display.
    qfile = fopen ( "q_data.txt", "w" );
    for ( int i = 1; i <= n; i++ ){
      fprintf ( qfile, "  %f", q[i] );
    }
    fprintf ( qfile, "\n" );
  }


 // Record the starting time.
  wtime = MPI_Wtime();

  // Compute the values of H at the next time, based on current data.
  for ( int step = 1; step <= Nsteps; step++ ){

    time_new = time + step*dt; 

    // Perform point to point communications here!!!!
    // COMPLETE THIS PART

    //Mr Karakuş has informed us to use a common tag for MPI communication in the lectures

    int tag_forward=25;
    int tag_backward=15;
    //tags are defined for sending operations, not recieving (nomenclature wise)

    //Send data to next processor
    if (rank<size-1)
    {
      MPI_Send(&q[n],1,MPI_DOUBLE,rank+1,tag_forward,MPI_COMM_WORLD);
    }

    //Send data to prev processor
    if(rank>0)
    {
      MPI_Send(&q[1],1,MPI_DOUBLE,rank-1,tag_backward,MPI_COMM_WORLD);
    }

    //Recieve data from next processor
    if(rank<size-1)
    {
      MPI_Recv(&q[n+1],1,MPI_DOUBLE,rank+1,tag_backward,MPI_COMM_WORLD, &status);
    }

    //Rcieve data from previous processor
    if (rank>0)
    {
      MPI_Recv(&q[0],1,MPI_DOUBLE,rank-1,tag_forward,MPI_COMM_WORLD,&status);
    }

    // UPDATE the solution based on central differantiation.
    // qn[i] = q[i] + dt*rhs(q,t)
    // For OpenMP make this loop parallel also
    for ( int i = 1; i <= n; i++ ){
      // COMPLETE THIS PART

      qn[i]=q[i]+dt*((k/(dx*dx))*(q[i-1]+2*q[i]+q[i+1])+source(x[i],time));


    }

  
    // q at the extreme left and right boundaries was incorrectly computed
    // using the differential equation.  
    // Replace that calculation by the boundary conditions.
    // global left endpoint 
    if (rank==0){
      qn[1] = boundary_condition ( x[1], time_new );
    }
    // global right endpoint 
    if (rank == size - 1 ){
      qn[n] = boundary_condition ( x[n], time_new );
    }

  // Update time and field.
    time = time_new;
    // For OpenMP make this loop parallel also
    for ( int i = 1; i <= n; i++ ){
      q[i] = qn[i];
    }

  // In single processor mode, add current solution data to output file.
    if (size == 1 && OUT==1){
      for ( int i = 1; i <= n; i++ ){
        fprintf ( qfile, "  %f", q[i] );
      }
      fprintf ( qfile, "\n" );
    }

  }

  
  // Record the final time.
  // if (rank == 0 ){
  wtime = MPI_Wtime( )-wtime;

  // Add local number of primes
  double global_time = 0.0; 
  MPI_Reduce( &wtime, &global_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

 if(rank==0)
   printf ( "  Wall clock elapsed seconds = %f\n", global_time );      


  if( size == 1 && OUT==1)
    fclose ( qfile ); 

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
