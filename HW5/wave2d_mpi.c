# include <stdio.h>
# include <stdlib.h>
# include <math.h>
# include <time.h>
# include <string.h>
# include <mpi.h>
#define BUFSIZE 512


// Function definitions
/******************************************************************************/
int main ( int argc, char *argv[] );
double exactSoln( double c, double x, double y, double t );
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny);
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *data); 
double readInputFile(char *fileName, char* tag); 

// Solver Info
/******************************************************************************
  Purpose:
    wave2d solves the wave equation in parallel using MPI.
  Discussion:
    Discretize the equation for u(x,t):
      d^2 u/dt^2  =  c^2 * (d^2 u/dx^2 + d^2 u/dy^2)  
      for 0 < x < 1, 0 < y < 1, t>0
    with boundary conditions and Initial conditions obtained from the exact solutions:
      u(x,y, t) = sin ( 2 * pi * ( x - c * t ) )
   Usage: serial -> ./wave input.dat  parallel> mpirun -np 4 ./wave input.dat 
******************************************************************************/

int main ( int argc, char *argv[] ){
  
  // Read input file for solution parameters
  double tstart = readInputFile(argv[1], "TSART"); // Start time
  double tend   = readInputFile(argv[1], "TEND");  // End time
  double dt     = readInputFile(argv[1], "DT");    // Time step size

  // Global node number in x and y
  int NX        = (int) readInputFile(argv[1], "NX"); // Global node numbers in x direction
  int NY        = (int) readInputFile(argv[1], "NY"); // Global node numbers in y direction

  double xmax = readInputFile(argv[1], "XMAX"); // domain boundaries
  double xmin = readInputFile(argv[1], "XMIN"); // domain boundaries
  double ymax = readInputFile(argv[1], "YMAX"); // domain boundaries
  double ymin = readInputFile(argv[1], "YMIN"); // domain boundaries
  double c = readInputFile(argv[1], "CONSTANT_C");

  double *qn, *q0, *q1;               // Solution field at t+dt, t and t-dt
  static int frame=0; 

  
  // DOMAIN DECOMPOSITION
  // For serial implementation nx = NX and ny = NY; 
  int nx = NX;      // local number of nodes in x direction
  int ny = NY;      // local number of nodes in y direction

  // ALLOCATE MEMORY for COORDINATES (x, y) and compute them
  double *x = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  double *y = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // find uniform spacing in x and y directions
  double hx = (xmax - xmin)/(NX-1.0); 
  double hy = (ymax - ymin)/(NY-1.0); 
  // Compute coordinates of the nodes
  for(int j=0; j < ny; ++j){
    for(int i=0; i < nx;++i){
      double xn = xmin + i*hx; 
      double yn = ymin + j*hy;  
      
      x[i+j*nx] = xn; 
      y[i+j*nx] = yn; 
    }
  }


  // ALLOCATE MEMORY for SOLUTION and its HISTORY
  // Solution at time (t+dt)
  qn = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // Solution at time (t)
  q0 = ( double * ) malloc ( nx*ny * sizeof ( double ) );
  // Solution at time t-dt
  q1 = ( double * ) malloc ( nx*ny * sizeof ( double ) );

  // USE EXACT SOLUTION TO FILL HISTORY
   for(int i=0; i<nx; i++){
      for(int j=1; j<ny; j++){
      const double xn = x[i+ j*nx]; 
      const double yn = y[i+ j*nx]; 
      // Exact solutions at history tstart and tstart+dt
      q0[i + j*nx] = exactSoln(c, xn, yn, tstart + dt);  
      q1[i + j*nx] = exactSoln(c, xn, yn, tstart);  
    }
  }

 
  // Write the initial solution 
  {
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d.csv", frame++);
    solverPlot(fname, x, y, nx, ny, q1);
  }

// RUN SOLVER 
  int Noutput = 10000; 
  int Nsteps=(tend - tstart)/dt;     // Assume  dt divides (tend- tstart)
  double alphax2 = pow((c*dt/hx),2); 
  double alphay2 = pow((c*dt/hy),2);
  
  // We already have 2 steps computed with exact solution
  double time = dt; 

  //Initialize mpi, get size, rank
  int rank, size;
  double wtime;
 
  MPI_Init ( &argc, &argv);
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
  MPI_Comm_size (MPI_COMM_WORLD, &size);
 wtime = MPI_Wtime(); // Start timing
  // for every time step
  for(int tstep = 2; tstep<=Nsteps+1; ++tstep){
    // increase  time
    time = tstart + tstep*dt; 

    // Apply Boundary Conditions i.e. at i, j = 0, i,j = nx-1, ny-1
    applyBC(q0, x, y, c, time, nx, ny); 



  if((ny-1)%size!=0){
    printf("Number of nodes on y direction cannot be divided to number of rank");
    MPI_Finalize();
    return 0;
  }
  //Number of nodes (rows) in y direction per processor
  int N = (ny-1)/size;
  int tag;
  MPI_Status status;
     // Update solution using second order central differencing in time and space
    int j=0;
    for(int i=1; i<nx-1; i++){ // exclude left right boundaries
      if (rank==0){
        j=1; // exclude top boundary
      }
      else{
        j=rank*N; //start from the first row of corresponding rank
      }

      for(j; j<(rank+1)*N ; j++){ //iterate for every row of corresponding rank
        
        if (rank+1==size & j==(rank+1)*N) {
          break; //exclude bottom boundary (last row of last rank)
        }

        const int n0   = i + j*nx; 
        const int nim1 = i - 1 + j*nx; // node i-1,j sağ
        const int nip1 = i + 1 + j*nx; // node i+1,j sol
        const int njm1 = i + (j-1)*nx; // node i, j-1 üst 
        const int njp1 = i + (j+1)*nx; // node i, j+1 alt


        //if j is equal the bottom edge of rank, we need communication to compute
        //q0 send/recieve to/from bottom/upper rank (burada upper bottom'a yolluyor)
        if (j==(rank+1)*N-1 & rank+1!=size){
          tag=0;//forward send tag
          MPI_Send (&q0[n0],1,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD);
        }
        if (j==(rank)*N & rank!=0){
          tag=0;
          MPI_Recv(&q0[njm1],1,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD,&status);
        }
        //q0 send/recieve to/from upper/bottom rank (burada bottom upper'a yolluyor)
        if (j==(rank)*N & rank!=0){
          tag=1;
          MPI_Send (&q0[n0],1,MPI_DOUBLE,rank-1,tag,MPI_COMM_WORLD);
        }
        if (j==(rank+1)*N-1 & rank+1!=size){
          tag=1;
          MPI_Recv (&q0[njp1],1,MPI_DOUBLE,rank+1,tag,MPI_COMM_WORLD,&status);
        }


        // update solution         
        qn[n0] = 2.0*q0[n0] - q1[n0] + alphax2*(q0[nip1]- 2.0*q0[n0] + q0[nim1])
                                     + alphay2*(q0[njp1] -2.0*q0[n0] + q0[njm1]);
      }
    }
    // Update history q1 = q0; q0 = qn, except the boundaries
    for(int i=1; i<nx-1; i++){
      for(int j=1; j<ny-1; j++){
        q1[i + j*nx] = q0[i + j*nx]; 
        q0[i + j*nx] = qn[i + j*nx]; 
        
      }
    }
    //*****************************************
    int master=0;
    
  /*
  	for (int current_rank=1; current_rank<size; ++current_rank){
      int tag2= current_rank;
      if(current_rank!=master &current_rank==rank){
          MPI_Send(&q0[current_rank*N*nx],N*nx,MPI_DOUBLE,master,tag2,MPI_COMM_WORLD);
      }
      if(rank==master){
          MPI_Recv(&q0[current_rank*nx*N],N*nx,MPI_DOUBLE,current_rank,tag2,MPI_COMM_WORLD,&status);
      }
    }
    
    */
    //*****************************************
   /*
    // Dampout a csv file for postprocessing
    if(tstep%Noutput == 0 & rank==master){ 
      char fname[BUFSIZ];
      sprintf(fname, "test_%04d.csv", frame++);
      solverPlot(fname, x, y, nx, ny, q0);
    }
    */
  }
  MPI_Finalize();
  
  // Stop timing and calculate the elapsed time
    wtime = MPI_Wtime() - wtime;
    printf("The time taken for the run %.4f seconds\n", wtime);
  

  // Compute Linf norm of error at tend
    double linf = 0.0; 
    for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
         double xn = x[i+ j*nx]; 
         double yn = y[i+ j*nx]; 
         // solution and the exact one
         double qn = q0[i+ j*nx]; 
         double qe = exactSoln(c, xn, yn, time);  
         linf  = fabs(qn-qe)>linf ? fabs(qn -qe):linf; 
      }
    }

    printf("Infinity norm of the error: %.4e %.8e \n", linf, time);

  return 0;
}




/***************************************************************************************/
double exactSoln( double c, double x, double y, double t){
  const double pi = 3.141592653589793;
  double value = sin( 2.0*pi*( x - c*t));
  return value;
}

/***************************************************************************************/
void applyBC(double *data,  double *x, double *y, double c, double time, int nx, int ny){

  // Apply Boundary Conditions
  double xn, yn; 

  for(int j=0; j<ny;++j){ // left right boundaries i.e. i=0 and i=nx-1
    xn = x[0 + j*nx]; 
    yn = y[0 + j*nx];    
    data[0 + j*nx] = exactSoln(c, xn, yn, time); 

    xn = x[nx-1 + j*nx]; 
    yn = y[nx-1 + j*nx];    
    data[nx-1 + j*nx] = exactSoln(c, xn, yn, time); 
  }

  
  for(int i=0; i< nx; ++i){ // top and  bottom boundaries i.e. j=0 and j=ny-1
    xn = x[i+ 0*nx]; 
    yn = y[i+ 0*nx]; 
    data[i + 0*nx] = exactSoln(c, xn, yn, time); 

    xn = x[i+ (ny-1)*nx]; 
    yn = y[i+ (ny-1)*nx];       
    data[i +  (ny-1)*nx] = exactSoln(c, xn, yn, time); 
  }
}

/* ************************************************************************** */
void solverPlot(char *fileName, double *x, double *y, int nx, int ny, double *Q){
    FILE *fp = fopen(fileName, "w");
    if (fp == NULL) {
        printf("Error opening file\n");
        return;
    }

    fprintf(fp, "X,Y,Z,Q \n");
     for(int i=0; i<nx; i++){
      for(int j=0; j<ny; j++){
        const double xn = x[i + j*nx]; 
        const double yn = y[i + j*nx]; 
        fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", xn, yn, 0.0, Q[i + j*nx]);
      }
    }
}


/* ************************************************************************** */
double readInputFile(char *fileName, char* tag){
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL) {
    printf("Error opening the input file\n");
    return -1;
  }

  int sk = 0; 
  double result; 
  char buffer[BUFSIZE];
  char fileTag[BUFSIZE]; 
  while(fgets(buffer, BUFSIZE, fp) != NULL){
    sscanf(buffer, "%s", fileTag);
    if(strstr(fileTag, tag)){
      fgets(buffer, BUFSIZE, fp);
      sscanf(buffer, "%lf", &result); 
      return result;
    }
    sk++;
  }

  if(sk==0){
    printf("could not find the tag: %s in the file %s\n", tag, fileName);
  }
}
