/* This is a sample Advection solver in C
The advection equation-> \partial q / \partial t - u \cdot \nabla q(x,y) = 0
The grid of NX by NX evenly spaced points are used for discretization.
The first and last points in each direction are boundary points.
Approximating the advection operator by 1st order finite difference.
*/
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "advection.h"

#define BUFSIZE 512
/* ************************************************************************** */
int main(int argc, char *argv[])
{
  // if(argc!=2){
  //   printf("Usage: ./levelSet input.dat\n");
  //   return -1;
  // }
  // set argv[1] to input file name
  argv[1] = "input.dat";
  static int frame = 0;

  // Create an advection solver
  solver_t advc;
  // Create uniform rectangular (Cartesian) mesh
  advc.msh = createMesh(argv[1]);
  // Create time stepper
  tstep_t tstep = createTimeStepper(advc.msh.Nnodes);
  // Create Initial Field
  initialCondition(&advc);

  // Read input file for time variables
  tstep.tstart = readInputFile(argv[1], "[TSART]");
  tstep.tend = readInputFile(argv[1], "[TEND]");
  tstep.dt = readInputFile(argv[1], "[DT]");
  tstep.time = 0.0;

  // adjust time step size
  int Nsteps = ceil((tstep.tend - tstep.tstart) / tstep.dt);
  tstep.dt = (tstep.tend - tstep.tstart) / Nsteps;

  // Read input file for OUTPUT FREQUENCY i.e. in every 1000 steps
  int Noutput = readInputFile(argv[1], "[OUTPUT_FREQUENCY]");

  // write the initial solution i.e. q at t = tstart
  {
    char fname[BUFSIZ];
    sprintf(fname, "test_%04d.csv", frame++);
    solverPlot(fname, &advc.msh, advc.q);
  }

  // ********************Time integration***************************************/
  // for every steps
  for (int step = 0; step < Nsteps; step++)
  {
    // for every stage
    for (int stage = 0; stage < tstep.Nstage; stage++)
    {
      // Call integration function
      RhsQ(&advc, &tstep, stage);
    }

    tstep.time = tstep.time + tstep.dt;

    if (step % Noutput == 0)
    {
      char fname[BUFSIZ];
      sprintf(fname, "test_%04d.csv", frame++);
      solverPlot(fname, &advc.msh, advc.q);
    }
  }
}

/* ************************************************************************** */
void RhsQ(solver_t *solver, tstep_t *tstep, int stage)
{

  // Get a pointer to the mesh
  mesh_t *mesh = &solver->msh;

  double *resq = tstep->resq;
  double *q_point = solver->q;

  // Loop over all nodes in the mesh
  for (int j = 0; j < mesh->NY; j++)
  {
    for (int i = 0; i < mesh->NX; i++)
    {
      // Calculate the index of the current node
      int node = j * mesh->NX + i;

      // Get the indices of the neighboring nodes
      int east = mesh->N2N[node * 4];
      int north = mesh->N2N[node * 4 + 1];
      int west = mesh->N2N[node * 4 + 2];
      int south = mesh->N2N[node * 4 + 3];

      // Get the properties of the current node
      double x = mesh->x[node];
      double y = mesh->y[node];
      double u = solver->u[node * 2];
      double v = solver->u[node * 2 + 1];
      double q = solver->q[node];

      // Get the properties of the east node
      double x_east = mesh->x[east];
      double x_west = mesh->x[west];

      double y_north = mesh->y[north];
      double y_south = mesh->y[south];

      double u_east = solver->u[east * 2];
      double u_west = solver->u[west * 2];

      double v_south = solver->u[south * 2 + 1];
      double v_north = solver->u[north * 2 + 1];

      double q_east = solver->q[east];
      double q_west = solver->q[west];
      double q_north = solver->q[north];
      double q_south = solver->q[south];

      double del_uq_del_x, del_vq_del_y, rhs;

      // Calculate PDEs
      if (u >= 0)
      {
        del_uq_del_x = (u * q - u_west * q_west) / (x - x_west);
      }
      else
      {
        del_uq_del_x = (u_east * q_east - u * q) / (x_east - x);
      }
      if (v >= 0)
      {
        del_vq_del_y = (v * q - v_south * q_south) / (y - y_south);
      }
      else
      {
        del_vq_del_y = (v_north * q_north - v * q) / (y_north - y);
      }

      // Calculate the right-hand side of the advection equation
      rhs = -(del_uq_del_x + del_vq_del_y);

      // Update the residual and the quantity q at the current node
      resq[node] = tstep->rk4a[stage] * resq[node] + tstep->dt * rhs;
      q_point[node] += tstep->rk4b[stage] * resq[node];
    }
  }
}

/* ************************************************************************** */
void initialCondition(solver_t *solver)
{
  mesh_t *msh = &(solver->msh);

  // Scalar field
  solver->q = (double *)malloc(msh->Nnodes * sizeof(double));
  double *q = solver->q;

  // Velocity field
  solver->u = (double *)malloc(2*msh->Nnodes * sizeof(double));
  double *u = solver->u;

  // Velocity field
  double xc = 0.5, yc = 0.75, r = 0.15;

  // Set up initial conditions for q, u, and v
  for (int j = 0; j < msh->NY; j++)
  {
    for (int i = 0; i < msh->NX; i++)
    {
      int node = j * msh->NX + i;
      int uComponent = node * 2;
      int vComponent = node * 2 + 1;
      double x = msh->x[node];
      double y = msh->y[node];

      // initial q calculation function is defined as q = (x - xc)^2 + (y - yc)^2 - r
      q[node] = sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc)) - r;

      u[uComponent] = sin(4.0 * M_PI * (x + 0.5)) * sin(4 * M_PI * (y + 0.5));
      u[vComponent] = cos(4 * M_PI * (x + 0.5)) * cos(4 * M_PI * (y + 0.5));
    }
  }
}

/* ************************************************************************** */
// void createMesh(struct mesh *msh){
mesh_t createMesh(char *inputFile)
{

  mesh_t msh;

  // Read required fields i.e. NX, NY, XMIN, XMAX, YMIN, YMAX

  msh.NX = readInputFile(inputFile, "[NX]");
  msh.NY = readInputFile(inputFile, "[NY]");
  msh.xmin = readInputFile(inputFile, "[XMIN]");
  msh.xmax = readInputFile(inputFile, "[XMAX]");
  msh.ymin = readInputFile(inputFile, "[YMIN]");
  msh.ymax = readInputFile(inputFile, "[YMAX]");

  msh.Nnodes = msh.NX * msh.NY;
  msh.x = (double *)malloc(msh.Nnodes * sizeof(double));
  msh.y = (double *)malloc(msh.Nnodes * sizeof(double));

  /*
  Compute Coordinates of the nodes
  */
  for (int j = 0; j < msh.NY; j++)
  {
    for (int i = 0; i < msh.NX; i++)
    {
      int node = j * msh.NX + i;
      double deltaX = (msh.xmax - msh.xmin) / (msh.NX - 1);
      double deltaY = (msh.ymax - msh.ymin) / (msh.NY - 1);

      msh.x[node] = msh.xmin + i * deltaX;
      msh.y[node] = msh.ymin + j * deltaY;
    }
  }

  // Create connectivity and periodic connectivity
  /*
  for every node 4 connections east north west and south
  Note that periodic connections require specific treatment
  */
  msh.N2N = (int *)malloc(4 * msh.Nnodes * sizeof(int));

  for (int j = 0; j < msh.NY; j++)
  {
    for (int i = 0; i < msh.NX; i++)
    {
      int eastNeighbour = 4 * (j * msh.NX + i);
      int northNeighbour = eastNeighbour + 1;
      int westNeighbour = eastNeighbour + 2;
      int southNeighbour = eastNeighbour + 3;

      msh.N2N[eastNeighbour] = j * msh.NX + (i + 1) % msh.NX;
      msh.N2N[northNeighbour] = (j + 1) * msh.NX + i;
      msh.N2N[westNeighbour] = j * msh.NX + (i - 1) % msh.NX;
      msh.N2N[southNeighbour] = (j - 1) * msh.NX + i;

      // Cover boundary conditions  
      if (i == msh.NX - 1)
      {
        msh.N2N[eastNeighbour] = j * msh.NX;
      }

      if (j == msh.NY - 1)
      {
        msh.N2N[northNeighbour] = i;
      }

      if (i == 0)
      {
        msh.N2N[westNeighbour] = j * msh.NX + (msh.NX - 1);
      }

      if (j == 0)
      {
        msh.N2N[southNeighbour] = (msh.NY - 1) * msh.NX + i;
      }
    }
  }

  return msh;
}

/* ************************************************************************** */
void solverPlot(char *fileName, mesh_t *msh, double *Q)
{
  FILE *fp = fopen(fileName, "w");
  if (fp == NULL)
  {
    printf("Error opening file\n");
    return;
  }

  fprintf(fp, "X,Y,Z,Q \n");
  for (int n = 0; n < msh->Nnodes; n++)
  {
    fprintf(fp, "%.8f, %.8f,%.8f,%.8f\n", msh->x[n], msh->y[n], 0.0, Q[n]);
  }
}

/* ************************************************************************** */
double readInputFile(char *fileName, char *tag)
{
  FILE *fp = fopen(fileName, "r");
  if (fp == NULL)
  {
    printf("Error opening the input file\n");
    return -1;
  }

  double value = 0.0;
  char line[100];
  while (fgets(line, sizeof(line), fp))
  {
    if (strstr(line, tag) != NULL)
    {
      if (fgets(line, sizeof(line), fp) != NULL)
      {
        sscanf(line, "%lf", &value);
        break;
      }
    }
  }

  fclose(fp);
  return value;
}

/* ************************************************************************** */
tstep_t createTimeStepper(int Nnodes)
{
  tstep_t tstep;
  tstep.Nstage = 5;
  tstep.resq = (double *)calloc(Nnodes, sizeof(double));
  tstep.rhsq = (double *)calloc(Nnodes, sizeof(double));
  tstep.rk4a = (double *)malloc(tstep.Nstage * sizeof(double));
  tstep.rk4b = (double *)malloc(tstep.Nstage * sizeof(double));
  tstep.rk4c = (double *)malloc(tstep.Nstage * sizeof(double));

  tstep.rk4a[0] = 0.0;
  tstep.rk4a[1] = -567301805773.0 / 1357537059087.0;
  tstep.rk4a[2] = -2404267990393.0 / 2016746695238.0;
  tstep.rk4a[3] = -3550918686646.0 / 2091501179385.0;
  tstep.rk4a[4] = -1275806237668.0 / 842570457699.0;

  tstep.rk4b[0] = 1432997174477.0 / 9575080441755.0;
  tstep.rk4b[1] = 5161836677717.0 / 13612068292357.0;
  tstep.rk4b[2] = 1720146321549.0 / 2090206949498.0;
  tstep.rk4b[3] = 3134564353537.0 / 4481467310338.0;
  tstep.rk4b[4] = 2277821191437.0 / 14882151754819.0;

  tstep.rk4c[0] = 0.0;
  tstep.rk4c[1] = 1432997174477.0 / 9575080441755.0;
  tstep.rk4c[2] = 2526269341429.0 / 6820363962896.0;
  tstep.rk4c[3] = 2006345519317.0 / 3224310063776.0;
  tstep.rk4c[4] = 2802321613138.0 / 2924317926251.0;
  return tstep;
}
