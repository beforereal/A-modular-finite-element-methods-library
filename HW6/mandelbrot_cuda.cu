/*******************************************************************************
To compile: gcc -O3 -o mandelbrot mandelbrot.c -lm
To create an image with 4096 x 4096 pixels: ./mandelbrot 4096 4096 
*******************************************************************************/
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

void writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI);

#define MXITER 1000

/*******************************************************************************/
// Define a complex number
typedef struct {
  double x;
  double y;
}complex_t;


/*******************************************************************************/
// Return iterations before z leaves mandelbrot set for given c
__device__ int testpoint(complex_t c){
  int iter;  
  complex_t z = c;

  for(iter=0; iter<MXITER; iter++){ 
    // real part of z^2 + c 
    double tmp = (z.x*z.x) - (z.y*z.y) + c.x;
    // update with imaginary part of z^2 + c
    z.y = z.x*z.y*2. + c.y;
    // update real part
    z.x = tmp; 
    // check bound
    if((z.x*z.x+z.y*z.y)>4.0){ return iter;}
  }
  return iter; 
}

/*******************************************************************************/
// perform Mandelbrot iteration on a grid of numbers in the complex plane
// record the  iteration counts in the count array
__global__ void mandelbrotKernel(int Nre, int Nim, complex_t cmin, complex_t dc, float *count) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < Nre && n < Nim) {
        complex_t c;
        c.x = cmin.x + dc.x * m;
        c.y = cmin.y + dc.y * n;
        count[m + n * Nre] = (float)testpoint(c);
    }
}

/*******************************************************************************/

void computeMandelbrotGPU(int Nre, int Nim, complex_t cmin, complex_t dc, float *count) {
    dim3 blockSize(16, 16);  // Adjust block size as needed
    dim3 gridSize((Nre + blockSize.x - 1) / blockSize.x, (Nim + blockSize.y - 1) / blockSize.y);

    mandelbrotKernel<<<gridSize, blockSize>>>(Nre, Nim, cmin, dc, count);
    cudaDeviceSynchronize();
}



/*******************************************************************************/




int main(int argc, char **argv){

  // to create a 4096x4096 pixel image 
  // usage: ./mandelbrot 4096 4096 
  
  int Nre = (argc==3) ? atoi(argv[1]): 4096;
  int Nim = (argc==3) ? atoi(argv[2]): 4096;
  
  // storage for the iteration counts
  float *count;
  count = (float*) malloc(Nre*Nim*sizeof(float));
      float *count_d;
    cudaMalloc((void**)&count_d, Nre * Nim * sizeof(float));

  // Parameters for a bounding box for "c" that generates an interesting image
  // const float centRe = -.759856, centIm= .125547;
  // const float diam  = 0.151579;
  const float centRe = -0.5, centIm= 0;
  const float diam  = 3.0;

  complex_t cmin; 
  complex_t cmax;
  complex_t dc;

  cmin.x = centRe - 0.5*diam;
  cmax.x = centRe + 0.5*diam;
  cmin.y = centIm - 0.5*diam;
  cmax.y = centIm + 0.5*diam;

  //set step sizes
dc.x = (cmax.x-cmin.x)/(Nre-1);
dc.y = (cmax.y-cmin.y)/(Nim-1);

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Ensure device synchronization before measuring time
cudaDeviceSynchronize();
cudaEventRecord(start, 0);

// compute mandelbrot set
computeMandelbrotGPU(Nre, Nim, cmin, dc, count_d);
  
// copy from the GPU back to the host here
cudaMemcpy(count, count_d, Nre * Nim * sizeof(float), cudaMemcpyDeviceToHost);
  
// Ensure device synchronization before measuring elapsed time
cudaDeviceSynchronize();
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop);

float elapsedTime;
cudaEventElapsedTime(&elapsedTime, start, stop);
printf("Elapsed time (CUDA): %f ms\n", elapsedTime);

  // output mandelbrot to ppm format image
  printf("Printing mandelbrot.ppm...");
  writeMandelbrot("mandelbrot.ppm", Nre, Nim, count, 0, 80);
  printf("done.\n");

  cudaFree(count_d);	
  free(count);

  exit(0);
  return 0;
}  


/* Output data as PPM file */
void saveppm(const char *filename, unsigned char *img, int width, int height){

  /* FILE pointer */
  FILE *f;
  
  /* Open file for writing */
  f = fopen(filename, "wb");
  
  /* PPM header info, including the size of the image */
  fprintf(f, "P6 %d %d %d\n", width, height, 255);

  /* Write the image data to the file - remember 3 byte per pixel */
  fwrite(img, 3, width*height, f);

  /* Make sure you close the file */
  fclose(f);
}



void writeMandelbrot(const char *fileName, int width, int height, float *img, int minI, int maxI){

  int n, m;
  unsigned char *rgb   = (unsigned char*) calloc(3*width*height, sizeof(unsigned char));
  
  for(n=0;n<height;++n){
    for(m=0;m<width;++m){
      int id = m+n*width;
      int I = (int) (768*sqrt((double)(img[id]-minI)/(maxI-minI)));
      
      // change this to change palette
      if(I<256)      rgb[3*id+2] = 255-I;
      else if(I<512) rgb[3*id+1] = 511-I;
      else if(I<768) rgb[3*id+0] = 767-I;
      else if(I<1024) rgb[3*id+0] = 1023-I;
      else if(I<1536) rgb[3*id+1] = 1535-I;
      else if(I<2048) rgb[3*id+2] = 2047-I;

    }
  }

  saveppm(fileName, rgb, width, height);

  free(rgb);
}
