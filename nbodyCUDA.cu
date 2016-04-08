//  qlogin -l gpu_host,gpus=1
//  /usr/local/cuda/include/cuda_runtime.h
//  nvcc nbodyCUDA.cu -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64

#include <stdio.h>

#define POINTS 7500
#define COL 3
#define CLUSTERS 50
#define X 0
#define Y 1
#define FLT_MAX 2.402823e+37

// [x location][y location][associated centroid (0-50)]
int inputPoints[POINTS][COL];
//  [x location][y location]
float xCentroids[CLUSTERS];
float yCentroids[CLUSTERS];

__global__ void parallel(float * xTotal, float * yTotal, float * nTotal, float * xCentroids, float * yCentroids){
  int idx = threadIdx.x;
  float x = xTotal[idx] / nTotal[idx];
  xCentroids[idx] = x;
  float y = yTotal[idx] / nTotal[idx];
  yCentroids[idx] = y;
}

__host__ void cuda_error_check() {
  if(cudaPeekAtLastError() != cudaSuccess) {
    printf("\nCUDA error: %s\n\n", cudaGetErrorString(cudaGetLastError()));
    cudaDeviceReset();
    //wait_exit();
    exit(1);
  } else {
    printf("\nNo cuda Errors Detected\n\n");
  }
}

void initialAssignCentroids(float xCentroids[], float yCentroids[]) {
  int i = 0;
  for(int x = 1; x <= 5; x++) {
    for(int y = 1; y <= 10; y++) {
      xCentroids[i] = x*11000+10000;
      yCentroids[i] = y*2200+10000;
      i++;
    }
  }
}

void printCentroids(float xCentroids[], float yCentroids[])
{
  int p;
  for(p=0; p<CLUSTERS; p++)
    {
      printf("[centroid]= %d ( %.2f , %.2f )\n", p, xCentroids[p], yCentroids[p]);
    }
}

void printInput(int score[][COL])
{
  int p;
  for(p=0; p<POINTS; p++)
  {
    printf("[%d %d]= %d [%d %d]= %d [centroid]= %d\n", p, X, score[p][X],
      p, Y, score[p][Y], score[p][2]);
  }
}

void readInput(int score[POINTS][COL])
{
  FILE *f=fopen("a3.txt", "r");

  int p;
  for(p=0; p<POINTS; p++)
    {
      fscanf(f, "%d %d", &score[p][X], &score[p][Y]);
      score[p][2] = 0;
    }
}

float pointDistance(int x1, int y1, float x2, float y2) {
  return sqrt(pow(x1-x2, 2)+pow(y1-y2, 2));
}

int assignCentroids(int points[][COL], float xCentroids[], float yCentroids[]) {
  int pointsUpdated = 0;
  for(int p = 0; p < POINTS; p++) {
    float minDist = FLT_MAX;
    int closestCentroidId = 0;
    for(int c = 0; c < CLUSTERS; c++) {
      float distance = pointDistance(points[p][X], points[p][Y], xCentroids[c], yCentroids[c]);
      if(distance < minDist) {
        minDist = distance;
        closestCentroidId = c;

      }

    }
    if(closestCentroidId != points[p][2]) {
      pointsUpdated++;
      points[p][2] = closestCentroidId;
    }
  }
  return pointsUpdated;
}

void moveCentroids(int points[][COL], float xCentroids[], float yCentroids[]) {
  static float xTotal[CLUSTERS];
  static float yTotal[CLUSTERS];
  static float nTotal[CLUSTERS];
  for(int i = 0; i<CLUSTERS; i++){
    xTotal[i] = yTotal[i] = nTotal[i] = 0;
  }

  for(int p = 0; p<POINTS; p++) {
    xTotal[points[p][2]] += points[p][0];
    yTotal[points[p][2]] += points[p][1];
    nTotal[points[p][2]] += 1;
  }

  for(int c = 0; c<CLUSTERS; c++){
    xCentroids[c] = xTotal[c]/nTotal[c];
    yCentroids[c] = yTotal[c]/nTotal[c];
  }
}

void moveCentroidsCUDA(int h_points[][3], float h_xCentroids[], float h_yCentroids[]) {
  static float h_xTotal[CLUSTERS];
  static float h_yTotal[CLUSTERS];
  static float h_nTotal[CLUSTERS];
  for(int i = 0; i<CLUSTERS; i++){
    h_xTotal[i] = h_yTotal[i] = h_nTotal[i] = 0;
  }

  for(int p = 0; p<POINTS; p++) {
    h_xTotal[h_points[p][2]] += h_points[p][0];
    h_yTotal[h_points[p][2]] += h_points[p][1];
    h_nTotal[h_points[p][2]] += 1;
  }

  const int TOTALS_ARRAY_SIZE = CLUSTERS;
  const int TOTALS_ARRAY_BYTES = TOTALS_ARRAY_SIZE * sizeof(float);

  const int CENTROID_ARRAY_SIZE = CLUSTERS;
  const int CENTROID_ARRAY_BYTES = CENTROID_ARRAY_SIZE * sizeof(float);

  float * d_xTotal;
  float * d_yTotal;
  float * d_nTotal;
  float * d_xCentroids;
  float * d_yCentroids;

  cudaMalloc((void**) &d_xTotal, TOTALS_ARRAY_BYTES);
  cudaMalloc((void**) &d_yTotal, TOTALS_ARRAY_BYTES);
  cudaMalloc((void**) &d_nTotal, TOTALS_ARRAY_BYTES);
  cudaMalloc((void**) &d_xCentroids, CENTROID_ARRAY_BYTES);
  cudaMalloc((void**) &d_yCentroids, CENTROID_ARRAY_BYTES);

  cudaMemcpy(d_xTotal, h_xTotal, TOTALS_ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_yTotal, h_yTotal, TOTALS_ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_nTotal, h_nTotal, TOTALS_ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_xCentroids, h_xCentroids, CENTROID_ARRAY_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_yCentroids, h_yCentroids, CENTROID_ARRAY_BYTES, cudaMemcpyHostToDevice);

  parallel<<<1, CLUSTERS>>>(d_xTotal, d_yTotal, d_nTotal, d_xCentroids, d_yCentroids);
  cuda_error_check();

  cudaMemcpy(h_xTotal, d_xTotal, TOTALS_ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_yTotal, d_yTotal, TOTALS_ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_nTotal, d_nTotal, TOTALS_ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_xCentroids, d_xCentroids, CENTROID_ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_yCentroids, d_yCentroids, CENTROID_ARRAY_BYTES, cudaMemcpyDeviceToHost);

  cudaFree(d_xTotal);
  cudaFree(d_yTotal);
  cudaFree(d_nTotal);
  cudaFree(d_xCentroids);
  cudaFree(d_yCentroids);
}

int main(int argc, char ** argv) {
  readInput(inputPoints);
  printInput(inputPoints);
  initialAssignCentroids(xCentroids, yCentroids);
  printCentroids(xCentroids, yCentroids);
  int u = 1;
  while(u > 0) {
    u = assignCentroids(inputPoints, xCentroids, yCentroids);
    printf("%d points updated\n", u);
    moveCentroids(inputPoints, xCentroids, yCentroids);
    //moveCentroidsCUDA(inputPoints, xCentroids, yCentroids);
  }
  printCentroids(xCentroids, yCentroids);

	return 0;
}
