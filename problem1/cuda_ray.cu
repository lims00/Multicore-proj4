
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include <sys/time.h>
// 추가된 CUDA 관련 헤더 파일
#include <cuda.h>
#include <chrono>

using namespace std::chrono;

#define CUDA 0
#define OPENMP 1
#define SPHERES 20
#define DIM 2048

#define rnd(x) (x * rand() / RAND_MAX)
#define INF 2e10f

struct Sphere {
    float r, b, g;
    float radius;
    float x, y, z;
    float (*hit)(struct Sphere* s, float ox, float oy, float* n);
};

// CUDA 커널 함수 추가
__device__ float hit(struct Sphere* s, float ox, float oy, float* n) {
    float dx = ox - s->x;
    float dy = oy - s->y;
    if (dx * dx + dy * dy < s->radius * s->radius) {
        float dz = sqrtf(s->radius * s->radius - dx * dx - dy * dy);
        *n = dz / sqrtf(s->radius * s->radius);
        return dz + s->z;
    }
    return -INF;
}

// CUDA 커널 함수 추가
__global__ void kernel(struct Sphere* s, unsigned char* ptr) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = x + y * DIM;
    float ox = (x - DIM / 2);
    float oy = (y - DIM / 2);

    float r = 0, g = 0, b = 0;
    float maxz = -INF;
    for (int i = 0; i < SPHERES; i++) {
        float n;
        float t = hit(&s[i], ox, oy, &n);
        if (t > maxz) {
            float fscale = n;
            r = s[i].r * fscale;
            g = s[i].g * fscale;
            b = s[i].b * fscale;
            maxz = t;
        }
    }

    ptr[offset * 4 + 0] = (int)(r * 255);
    ptr[offset * 4 + 1] = (int)(g * 255);
    ptr[offset * 4 + 2] = (int)(b * 255);
    ptr[offset * 4 + 3] = 255;
}

void ppm_write(unsigned char* bitmap, int xdim, int ydim, FILE* fp) {
    int i, x, y;
    fprintf(fp, "P3\n");
    fprintf(fp, "%d %d\n", xdim, ydim);
    fprintf(fp, "255\n");
    for (y = 0; y < ydim; y++) {
        for (x = 0; x < xdim; x++) {
            i = x + y * xdim;
            fprintf(fp, "%d %d %d ", bitmap[4 * i], bitmap[4 * i + 1], bitmap[4 * i + 2]);
        }
        fprintf(fp, "\n");
    }
}

int main(void) {
    int no_threads;
    int option;
    int x, y;
    unsigned char* bitmap;
    
    
    srand(time(NULL));

    FILE* fp = fopen("cudaresult", "w");

    

    struct Sphere* temp_s = (struct Sphere*)malloc(sizeof(struct Sphere) * SPHERES);
    for (int i = 0; i < SPHERES; i++) {
        temp_s[i].r = rnd(1.0f);
        temp_s[i].g = rnd(1.0f);
        temp_s[i].b = rnd(1.0f);
        temp_s[i].x = rnd(2000.0f) - 1000;
        temp_s[i].y = rnd(2000.0f) - 1000;
        temp_s[i].z = rnd(2000.0f) - 1000;
        temp_s[i].radius = rnd(200.0f) + 40;
    }

    bitmap = (unsigned char*)malloc(sizeof(unsigned char) * DIM * DIM * 4);
    auto start = high_resolution_clock::now();
           // GPU에서 사용할 메모리 할당
    struct Sphere* dev_s;
    unsigned char* dev_bitmap;
    cudaMalloc((void**)&dev_s, sizeof(struct Sphere) * SPHERES);
    cudaMalloc((void**)&dev_bitmap, sizeof(unsigned char) * DIM * DIM * 4);
    // CPU에서 GPU로 데이터 복사
    cudaMemcpy(dev_s, temp_s, sizeof(struct Sphere) * SPHERES, cudaMemcpyHostToDevice);
    // CUDA 커널 호출
    dim3 blocksPerGrid(DIM / 16, DIM / 16);
    dim3 threadsPerBlock(16, 16);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_s, dev_bitmap);
    // GPU에서 결과 이미지 데이터를 CPU로 복사
    cudaMemcpy(bitmap, dev_bitmap, sizeof(unsigned char) * DIM * DIM * 4, cudaMemcpyDeviceToHost);
    // GPU 메모리 해제
    cudaFree(dev_s);
    cudaFree(dev_bitmap);
    
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(stop - start);
    printf("DEVICE(CUDA) execution time:%dms",(int)duration.count());
    ppm_write(bitmap, DIM, DIM, fp);
    fclose(fp);
    free(bitmap);
    free(temp_s);

    return 0;
}
