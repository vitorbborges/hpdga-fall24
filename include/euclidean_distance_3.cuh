// TODO: Refactor this to calculate ddistances in a fllatened array

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void euclidean_distance_kepler(const float* vec1, const float* vec2, float* distances, int num_vectors, int dimensions) {
    extern __shared__ float sharedData[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + tid;

    if (idx >= num_vectors) return;

    // Partial sum computation
    float sum = 0.0f;
    for (int j = tid; j < dimensions; j += blockDim.x) {
        float diff = vec1[idx * dimensions + j] - vec2[idx * dimensions + j];
        sum += diff * diff;
    }

    // Warp reduction
    sum = warpReduceSum(sum);

    // Block reduction using shared memory
    if ((tid & 31) == 0) sharedData[tid / 32] = sum;
    __syncthreads();

    if (tid < blockDim.x / 32) sum = warpReduceSum(sharedData[tid]);
    if (tid == 0) distances[blockIdx.x] = sqrtf(sum);
}