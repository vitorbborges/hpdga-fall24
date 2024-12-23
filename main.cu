#include "priority_queue.cuh"

__global__ void test_priority_queue(Neighbor* data_points, int num_points, HeapType heapType) {
    __shared__ Neighbor shared_heap[MAX_HEAP_SIZE];
    __shared__ int shared_size;

    PriorityQueue pq(shared_heap, &shared_size, heapType);

    if (threadIdx.x == 0) {
        // Insert and extract elements in unusual orders
        pq.insert({50.0, 1});
        pq.insert({10.0, 2});
        pq.insert({70.0, 3});
        pq.print_heap();

        Neighbor top = pq.top();
        printf("Accessed Top (without extraction): (%f, %d)\n", top.distance, top.id);

        top = pq.pop();
        printf("Extracted Top: (%f, %d)\n", top.distance, top.id);
        pq.print_heap();

        pq.insert({30.0, 4});
        pq.insert({90.0, 5});
        pq.print_heap();

        top = pq.top();
        printf("Accessed Top (without extraction): (%f, %d)\n", top.distance, top.id);

        top = pq.pop();
        printf("Extracted Top: (%f, %d)\n", top.distance, top.id);
        pq.print_heap();

        pq.insert({20.0, 6});
        pq.insert({40.0, 7});
        pq.print_heap();

        while (shared_size > 0) {
            top = pq.top();
            printf("Accessed Top (without extraction): (%f, %d)\n", top.distance, top.id);
            top = pq.pop();
            printf("Extracted Top: (%f, %d)\n", top.distance, top.id);
            pq.print_heap();
        }
    }
}

int main() {
    const int num_points = 0; // No initial points needed for this test

    Neighbor* d_data_points;
    cudaMalloc(&d_data_points, num_points * sizeof(Neighbor));

    test_priority_queue<<<1, 32>>>(d_data_points, num_points, MAX_HEAP);
    cudaDeviceSynchronize();

    cudaFree(d_data_points);
    return 0;
}