#include <iostream>

using namespace std;

#define MAGIC 1234

__global__ void kernel(int *array) {
	int value = *array;
        value += 1;
	*array = value;
}



int main() {

	int global = MAGIC;
	int *d_global;
	cudaMalloc(&d_global, sizeof(int));
	cudaMemcpy(d_global, &global, sizeof(int), cudaMemcpyHostToDevice);

	kernel<<<1,1>>>(d_global);

	cudaMemcpy(&global, d_global, sizeof(int), cudaMemcpyDeviceToHost);

	if (global == MAGIC+1) {
		cout << "Test Passed" << endl;
	} else {
		cout << "Test Failed " << global << endl;
	}
	return 0;
}
