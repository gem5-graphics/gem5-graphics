

#include <iostream>

#define MAGIC 1234

using namespace std;

__constant__ int constantMem[1];


__global__ void kernel(int *out) {

	*out = constantMem[0];

}



int main() {

	int* input = (int*)malloc(sizeof(int));

	*input = MAGIC;

	cudaMemcpyToSymbol(constantMem, input, sizeof(int));

	int *dout;

	cudaMalloc(&dout, sizeof(int));

	cudaMemset(dout, 0, sizeof(int));

	kernel<<<1,1>>>(dout);

	int *out = (int*)malloc(sizeof(int));

	cudaMemcpy(out, dout, sizeof(int), cudaMemcpyDeviceToHost);

	if (*out == MAGIC) {
		cout << "Test Passed" << endl;
	} else {
		cout << "Test Failed " << *out << endl;
	}


	return 0;
}