
#include <iostream>

using namespace std;

__global__ void kernel3(int3 *input3, int3 *output3) {
	*output3 = *input3;
}

int main() {

	int3 *input3, *output3;
	int3 *dinput3, *doutput3;

	input3 = (int3*)malloc(sizeof(int3));
	output3 = (int3*)malloc(sizeof(int3));

	cudaMalloc(&dinput3, sizeof(int3));
	cudaMalloc(&doutput3, sizeof(int3));

	input3->x = 1;
	input3->y = 2;
	input3->z = 3;
	output3->x = 42;
	output3->y = 43;
	output3->z = 43;

	cudaMemcpy(dinput3, input3, sizeof(int3), cudaMemcpyHostToDevice);
	cudaMemcpy(doutput3, output3, sizeof(int3), cudaMemcpyHostToDevice);

	kernel3<<<1,1>>>(dinput3, doutput3);

	cudaMemcpy(output3, doutput3, sizeof(int3), cudaMemcpyDeviceToHost);

	bool passed = true;
	if (input3->x != output3->x) {
		cerr << "output3->x wrong! " << output3->x << endl;
		passed = false;
	}
	if (input3->y != output3->y) {
		cerr << "output3->y wrong! " << output3->y << endl;
		passed = false;
	}
	if (input3->z != output3->z) {
		cerr << "output3->z wrong! " << output3->z << endl;
		passed = false;
	}

	if (passed) {
		cout << "Vector 3 Test passed" << endl;
	}

	if (passed)
		return 0;
	else
		return 1;
}