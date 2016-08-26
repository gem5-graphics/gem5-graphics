
#include <iostream>

using namespace std;

__global__ void kernel4(int4 *input4, int4 *output4) {
	*output4 = *input4;
}

__global__ void kernel2(int2 *input2, int2 *output2) {
	*output2 = *input2;
}

__global__ void kernelArr4(int4 *input4, int4 *output4) {
	int tid = threadIdx.x;
	output4[tid] = input4[tid];
}

int main() {
	int4 *input4, *output4;
	int4 *dinput4, *doutput4;

	input4 = (int4*)malloc(sizeof(int4));
	output4 = (int4*)malloc(sizeof(int4));

	cudaMalloc(&dinput4, sizeof(int4));
	cudaMalloc(&doutput4, sizeof(int4));

	input4->x = 1;
	input4->y = 2;
	input4->z = 3;
	input4->w = 4;
	output4->x = 42;
	output4->y = 43;
	output4->z = 44;
	output4->w = 45;

	cudaMemcpy(dinput4, input4, sizeof(int4), cudaMemcpyHostToDevice);
	cudaMemcpy(doutput4, output4, sizeof(int4), cudaMemcpyHostToDevice);

	kernel4<<<1,1>>>(dinput4, doutput4);

	cudaMemcpy(output4, doutput4, sizeof(int4), cudaMemcpyDeviceToHost);

	bool passed = true;
	if (input4->x != output4->x) {
		cerr << "output4->x wrong! " << output4->x << endl;
		passed = false;
	}
	if (input4->y != output4->y) {
		cerr << "output4->y wrong! " << output4->y << endl;
		passed = false;
	}
	if (input4->z != output4->z) {
		cerr << "output4->z wrong! " << output4->z << endl;
		passed = false;
	}
	if (input4->w != output4->w) {
		cerr << "output4->w wrong! " << output4->w << endl;
		passed = false;
	}

	if (passed) {
		cout << "Vector 4 Test passed" << endl;
	}

	int2 *input2, *output2;
	int2 *dinput2, *doutput2;

	input2 = (int2*)malloc(sizeof(int2));
	output2 = (int2*)malloc(sizeof(int2));

	cudaMalloc(&dinput2, sizeof(int2));
	cudaMalloc(&doutput2, sizeof(int2));

	input2->x = 1;
	input2->y = 2;
	output2->x = 42;
	output2->y = 43;

	cudaMemcpy(dinput2, input2, sizeof(int2), cudaMemcpyHostToDevice);
	cudaMemcpy(doutput2, output2, sizeof(int2), cudaMemcpyHostToDevice);

	kernel2<<<1,1>>>(dinput2, doutput2);

	cudaMemcpy(output2, doutput2, sizeof(int2), cudaMemcpyDeviceToHost);

	bool passed2 = true;
	if (input2->x != output2->x) {
		cerr << "output2->x wrong! " << output2->x << endl;
		passed2 = false;
	}
	if (input2->y != output2->y) {
		cerr << "output2->y wrong! " << output2->y << endl;
		passed2 = false;
	}

	if (passed2) {
		cout << "Vector 2 Test passed" << endl;
	} else {
		passed = false;
	}

	int4 *inputArr4, *outputArr4;
	int4 *dinputArr4, *doutputArr4;

	inputArr4 = (int4*)malloc(sizeof(int4)*32);
	outputArr4 = (int4*)malloc(sizeof(int4)*32);

	cudaMalloc(&dinputArr4, sizeof(int4)*32);
	cudaMalloc(&doutputArr4, sizeof(int4)*32);

	for (int i=0; i<32; i++) {
		inputArr4[i].x = 5;
		inputArr4[i].y = 2+i*4;
		inputArr4[i].z = 3+i*4;
		inputArr4[i].w = 4+i*4;
		outputArr4[i].x = 4001+i*4;
		outputArr4[i].y = 4002+i*4;
		outputArr4[i].z = 4003+i*4;
		outputArr4[i].w = 4004+i*4;
	}

	cudaMemcpy(dinputArr4, inputArr4, sizeof(int4)*32, cudaMemcpyHostToDevice);
	cudaMemcpy(doutputArr4, outputArr4, sizeof(int4)*32, cudaMemcpyHostToDevice);

	kernelArr4<<<1,32>>>(dinputArr4, doutputArr4);

	cudaMemcpy(outputArr4, doutputArr4, sizeof(int4)*32, cudaMemcpyDeviceToHost);

	bool passedArr = true;
	for (int i=0; i<32; i++) {
		if (outputArr4[i].x != inputArr4[i].x) {
			cerr << "outputArr4[" << i << "].x wrong! " << outputArr4[i].x << endl;
			passedArr = false;
		}
		if (outputArr4[i].y != inputArr4[i].y) {
			cerr << "outputArr4[" << i << "].y wrong! " << outputArr4[i].y << endl;
			passedArr = false;
		}
		if (outputArr4[i].z != inputArr4[i].z) {
			cerr << "outputArr4[" << i << "].z wrong! " << outputArr4[i].z << endl;
			passedArr = false;
		}
		if (outputArr4[i].w != inputArr4[i].w) {
			cerr << "outputArr4[" << i << "].w wrong! " << outputArr4[i].w << endl;
			passedArr = false;
		}
	}

	if (passedArr) {
		cout << "Vector Array Test passed" << endl;
	} else {
		passed = false;
	}

	if (passed)
		return 0;
	else
		return 1;
}