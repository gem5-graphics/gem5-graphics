
#include <iostream>
using namespace std;

int main() {

    for (int i = 0; i <= 46; i++) {
        cout << i << ": " << cudaGetErrorString((cudaError_t)i) << endl;
    }

}
