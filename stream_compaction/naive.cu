#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernNaiveScan(int n, int offset, int *odata, const int *idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if(index >= offset) {
                odata[index] = idata[index - offset] + idata[index];
            }
            else {
                odata[index] = idata[index];
			}
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {

            int* buffer_1 = nullptr;
			int* buffer_2 = nullptr;

			cudaMalloc((void**)&buffer_1, n * sizeof(int));
			cudaMalloc((void**)&buffer_2, n * sizeof(int));
			cudaMemcpy(buffer_1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
			int B = 1024;
            for (int d = 1; d <= n; d *= 2) {
				kernNaiveScan <<<(n + B - 1) / B, B >>> (n, d, buffer_2, buffer_1);
				std::swap(buffer_1, buffer_2);
            }


			cudaMemcpy(odata + 1, buffer_1, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);
			odata[0] = 0;

			timer().endGpuTimer();
			cudaFree(buffer_1);
			cudaFree(buffer_2);

        }
    }
}
