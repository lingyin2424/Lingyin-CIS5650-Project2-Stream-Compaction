#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

#define LOG_NUM_BANKS 5              // 2^5 = 32
#define NUM_BANKS (1 << LOG_NUM_BANKS)
#define CONFLICT_FREE_OFFSET(idx) ((idx) >> LOG_NUM_BANKS)

        __global__ void blellochScan(int n, int* idata) {
			// 实际上只发起了 1 个 block n/2 线程
            //return;
			int index = threadIdx.x + blockIdx.x * blockDim.x;
   //         if(index >= n) {
   //             return;
			//}

            auto cf = [](int idx) { return idx + (idx >> LOG_NUM_BANKS); };
                
			extern __shared__ int data[]; // allocated on invocation

            data[cf(index * 2)] = idata[index * 2];
            data[cf(index * 2 + 1)] = idata[index * 2 + 1];
            __syncthreads();
            for (int d = 1; d < n; d *= 2) {
                if (index < (n / 2 / d)) {
                    data[cf(index * d * 2 + d * 2 - 1)] += data[cf(index * d * 2 - 1 + d)];
                }
				__syncthreads();
            }

            if (index == 0) {
                data[cf(n - 1)] = 0;
			}
            __syncthreads();

            for(int d = n / 2; d >= 1; d /= 2) {
                if(index < (n / 2 / d)) {
                    // [index - d + 1, index]
                    int fa = index * d * 2 + d * 2 - 1;
                    int lson = fa - d;
                    int rson = fa;
                    
                    int t = data[cf(lson)];
                    data[cf(lson)] = data[cf(fa)];
                    data[cf(rson)] = data[cf(fa)] + t;
                }

				__syncthreads();
            }

            idata[index * 2] = data[cf(index * 2)];
            idata[index * 2 + 1] = data[cf(index * 2 + 1)];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int nn = n;
            while (nn != (nn & -nn)) {
                nn += (nn & -nn);
            }

            int B = 256;
            nn = std::max(B, nn);
			int* temp_data = nullptr;
			cudaMalloc((void**)&temp_data, nn * sizeof(int));
			cudaMemset(temp_data, 0, nn * sizeof(int));
			cudaMemcpy(temp_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			blellochScan<< <1, nn / 2, (nn + nn / NUM_BANKS) * sizeof(int) >> > (nn, temp_data);

			cudaMemcpy(odata, temp_data, n * sizeof(int), cudaMemcpyDeviceToHost);

			cudaFree(temp_data);

            timer().endGpuTimer();


            for (int s = 0, i = 0; i < n; i++) {
                if (odata[i] != s) {
					printf("Error at %d: %d != %d\n", i, odata[i], s);
                }
				s += idata[i];
            }

   //         for (int i = 0; i < 10; i++) {
			//	printf("%d ", idata[i]);
   //         }
			//printf("\n");
   //         for (int i = 0; i < 10; i++) {
			//	printf("%d ", odata[i]);
   //         }
			//printf("\n");

        }

        __global__ void kernMapToBoolean(int n, int* bools, const int* idata) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return;
            }
            bools[index] = (idata[index] != 0) ? 1 : 0;
        }

        __global__ void kernScatter(int n, int* odata, const int* idata, const int* bools, const int* indices) {
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if (index >= n) {
                return;
            }
            if (bools[index]) {
                odata[indices[index]] = idata[index];
            }
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            //timer().startGpuTimer();
            
            // Step 1: Allocate device memory
            int* dev_idata = nullptr;
            int* dev_bools = nullptr;
            int* dev_indices = nullptr;
            int* dev_odata = nullptr;
            
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            
            // Step 2: Copy input data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            
            // Step 3: Map to boolean array (1 if non-zero, 0 if zero)
            int blockSize = 256;
            int gridSize = (n + blockSize - 1) / blockSize;
            kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_bools, dev_idata);
            
            // Step 4: Perform exclusive scan on boolean array
            int* temp_bools = new int[n];
            int* temp_indices = new int[n];
            cudaMemcpy(temp_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
            
            scan(n, temp_indices, temp_bools);
            
            cudaMemcpy(dev_indices, temp_indices, n * sizeof(int), cudaMemcpyHostToDevice);
            
            // Step 5: Scatter non-zero elements to output
            kernScatter<<<gridSize, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            // Step 6: Calculate final count
            int count = temp_indices[n - 1] + temp_bools[n - 1];
            
            // Step 7: Copy result back to host
            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            
            // Clean up
            delete[] temp_bools;
            delete[] temp_indices;
            cudaFree(dev_idata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            cudaFree(dev_odata);
            
            //timer().endGpuTimer();
            return count;
        }
    }
}


/*
#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void blellochScan(int n, int* data) {
            //return;
            int index = threadIdx.x + blockIdx.x * blockDim.x;
            if(index >= n) {
                return;
            }

            for (int d = 2; d < n; d *= 2) {
                if ((index + 1) % d == 0) {
                    data[index] += data[index - d / 2];
                }
                //return;
                __syncthreads();
            }
            if (index == n - 1) {
                data[index] = 0;
            }

            for(int d = n; d >= 2; d /= 2) {
                if ((index + 1) % d == 0) {
                    // [index - d + 1, index]
                    int lson = index - d / 2;
                    int rson = index;

                    int t = data[lson];
                    data[lson] = data[index];
                    data[rson] = data[index] + t;
                }

                __syncthreads();
            }
        }

   
void scan(int n, int* odata, const int* idata) {
    timer().startGpuTimer();
    // TODO

    int nn = n;
    while (nn != (nn & -nn)) {
        nn += (nn & -nn);
    }

    int B = 256;
    nn = std::max(B, nn);
    int* temp_data = nullptr;
    cudaMalloc((void**)&temp_data, nn * sizeof(int));
    cudaMemset(temp_data, 0, nn * sizeof(int));
    cudaMemcpy(temp_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
    blellochScan << <1, nn >> > (nn, temp_data);

    cudaMemcpy(odata, temp_data, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(temp_data);

    timer().endGpuTimer();

    for (int i = 0; i < 10; i++) {
        printf("%d ", idata[i]);
    }
    printf("\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", odata[i]);
    }
    printf("\n");

}


int compact(int n, int* odata, const int* idata) {
    timer().startGpuTimer();
    // TODO
    timer().endGpuTimer();
    return -1;
}
    }
}



*/