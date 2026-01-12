#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include<iostream>


template<typename T>    
void print_gpu(const T* data, int n, char* name = nullptr) {
    if (name != nullptr) {
        std::cout << name << ": \n";
	}
    T* temp = new T[n];
    cudaMemcpy(temp, data, n * sizeof(T), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) {
        std::cout << temp[i] << " ";
    }
    std::cout << std::endl;
	delete[] temp;
}

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
        return timer;
 }

#define LOG_NUM_BANKS 5          // 2^5 = 32
#define NUM_BANKS (1 << LOG_NUM_BANKS)
        
        __global__ void blellochScan(int n, int* global_idata, int* blockSums) {
            //return;
			int index = threadIdx.x;

			int* idata = global_idata + blockIdx.x * n;

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
    if (blockSums != nullptr) {
blockSums[blockIdx.x] = data[cf(n - 1)];
}
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

        /*
        __global__ void blellochScan(int n, int* global_idata, int* blockSums) {
            int index = threadIdx.x;
            int* idata = global_idata + blockIdx.x * n;

            __syncthreads();
            for (int d = 1; d < n; d *= 2) {
     //           if ((index + 1) % (d * 2) == 0) {
					//idata[index] += idata[index - d];
     //           }
                if (index < (n / 2 / d)) {
                    idata[index * d * 2 + d * 2 - 1] += idata[index * d * 2 - 1 + d];
                }
                __syncthreads();
            }

            if (index == 0) {
                if (blockSums != nullptr) {
                    blockSums[blockIdx.x] = idata[n - 1];
                }
                idata[n - 1] = 0;
            }
            __syncthreads();


            for (int d = n; d >= 2; d /= 2) {
                if (index < (n / 2 / d)) {
                    int fa = index * d * 2 + d * 2 - 1;
                    int lson = fa - d;
                    int rson = fa;
                //if((index + 1) % d == 0) {
                //    int lson = index - d / 2;
                //    int rson = index;
                    int t = idata[lson];
                    idata[lson] = idata[index];
                    idata[rson] = idata[index] + t;
				}
                __syncthreads();
            }
        }
     */

        __global__ void AddBlockSums(int n, int* global_idata, const int* blockSums) {
     int index = threadIdx.x;
 int* idata = global_idata + blockIdx.x * n;
			idata[index] += blockSums[blockIdx.x];
    
		}

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        static int ceil_to_pow2(int n) {
            int ret = 1;
            while (ret < n) {
             ret <<= 1;
     }
      return ret;
        }



        void scan(int n, int* odata, const int* idata) {
      // TODO

       int B = 1024;
		    int nn = std::max(ceil_to_pow2(n), B);
			int nnn = std::max(ceil_to_pow2(nn / B), B);
		
         int* temp_data = nullptr;
			cudaMalloc((void**)&temp_data, nn * sizeof(int));
			checkCUDAError("cudaMalloc temp_data failed");
    	cudaMemset(temp_data, 0, nn * sizeof(int));
			checkCUDAError("cudaMemset temp_data failed");
       cudaMemcpy(temp_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy to temp_data failed");
			
        int* blockSums = nullptr;
			cudaMalloc((void**)&blockSums, nnn * sizeof(int));
			checkCUDAError("cudaMalloc blockSums failed");
			cudaMemset(blockSums, 0, nnn * sizeof(int));
			checkCUDAError("cudaMemset blockSums failed");

            timer().startGpuTimer();
            blellochScan<< <nn / B, B / 2, (B + B / NUM_BANKS) * sizeof(int) >> > (B, temp_data, blockSums);
   checkCUDAError("blellochScan kernel launch failed");
         cudaDeviceSynchronize();
  checkCUDAError("blellochScan synchronization failed");
     

            if (nn / B > 1) {
      cudaDeviceSynchronize();
				blellochScan << <1, nnn / 2, (nnn + nnn / NUM_BANKS) * sizeof(int) >> > (nnn, blockSums, nullptr);
    checkCUDAError("blellochScan (blockSums) kernel launch failed");
            cudaDeviceSynchronize();
    checkCUDAError("blellochScan (blockSums) synchronization failed");
      AddBlockSums << <nn / B, B >> > (B, temp_data, blockSums);
              checkCUDAError("AddBlockSums kernel launch failed");
                cudaDeviceSynchronize();
           checkCUDAError("AddBlockSums synchronization failed");
         }
            timer().endGpuTimer();

			cudaMemcpy(odata, temp_data, n * sizeof(int), cudaMemcpyDeviceToHost);
			checkCUDAError("cudaMemcpy from temp_data failed");

			cudaFree(temp_data);
			cudaFree(blockSums);

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
     
      // Step 1: Allocate device memory
  int* dev_idata = nullptr;
          int* dev_bools = nullptr;
            int* dev_indices = nullptr;
            int* dev_odata = nullptr;
            
        cudaMalloc((void**)&dev_idata, n * sizeof(int));
   checkCUDAError("cudaMalloc dev_idata failed");
 cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed");
  cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
   checkCUDAError("cudaMalloc dev_odata failed");
            
     // Step 2: Copy input data to device
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
checkCUDAError("cudaMemcpy to dev_idata failed");
        
            // Step 3: Map to boolean array (1 if non-zero, 0 if zero)
     int blockSize = 256;
            int gridSize = (n + blockSize - 1) / blockSize;
      kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_bools, dev_idata);
            checkCUDAError("kernMapToBoolean kernel launch failed");
     cudaDeviceSynchronize();
            checkCUDAError("kernMapToBoolean synchronization failed");
      
  // Step 4: Perform exclusive scan on boolean array
          int* temp_bools = new int[n];
    int* temp_indices = new int[n];
            cudaMemcpy(temp_bools, dev_bools, n * sizeof(int), cudaMemcpyDeviceToHost);
   checkCUDAError("cudaMemcpy from dev_bools failed");
   
            scan(n, temp_indices, temp_bools);
 
    cudaMemcpy(dev_indices, temp_indices, n * sizeof(int), cudaMemcpyHostToDevice);
      checkCUDAError("cudaMemcpy to dev_indices failed");
            
     // Step 5: Scatter non-zero elements to output
 kernScatter<<<gridSize, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
        checkCUDAError("kernScatter kernel launch failed");
            cudaDeviceSynchronize();
 checkCUDAError("kernScatter synchronization failed");
            
 // Step 6: Calculate final count
    int count = temp_indices[n - 1] + temp_bools[n - 1];
      
            // Step 7: Copy result back to host
      cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy from dev_odata failed");
            
       // Clean up
            delete[] temp_bools;
delete[] temp_indices;
      cudaFree(dev_idata);
        cudaFree(dev_bools);
          cudaFree(dev_indices);
       cudaFree(dev_odata);
         
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
