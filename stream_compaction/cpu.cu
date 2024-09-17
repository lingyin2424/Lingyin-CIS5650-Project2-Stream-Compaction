#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // Exclusive scan (prefix sum)
            // Output[0] = 0
            // Output[i] = Input[0] + Input[1] + ... + Input[i-1]
            if (n > 0) {
                odata[0] = 0;
                for (int i = 1; i < n; i++) {
                    odata[i] = odata[i - 1] + idata[i - 1];
                }
            }
            
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // Simply iterate through input and copy non-zero elements
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (idata[i] != 0) {
                    odata[count] = idata[i];
                    count++;
                }
            }
            
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            
            // Step 1: Create boolean array (map)
            int *bools = new int[n];
            for (int i = 0; i < n; i++) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }
            
            // Step 2: Perform exclusive scan on boolean array
            int *indices = new int[n];
            indices[0] = 0;
            for (int i = 1; i < n; i++) {
                indices[i] = indices[i - 1] + bools[i - 1];
            }
            
            // Step 3: Scatter - write elements to output based on indices
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (bools[i] == 1) {
                    odata[indices[i]] = idata[i];
                    count++;
                }
            }
            
            // Clean up temporary arrays
            delete[] bools;
            delete[] indices;
            
            timer().endCpuTimer();
            return count;
        }
    }
}
