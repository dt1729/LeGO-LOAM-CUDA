#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

__device__ __global__ uint16_t *allPushedIndX; // array for tracking points of a segmented object
__device__ __global__ uint16_t *allPushedIndY;

__device__ __global__ uint16_t *queueIndX; // array for breadth-first search process of segmentation, for speed
__device__ __global__ uint16_t *queueIndY;

// TODO: Update global variables using a function that copies data from pointcloud to a flattened array
__global__ void groundRemovalCUDA(float *fullCloudX, float *fullCloudY, float *fullCloudZ, float *groundMat){
    size_t lowerInd, upperInd;
    float diffX, diffY, diffZ, angle;
    // groundMat
    // -1, no valid info to check if ground of not
    //  0, initial value, after validation, means not ground
    //  1, ground
    // These are written under the assumption that the data transferred to CUDA is 
    lowerInd = threadIdx.x * blockDim.x + blockIdx.x;
    upperInd = (threadIdx.x + 1)* blockDim.x  + blockIdx.x;

    if(intensity[lowerInd] == -1 || intensity[upperInd] == -1){
        groundMat[threadIdx.x + blockIdx.x * blockDim.x] = -1;
        return;
    }

    diffX = fullCloudX[upperInd] - fullCloudX[lowerInd];
    diffY = fullCloudY[upperInd] - fullCloudY[lowerInd];
    diffZ = fullCloudZ[upperInd] - fullCloudZ[lowerInd];
    
    angle = atan2(diffZ, sqrt(diffX*diffX + diffY*diffY) ) * 180 / M_PI;

    if (abs(angle - sensorMountAngle) <= 10){
        groundMat[threadIdx.x + blockIdx.x * blockDim.x] = 1;
        groundMat[(threadIdx.x + 1) + blockIdx.x * blockDim.x] = 1;
    }
}

// TODO: At the time of creation of clouds from ROSMsg, create flat arrays as well
void groundRemovalCUDAcall(){

    // Assign memory for host device
    float* fullCloudX, fullCloudY, fullCloudZ, groundMatCUDA;

    // Assign memory for remote device
    cudaMallocManaged(&fullCloudX, fullCloud->points.size()*sizeof(float));
    cudaMallocManaged(&fullCloudY, fullCloud->points.size()*sizeof(float));
    cudaMallocManaged(&fullCloudZ, fullCloud->points.size()*sizeof(float));

    cudaMallocManaged(&groundMatCUDA, Horizon_SCAN*groundScanInd*sizeof(float));
    
    // See if CUDA memory prefetch needs to be done here.

    // Copy from host to device
    cudaMemcpy(&fullCloudX, flat_array_X, fullCloud->points.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&fullCloudY, flat_array_Y, fullCloud->points.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(&fullCloudZ, flat_array_Z, fullCloud->points.size()*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(&groundMatCUDA, flat_array_groundMat, Horizon_SCAN*groundScanInd*sizeof(float), cudaMemcpyHostToDevice);

    // Call the groundRemovalCuda function with <<<blocksPerGrid, threadsPerBlock>>>
    int threadsPerBlock = groundScanInd;
    int blocksPerGrid   = (Horizon_SCAN*groundScanInd);
    groundRemovalCUDA<<<blocksPerGrid, threadsPerBlock>>>(fullCloudX, fullCloudY, fullCloudZ, flat_array_groundMat);

    // Memcopy back to host cudaMemcpy
    cudaMemcpy(&groundMatCUDA, flat_array_groundMat, Horizon_SCAN*groundScanInd*sizeof(float), cudaMemcpyDeviceToHost);

    // Save groundMat the memory to pcl clouds.
    // TODO: ADD OPENMP Parfor to copy arrays to PCL cloud.

    // extract ground cloud (groundMat == 1)
    // mark entry that doesn't need to label (ground and invalid point) for segmentation
    // note that ground remove is from 0~N_SCAN-1, need rangeMat for mark label matrix for the 16th scan

    for (size_t i = 0; i < N_SCAN; ++i){
        for (size_t j = 0; j < Horizon_SCAN; ++j){
            if (groundMat.at<int8_t>(i,j) == 1 || rangeMat.at<float>(i,j) == FLT_MAX){
                labelMat.at<int>(i,j) = -1;
            }
        }
    }

    if (pubGroundCloud.getNumSubscribers() != 0){
        for (size_t i = 0; i <= groundScanInd; ++i){
            for (size_t j = 0; j < Horizon_SCAN; ++j){
                if (groundMat.at<int8_t>(i,j) == 1)
                    groundCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
            }
        }
    }

    // Free cuda memory cudaFree
    cudaFree(fullCloudX);
    cudaFree(fullCloudY);
    cudaFree(fullCloudZ);
    cudaFree(groundMat);
}

__global__ void cloudSegmentationCUDA(void* labelMat, void* groundMat, void* outlierCloud){
    if (labelMat->at<int>(blockIdx.x * blockDim.x, threadIdx.x) == 0)
        labelComponentsCUDA();

    __device__ int sizeOfSegCloud = 0;

    startRingIndexCUDA[blockIdx.x * blockDim.x] = sizeOfSegCloud -1 + 5;

    if(labelMat->at<int>(blockIdx.x * blockDim.x, threadIdx.x) > 0 || groundMat->at<int_t>(blockIdx.x * blockDim.x, threadIdx.x) == 1){
        if(labelMat->at<int>(blockIdx.x * blockDim.x, threadIdx.x) == 999999){
            if(blockIdx.x * blockDim.x > groundScanInd && threadIdx % 5 == 0){
                outlierCloud->push_back(fullCloud->points[blockIdx.x * blockDim.x + threadIdx.x*Horizon_SCAN]);
                return;
            }
            else{
                return;
            }
        }
    }

    // majority of ground points are skipped
    if (groundMat->at<int8_t>(blockIdx.x * blockDim.x, threadIdx.x) == 1){
        if (threadIdx.x%5!=0 && threadIdx.x>5 && threadIdx.x<Horizon_SCAN-5)
            return;
    }
}

void cloudSegmentationCUDAcall(){
    cv::Mat *rangeMat_ptr; // range matrix for range image
    cv::Mat *labelMat_ptr; // label matrix for segmentaiton marking
    cv::Mat *groundMat_ptr; // ground matrix for ground cloud marking

    cudaMallocManaged((void**)&rangeMat_ptr, sizeof(rangeMat));
    cudaMallocManaged((void**)&labelMat_ptr, sizeof(labelMat));
    cudaMallocManaged((void**)&groundMat_ptr, sizeof(groundMat));

    cudaMemcpy(rangeMat_ptr, &rangeMat, sizeof(cv::Mat), cudaMemcpyHostToDevice);
    cudaMemcpy(labelMat_ptr, &labelMat, sizeof(cv::Mat), cudaMemcpyHostToDevice);
    cudaMemcpy(groundMat_ptr, &groundMat, sizeof(cv::Mat), cudaMemcpyHostToDevice);

    cloudSegmentationCUDA<<<Horizon_SCAN*groundScanInd>>>(labelMat_ptr, groundMat_ptr, outlierCloud_ptr);

    cudaMemcpy(&rangeMat, rangeMat_ptr, sizeof(cv::Mat), cudaMemcpyDeviceToHost);
    cudaMemcpy(&labelMat, labelMat_ptr, sizeof(cv::Mat), cudaMemcpyDeviceToHost);
    cudaMemcpy(&groundMat, groundMat_ptr, sizeof(cv::Mat), cudaMemcpyDeviceToHost);

    cudaFree(rangeMat_ptr);
    cudaFree(labelMat_ptr);
    cudaFree(groundMat_ptr);

    // mark ground points so they will not be considered as edge features later
    segMsg.segmentedCloudGroundFlag[sizeOfSegCloud] = (groundMat.at<int8_t>(i,j) == 1);
    // mark the points' column index for marking occlusion later
    segMsg.segmentedCloudColInd[sizeOfSegCloud] = j;
    // save range info
    segMsg.segmentedCloudRange[sizeOfSegCloud]  = rangeMat.at<float>(i,j);
    // save seg cloud
    segmentedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
    // size of seg cloud
    ++sizeOfSegCloud;
}

// TODO: verify if the code calls value from other kernel while working on this,
// because that would enforce sequential execution
// TODO: If we can't use class member objects call p1, p2
// TODO: Update labelMat object with void pointer as used in cloudSegmentationCUDA
__device__ void labelComponentsCUDA(int row, int col, void* labelMat){
    float d1, d2, alpha, angle;
    int fromIndX, fromIndY, thisIndX, thisIndY; 
    bool lineCountFlag[N_SCAN] = {false};

    // Make these variables in global namespace, 
    // is defining a variable in class directly inherited in the kernel function
    p1[0] = row;
    p2[0] = column;

    int queueSize = 1;
    int queueStartInd = 0;
    int queueEndInd = 1;

    allPushedIndX[0] = row;
    allPushedIndY[0] = col;
    int allPushedIndSize = 1;

    while(queueSize > 0){
        // Pop point
        fromIndX = p1[queueStartInd];
        fromIndY = p2[queueStartInd];
        --queueSize;
        ++queueStartInd;
        // Mark popped point
        labelMat.at<int>(fromIndX, fromIndY) = labelCount;
        // Loop through all the neighboring grids of popped grid
        for (auto iter = neighborIterator.begin(); iter != neighborIterator.end(); ++iter){
            // new index
            thisIndX = fromIndX + (*iter).first;
            thisIndY = fromIndY + (*iter).second;
            // index should be within the boundary
            if (thisIndX < 0 || thisIndX >= N_SCAN)
                continue;
            // at range image margin (left or right side)
            if (thisIndY < 0)
                thisIndY = Horizon_SCAN - 1;
            if (thisIndY >= Horizon_SCAN)
                thisIndY = 0;
            // prevent infinite loop (caused by put already examined point back)
            if (labelMat.at<int>(thisIndX, thisIndY) != 0)
                continue;

            d1 = std::max(rangeMat.at<float>(fromIndX, fromIndY), 
                            rangeMat.at<float>(thisIndX, thisIndY));
            d2 = std::min(rangeMat.at<float>(fromIndX, fromIndY), 
                            rangeMat.at<float>(thisIndX, thisIndY));

            if ((*iter).first == 0)
                alpha = segmentAlphaX;
            else
                alpha = segmentAlphaY;

            angle = atan2(d2*sin(alpha), (d1 -d2*cos(alpha)));

            if (angle > segmentTheta){

                p1[queueEndInd] = thisIndX;
                p2[queueEndInd] = thisIndY;
                ++queueSize;
                ++queueEndInd;

                labelMat.at<int>(thisIndX, thisIndY) = labelCount;
                lineCountFlag[thisIndX] = true;

                allPushedIndX[allPushedIndSize] = thisIndX;
                allPushedIndY[allPushedIndSize] = thisIndY;
                ++allPushedIndSize;
            }
        }
    }

    // check if this segment is valid
    bool feasibleSegment = false;
    if (allPushedIndSize >= 30)
        feasibleSegment = true;
    else if (allPushedIndSize >= segmentValidPointNum){
        int lineCount = 0;
        for (size_t i = 0; i < N_SCAN; ++i)
            if (lineCountFlag[i] == true)
                ++lineCount;
        if (lineCount >= segmentValidLineNum)
            feasibleSegment = true;            
    }
    // segment is valid, mark these points
    if (feasibleSegment == true){
        ++labelCount;
    }else{ // segment is invalid, mark these points
        for (size_t i = 0; i < allPushedIndSize; ++i){
            labelMat.at<int>(allPushedIndX[i], allPushedIndY[i]) = 999999;
        }
    }
}

