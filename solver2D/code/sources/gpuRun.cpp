/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira
Instituto Superior T�cnico - Universidade de Lisboa
Av. Rovisco Pais 1, 1049-001 Lisboa, Portugal

This file is part of STAV-2D.

STAV-2D is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or any
later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see http://www.gnu.org/licenses/.

///////////////////////////////////////////////////////////////////////////////////////////////*/


/////////////////////////////////////////////////////////////////////////////////////////////////

// STL
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>

// OpenMP
#include <omp.h>

// CUDA
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

// STAV
#include "../headers/compile.hpp"
#include "../headers/gpuRun.hpp"
#include "../headers/boundaries.hpp"
#include "../headers/numerics.hpp"
#include "../headers/forcing.hpp"
#include "../headers/sediment.hpp"
#include "../headers/mesh.hpp"
#ifdef __STAV_MPI__
#include "../headers/mpiRun.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////

kernelLaunch::kernelLaunch(){

	gridSize = 0;
	blockSize = 0;
	minGridSize = 0;
}

kernelLaunch setBndRefValues;
kernelLaunch setBndConditions;

kernelLaunch getFluxes;
kernelLaunch applyFluxes;
kernelLaunch applyCorrections;
kernelLaunch applySources;

void kernelLaunch::setKernel(void* targetKernel, int arraySize){
	
	//cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, targetKernel, 0, arraySize);
	blockSize = 256;
	gridSize = (arraySize + blockSize - 1) / blockSize;
}

GLOBAL void setBndRefValuesKernel(physicalBoundary* ptrPhysicalBoundaries, int numBoundaries){

	int b = blockIdx.x * blockDim.x + threadIdx.x;
	if (b < numBoundaries)
		if (ptrPhysicalBoundaries[b].isUniformInlet)
			ptrPhysicalBoundaries[b].setRefValue();
}

GLOBAL void setBndConditionsKernel(elementGhost* gpuElemGhost, int numElemGhosts){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElemGhosts)
		gpuElemGhost[i].getConditions();
}

GLOBAL void getFluxesKernel(elementFlow* gpuElemFlow, int numElems){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElems)
		gpuElemFlow[i].computeFluxes();
}

GLOBAL void applyFluxesKernel(elementFlow* gpuElemFlow, int numElems){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElems)
		gpuElemFlow[i].applyFluxes();
}

GLOBAL void applyCorrectionsKernel(elementFlow* gpuElemFlow, int numElems){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElems)
		gpuElemFlow[i].applyCorrections();
}

GLOBAL void applySourcesKernel(element* gpuElem, int numElems){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numElems)
		gpuElem[i].applySourceTerms();
}

void simulationSTAV2D::runOnGPU(){

	std::cout << std::endl << std::endl;

    stepWClockTime = 0.0f;
    totalWClockTime = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    setBndRefValues.setKernel((void*) setBndRefValuesKernel, cpuBoundaries.numBoundaries);
	setBndConditions.setKernel((void*) setBndConditionsKernel, cpuBoundaries.numElemGhosts);

	getFluxes.setKernel((void*) getFluxesKernel, cpuMesh.numElems);
	applyFluxes.setKernel((void*) applyFluxesKernel, cpuMesh.numElems);
	applyCorrections.setKernel((void*) applyCorrectionsKernel, cpuMesh.numElems);
	applySources.setKernel((void*) applySourcesKernel, cpuMesh.numElems);

    while (currentTime <= finalTime) {

        cudaEventRecord(start);
        cudaEventSynchronize(start);

        if (cpuBoundaries.numBoundaries > 0){
            setBndRefValuesKernel <<< setBndRefValues.gridSize, setBndRefValues.blockSize >>> (gpuPhysicalBoundaries, cpuBoundaries.numBoundaries);
            setBndConditionsKernel <<< setBndConditions.gridSize, setBndConditions.blockSize >>> (gpuElemGhost, cpuBoundaries.numElemGhosts);
        }

        getFluxesKernel <<< getFluxes.gridSize, getFluxes.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
        cudaDeviceSynchronize();
        float gpuDt = thrust::reduce(thrust::device, cpuMesh.gpuElemDt, cpuMesh.gpuElemDt + cpuMesh.numElems, 99999999999.0, thrust::minimum<double>());
        cpuNumerics.dt = min(10.0, gpuDt);
        cudaMemcpyToSymbol(gpuNumerics.dt, &cpuNumerics.dt, sizeof(double));
        cudaDeviceSynchronize();

        applyFluxesKernel <<< applyFluxes.gridSize, applyFluxes.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
        applyCorrectionsKernel <<< applyCorrections.gridSize, applyCorrections.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
        applySourcesKernel <<< applySources.gridSize, applySources.blockSize >>> (cpuMesh.gpuElem, cpuMesh.numElems);
        cudaDeviceSynchronize();

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&stepWClockTime, start, stop);
        stepWClockTime /= 1000.0f;
        totalWClockTime += stepWClockTime;

        step++;
        currentTime += float(cpuNumerics.dt);
        cudaMemcpyToSymbol(gpuCurrentTime, &currentTime, sizeof(float));

        std::cout << std::fixed << std::setprecision(6) << "  dt (s): " << cpuNumerics.dt;
        std::cout << std::fixed << std::setprecision(4) << ",  GPU (s): " << stepWClockTime;
        std::cout << std::fixed << std::setprecision(3) << ",  Ratio: " << cpuNumerics.dt / stepWClockTime
                  << ",  Time (s): " << currentTime
                  << ",  Comp. (%): " << currentTime / finalTime * 100.0f
                  << ",  CUDA: " << 1 << std::endl;

        cpuNumerics.dt = 99999999999.0;
        cpuOutput.exportAll();
    }

	cpuOutput.exportMaxima();

	std::cout << std::endl;
	std::cout << std::fixed << std::setprecision(3) << "  Total Simulation Time (s): " << totalWClockTime << std::endl;
	std::cout << std::endl;
}

#ifdef __STAV_MPI__
void simulationSTAV2D::runSlaveOnGPU(){

	MPI_Barrier(COMPUTE);
	MPI_Barrier(GLOBALS);

	stepWClockTime = float(omp_get_wtime());

	setBndRefValues.setKernel((void*) setBndRefValuesKernel, cpuBoundaries.numBoundaries);
	setBndConditions.setKernel((void*) setBndConditionsKernel, cpuBoundaries.numElemGhosts);

	getFluxes.setKernel((void*) getFluxesKernel, cpuMesh.numElems);
	applyFluxes.setKernel((void*) applyFluxesKernel, cpuMesh.numElems);
	applyCorrections.setKernel((void*) applyCorrectionsKernel, cpuMesh.numElems);
	applySources.setKernel((void*) applySourcesKernel, cpuMesh.numElems);

	while (currentTime <= finalTime) {

		if(step > 0){
			for (unsigned i = 0; i < cpuMesh.numRemoteElems; ++i)
				cpuMesh.elemRemote[i].recvState();
			cudaMemcpy(cpuMesh.gpuElemRemoteState, cpuMesh.elemRemoteState, sizeof(elementState)*cpuMesh.numRemoteElems, cudaMemcpyHostToDevice);
			cudaMemcpy(cpuMesh.gpuElemRemoteScalars, cpuMesh.elemRemoteScalars, sizeof(elementScalars)*cpuMesh.numRemoteElems, cudaMemcpyHostToDevice);
		}

		if (cpuBoundaries.numElemGhosts > 0){
			setBndRefValuesKernel <<< setBndRefValues.gridSize, setBndRefValues.blockSize >>> (gpuPhysicalBoundaries, cpuBoundaries.numBoundaries);
			cudaMemcpy(&cpuInletRefValue.front(), gpuInletRefValue, sizeof(float)*cpuBoundaries.numBoundaries, cudaMemcpyDeviceToHost);
			cudaMemcpy(&cpuInletRefFactor.front(), gpuInletRefFactor, sizeof(float)*cpuBoundaries.numBoundaries, cudaMemcpyDeviceToHost);
		}

		if(cpuBoundaries.numBoundaries > 0){
			MPI_Allgather(&cpuInletRefValue.front(), int(cpuInletRefValue.size()), MPI_FLOAT,
					&cpuInletRefValueBuffer.front(), int(cpuInletRefValue.size()), MPI_FLOAT, GLOBALS);
			MPI_Allgather(&cpuInletRefFactor.front(), int(cpuInletRefFactor.size()), MPI_FLOAT,
					&cpuInletRefFactorBuffer.front(), int(cpuInletRefFactor.size()), MPI_FLOAT, GLOBALS);
			for(unsigned b = 0; b < cpuBoundaries.numBoundaries; ++b){
				cpuInletRefValue[b] = 0.0f;
				cpuInletRefFactor[b] = 0.0f;
				for(unsigned pr = 0; pr < myProc.worldSize; ++pr){
					cpuInletRefValue[b] = cpuInletRefValue[b] + cpuInletRefValueBuffer[b + pr*cpuBoundaries.numBoundaries];
					cpuInletRefFactor[b] = cpuInletRefFactor[b] + cpuInletRefFactorBuffer[b + pr*cpuBoundaries.numBoundaries];
				}
			}
		}

		if (cpuBoundaries.numElemGhosts > 0){
			cudaMemcpy(gpuInletRefValue, &cpuInletRefValue.front(), sizeof(float)*cpuBoundaries.numBoundaries, cudaMemcpyHostToDevice);
			cudaMemcpy(gpuInletRefFactor, &cpuInletRefFactor.front(), sizeof(float)*cpuBoundaries.numBoundaries, cudaMemcpyHostToDevice);
			setBndConditionsKernel <<< setBndConditions.gridSize, setBndConditions.blockSize >>> (gpuElemGhost, cpuBoundaries.numElemGhosts);
		}

		getFluxesKernel <<< getFluxes.gridSize, getFluxes.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);

        float gpuDt = thrust::reduce(thrust::device, cpuMesh.gpuElemDt, cpuMesh.gpuElemDt + cpuMesh.numElems, 99999999999.0, thrust::minimum<double>());
        cpuNumerics.dt = min(10.0, gpuDt);
		MPI_Allreduce(&cpuNumerics.dt, &cpuNumerics.dt, 1, MPI_DOUBLE, MPI_MIN, GLOBALS);
		cudaMemcpyToSymbol(gpuNumerics.dt, &cpuNumerics.dt, sizeof(double));

		step++;
		currentTime += float(cpuNumerics.dt);
		cudaMemcpyToSymbol(gpuCurrentTime, &currentTime, sizeof(float));

		applyFluxesKernel <<< applyFluxes.gridSize, applyFluxes.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
		applyCorrectionsKernel <<< applyCorrections.gridSize, applyCorrections.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
		//applySourcesKernel <<< applySources.gridSize, applySources.blockSize >>> (cpuMesh.gpuElem, cpuMesh.numElems);

		if(currentTime < finalTime){
			cudaMemcpy(cpuMesh.elemConnectedState, cpuMesh.gpuElemConnectedState, sizeof(elementState)*cpuMesh.numConnectedElems, cudaMemcpyDeviceToHost);
			cudaMemcpy(cpuMesh.elemConnectedScalars, cpuMesh.gpuElemConnectedScalars, sizeof(elementScalars)*cpuMesh.numConnectedElems, cudaMemcpyDeviceToHost);
			for (unsigned i = 0; i < cpuMesh.numConnectedElems; ++i)
				cpuMesh.elemConnected[i].sendState();
		}

		stepWClockTime = float(omp_get_wtime()) - stepWClockTime;

		if (step % timerStepsSKIP == 0 && step > 0)
			MPI_Gather(&stepWClockTime, 1, MPI_FLOAT, nullptr, 1, MPI_FLOAT, 0, GLOBALS);

		stepWClockTime = float(omp_get_wtime());

		MPI_Bcast(&globals.front(), int(globals.size()), MPI_UNSIGNED, 0, GLOBALS);
		MPI_Barrier(GLOBALS);

		myProc.writeResults = bool(globals[0]);
		myProc.reloadMesh = bool(globals[1]);

		if (myProc.writeResults || myProc.reloadMesh)
			gatherResults();

		if (myProc.writeResults)
			myProc.writeResults = false;

		if (myProc.reloadMesh){
			cpuMesh.reset();
			cpuMesh.recvFromMaster();
			myProc.reloadMesh = false;
		}

		MPI_Barrier(COMPUTE);
	}
}
#endif

/* Next step: thrust
 *
    struct functorFluxes{
      CPU GPU void operator()(elementFlow &flow) const { flow.computeFluxes(); }
    };

    thrust::for_each(thrust::device, cpuMesh.gpuElemFlow, cpuMesh.gpuElemFlow + cpuMesh.numElems, elementFlow::functorFluxes());

    float duration = 0.0f;
    int iterations = 5000;
    cudaEventRecord(start);
    cudaEventSynchronize(start);
    while(iterations > 0){
      int blockSize = cpuMesh.numElems;
      int numBlocks = cpuMesh.numElems/blockSize;
      elementFlow* begin = cpuMesh.gpuElemFlow;
      elementFlow* end = cpuMesh.gpuElemFlow + blockSize;
      for(int b = 0; b < numBlocks; ++b){
        //thrust::for_each(thrust::device, begin, end, elementFlow::functorFluxes());
        getFluxesKernel <<< getFluxes.gridSize, getFluxes.blockSize >>> (begin, blockSize);
        begin = end + 1;
        end = begin + blockSize;
      }
      iterations--;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&duration, start, stop);

    std::cout << duration;
    exitOnKeypress(0);

/*

// Debug CPU-GPU

/* ///////////////////////////////////////////////////////////////////// DEBUG INITIAL

float checkVarCPU[maxCONSERVED] = { 0.0f };
for (unsigned i = 0; i < cpuMesh.numElems; ++i)
for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarCPU[p] += cpuMesh.elemFluxes[i].flux[p];

elementFluxes* copyFluxesGPU = new elementFluxes[cpuMesh.numElems];
cudaError_t toGPU = cudaMemcpy(copyFluxesGPU, cpuMesh.gpuElemFluxes, sizeof(elementFluxes)*cpuMesh.numElems, cudaMemcpyDeviceToHost);

float checkVarGPU[maxCONSERVED] = { 0.0f };
for (unsigned i = 0; i < cpuMesh.numElems; ++i)
for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarGPU[p] += copyFluxesGPU[i].flux[p];

for (unsigned b = 0; b < cpuBoundaries.numBoundaries; ++b)
if (cpuBoundaries.physical[b].isUniformInlet)
cpuBoundaries.physical[b].setRefValue();

for (unsigned i = 0; i < cpuBoundaries.numElemGhosts; ++i)
cpuBoundaries.elemGhost[i].getConditions();

///////////////////////////////////////////////////////////////////// - */

/* ///////////////////////////////////////////////////////////////////// DEBUG BOUNDARIES

float* copyRefValue = new float[cpuBoundaries.numBoundaries];
toGPU = cudaMemcpy(copyRefValue, gpuInletRefValue, sizeof(float)*cpuBoundaries.numBoundaries, cudaMemcpyDeviceToHost);

float* copyRefFactor = new float[cpuBoundaries.numBoundaries];
toGPU = cudaMemcpy(copyRefFactor, gpuInletRefFactor, sizeof(float)*cpuBoundaries.numBoundaries, cudaMemcpyDeviceToHost);

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarCPU[p] = 0.0f;

for (unsigned i = 0; i < cpuBoundaries.numElemGhosts; ++i){
checkVarCPU[0] += cpuBoundaries.elemGhost[i].state.h;
checkVarCPU[1] += cpuBoundaries.elemGhost[i].state.vel.x;
checkVarCPU[2] += cpuBoundaries.elemGhost[i].state.vel.y;
}

elementGhost* copyGhostsGPU = new elementGhost[cpuBoundaries.numElemGhosts];
toGPU = cudaMemcpy(copyGhostsGPU, gpuElemGhost, sizeof(elementGhost)*cpuBoundaries.numElemGhosts, cudaMemcpyDeviceToHost);

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarGPU[p] = 0.0f;

for (unsigned i = 0; i < cpuBoundaries.numElemGhosts; ++i){
checkVarGPU[0] += copyGhostsGPU[i].state.h;
checkVarGPU[1] += copyGhostsGPU[i].state.vel.x;
checkVarGPU[2] += copyGhostsGPU[i].state.vel.y;
}

double threadPrivateDt = 99999999999.0;
for (unsigned i = 0; i < cpuMesh.numElems; ++i){
cpuMesh.elemFlow[i].computeFluxes();
if (cpuMesh.elemDt[i] < threadPrivateDt)
threadPrivateDt = cpuMesh.elemDt[i];
}

///////////////////////////////////////////////////////////////////// - */

/* ///////////////////////////////////////////////////////////////////// DEBUG FLUXES

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarCPU[p] = 0.0f;

for (unsigned i = 0; i < cpuMesh.numElems; ++i)
for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarCPU[p] += cpuMesh.elemFluxes[i].flux[p];

toGPU = cudaMemcpy(copyFluxesGPU, cpuMesh.gpuElemFluxes, sizeof(elementFluxes)*cpuMesh.numElems, cudaMemcpyDeviceToHost);

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarGPU[p] = 0.0f;

for (unsigned i = 0; i < cpuMesh.numElems; ++i)
for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarGPU[p] += copyFluxesGPU[i].flux[p];

///////////////////////////////////////////////////////////////////// - */

/* ///////////////////////////////////////////////////////////////////// DEBUG DT

if (abs(cpuNumerics.dt - threadPrivateDt) <= 1.0e-6)
cpuNumerics.dt = threadPrivateDt;

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarCPU[p] = 0.0f;

for (unsigned i = 0; i < cpuMesh.numElems; ++i){
checkVarCPU[0] += cpuMesh.elemState[i].h;
checkVarCPU[1] += cpuMesh.elemState[i].vel.x;
checkVarCPU[2] += cpuMesh.elemState[i].vel.y;
}

elementState* copyStateGPU = new elementState[cpuMesh.numElems];
toGPU = cudaMemcpy(copyStateGPU, cpuMesh.gpuElemState, sizeof(elementState)*cpuMesh.numElems, cudaMemcpyDeviceToHost);

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarGPU[p] = 0.0f;

for (unsigned i = 0; i < cpuMesh.numElems; ++i){
checkVarGPU[0] += copyStateGPU[i].h;
checkVarGPU[1] += copyStateGPU[i].vel.x;
checkVarGPU[2] += copyStateGPU[i].vel.y;
}

for (unsigned i = 0; i < cpuMesh.numElems; ++i)
cpuMesh.elemFlow[i].applyFluxes();

///////////////////////////////////////////////////////////////////// - */

/* ///////////////////////////////////////////////////////////////////// DEBUG UPDATE

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarCPU[p] = 0.0f;

for (unsigned i = 0; i < cpuMesh.numElems; ++i){
checkVarCPU[0] += cpuMesh.elemState[i].h;
checkVarCPU[1] += cpuMesh.elemState[i].vel.x;
checkVarCPU[2] += cpuMesh.elemState[i].vel.y;
}

toGPU = cudaMemcpy(copyStateGPU, cpuMesh.gpuElemState, sizeof(elementState)*cpuMesh.numElems, cudaMemcpyDeviceToHost);

for (unsigned p = 0; p < maxCONSERVED; ++p)
checkVarGPU[p] = 0.0f;

for (unsigned i = 0; i < cpuMesh.numElems; ++i){
checkVarGPU[0] += copyStateGPU[i].h;
checkVarGPU[1] += copyStateGPU[i].vel.x;
checkVarGPU[2] += copyStateGPU[i].vel.y;
}

for (unsigned i = 0; i < cpuMesh.numElems; ++i)
cpuMesh.elemFlow[i].applyCorrections();

///////////////////////////////////////////////////////////////////// - */
