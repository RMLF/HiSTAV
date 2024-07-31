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

GLOBAL void reduceDtKernel(int arraySize, int blockSize, const double* gpuDtArray, double* minDtBlocks){

	extern __shared__ double sharedMem[];

	unsigned thread = threadIdx.x;
	unsigned dataIdx = blockIdx.x*blockDim.x + threadIdx.x;

	sharedMem[thread] = dataIdx < arraySize ? gpuDtArray[dataIdx] : 99999999999.0;

	__syncthreads();

	if (blockSize >= 1024){
		if (thread < 512)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 512]);
		__syncthreads();
	}

	if(blockSize >= 512){
		if(thread < 256)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 256]);
		__syncthreads();
	}

	if(blockSize >= 256){
		if(thread < 128)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 128]);
		__syncthreads();
	}

	if(blockSize >= 128){
		if(thread < 64)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 64]);
		__syncthreads();
	}

	if(thread < 32){
		if(blockSize >= 64)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 32]);
		if(blockSize >= 32)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 16]);
		if(blockSize >= 16)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 8]);
		if(blockSize >= 8)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 4]);
		if(blockSize >= 4)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 2]);
		if(blockSize >= 2)
			sharedMem[thread] = min(sharedMem[thread], sharedMem[thread + 1]);
	}

	if(thread == 0)
		minDtBlocks[blockIdx.x] = sharedMem[0];
}

double simulationSTAV2D::reduceDtOnGPU(){
	
	static int blockSize = 1024;
	static int sharedMemSize = sizeof(double)*blockSize;

	static int blocksLevel2, blocksLevel1, blocksLevel0;
	static double *gpuMinDtLevel2, *gpuMinDtLevel1, *gpuMinDtLevel0;
	
	if (step != 100){
		
		blocksLevel2 = (cpuMesh.numElems + blockSize - 1) / blockSize;
		blocksLevel1 = (blocksLevel2 + blockSize - 1) / blockSize;
		blocksLevel0 = (blocksLevel1 + blockSize - 1) / blockSize;

		if (blocksLevel0 != 1){
			std::cout << std::endl << " Error @ reduceDtOnGPU: mesh is too large, maximum mesh size is 1024^3!" << std::endl;
			exitOnKeypress(1);
		}

		cudaMalloc(&gpuMinDtLevel2, sizeof(double)*blocksLevel2);
		if (blocksLevel2 > 1)
			cudaMalloc(&gpuMinDtLevel1, sizeof(double)*blocksLevel1);
		if (blocksLevel1 > 1)
			cudaMalloc(&gpuMinDtLevel0, sizeof(double));
	}

	double minDt = 99999999999.0;

	reduceDtKernel <<< blocksLevel2, blockSize, sharedMemSize >> > (cpuMesh.numElems, blockSize, cpuMesh.gpuElemDt, gpuMinDtLevel2);
	if (blocksLevel2 > 1){
		reduceDtKernel <<< blocksLevel1, blockSize, sharedMemSize >>> (blocksLevel2, blockSize, gpuMinDtLevel2, gpuMinDtLevel1);
		if (blocksLevel1 > 1){
			reduceDtKernel <<< blocksLevel0, blockSize, sharedMemSize >>> (blocksLevel1, blockSize, gpuMinDtLevel1, gpuMinDtLevel0);
			cudaMemcpy(&minDt, gpuMinDtLevel0, sizeof(double), cudaMemcpyDeviceToHost);
		}else
			cudaMemcpy(&minDt, gpuMinDtLevel1, sizeof(double), cudaMemcpyDeviceToHost);
	}else
		cudaMemcpy(&minDt, gpuMinDtLevel2, sizeof(double), cudaMemcpyDeviceToHost);

	return minDt;
}

void simulationSTAV2D::runOnGPU(){

	std::cout << std::endl << std::endl;

	stepWClockTime = float(omp_get_wtime());
	totalWClockTime = float(omp_get_wtime());

	setBndRefValues.setKernel((void*) setBndRefValuesKernel, cpuBoundaries.numBoundaries);
	setBndConditions.setKernel((void*) setBndConditionsKernel, cpuBoundaries.numElemGhosts);

	getFluxes.setKernel((void*) getFluxesKernel, cpuMesh.numElems);
	applyFluxes.setKernel((void*) applyFluxesKernel, cpuMesh.numElems);
	applyCorrections.setKernel((void*) applyCorrectionsKernel, cpuMesh.numElems);
	applySources.setKernel((void*) applySourcesKernel, cpuMesh.numElems);

	while (currentTime <= finalTime) {

		if (cpuBoundaries.numBoundaries > 0){
			setBndRefValuesKernel <<< setBndRefValues.gridSize, setBndRefValues.blockSize >>> (gpuPhysicalBoundaries, cpuBoundaries.numBoundaries);
			setBndConditionsKernel <<< setBndConditions.gridSize, setBndConditions.blockSize >>> (gpuElemGhost, cpuBoundaries.numElemGhosts);
		}

		getFluxesKernel <<< getFluxes.gridSize, getFluxes.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);

		cpuNumerics.dt = reduceDtOnGPU();
		cudaMemcpyToSymbol(gpuNumerics.dt, &cpuNumerics.dt, sizeof(double));

		applyFluxesKernel <<< applyFluxes.gridSize, applyFluxes.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
		applyCorrectionsKernel <<< applyCorrections.gridSize, applyCorrections.blockSize >>> (cpuMesh.gpuElemFlow, cpuMesh.numElems);
		applySourcesKernel <<< applySources.gridSize, applySources.blockSize >>> (cpuMesh.gpuElem, cpuMesh.numElems);

		step++;
		currentTime += float(cpuNumerics.dt);
		cudaMemcpyToSymbol(gpuCurrentTime, &currentTime, sizeof(float));
		stepWClockTime = float(omp_get_wtime()) - stepWClockTime;

		std::cout << std::fixed << std::setprecision(6) << "  dt (s): " << cpuNumerics.dt;
		std::cout << std::fixed << std::setprecision(3) << ",  CPU (s): " << stepWClockTime << ",  Ratio: ";
		std::cout << cpuNumerics.dt / stepWClockTime << ",  Time (s): " << currentTime << ",  Comp. (%): ";
		std::cout << currentTime / finalTime * 100.0f << ",  CUDA: " << 1 << std::endl;

		stepWClockTime = float(omp_get_wtime() - stepWClockTime);
		stepWClockTime = float(omp_get_wtime());
		cpuNumerics.dt = 99999999999.0;

		cpuOutput.exportAll();
	}

	cpuOutput.exportMaxima();

	std::cout << std::endl;
	std::cout << std::fixed << std::setprecision(3) << "  Total Simulation Time (s): " << (totalWClockTime = float(omp_get_wtime()) - totalWClockTime) << std::endl;
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

		cpuNumerics.dt = reduceDtOnGPU();
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
