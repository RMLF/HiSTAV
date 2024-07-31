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
#include <vector>
#include <algorithm>

// OpenMP
#include <omp.h>

// CUDA
#include <device_launch_parameters.h>

// STAV
#include "../headers/compile.hpp"
#include "../headers/common.hpp"
#include "../headers/control.hpp"
#include "../headers/gpuIO.hpp"
#include "../headers/gpuRun.hpp"
#include "../headers/boundaries.hpp"
#include "../headers/numerics.hpp"
#include "../headers/forcing.hpp"
#include "../headers/sediment.hpp"
#include "../headers/mesh.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


void physicsParameters::copyToGPU(){

	cudaError_t toGPU;
	toGPU = cudaMemcpyToSymbol(gpuPhysics, &cpuPhysics, sizeof(physicsParameters));

	if (toGPU != cudaSuccess){
		std::cout << "   -> CUDA Error @ physicsParameters::copyToGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void numericsParameters::copyToGPU(){

	cudaError_t toGPU;
	toGPU = cudaMemcpyToSymbol(gpuNumerics, &cpuNumerics, sizeof(numericsParameters));

	if (toGPU != cudaSuccess){
		std::cout << "   -> CUDA Error @ numericsParameters::copyToGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void bedParameters::copyToGPU(){

	cudaError_t toGPU;
	toGPU = cudaMemcpyToSymbol(gpuBed, &cpuBed, sizeof(bedParameters));

	if (toGPU != cudaSuccess){
		std::cout << "   -> CUDA Error @ bedParameters::copyToGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void forcingParameters::copyToGPU(){

	bool allDone = true;
	cudaError_t toGPU;

	toGPU = cudaMemcpyToSymbol(gpuForcing, &cpuForcing, sizeof(forcingParameters));
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	forcingParameters debug;
	toGPU = cudaMemcpyFromSymbol(&debug, gpuForcing, sizeof(forcingParameters));

	if (useRainfall){
		timeseries* gpuPointer;
		toGPU = cudaGetSymbolAddress((void**)&gpuPointer, gpuForcing.rainGauge);
		allDone = (toGPU == cudaSuccess) ? allDone : false;
		toGPU = cudaMalloc(&gpuPointer, sizeof(timeseries)*cpuForcing.numRainGauges);
		allDone = (toGPU == cudaSuccess) ? allDone : false;
		toGPU = cudaMemcpy(gpuPointer, &cpuForcing.rainGauge[0], sizeof(timeseries)*cpuForcing.numRainGauges, cudaMemcpyHostToDevice);
		allDone = (toGPU == cudaSuccess) ? allDone : false;
	}

	if (!allDone){
		std::cout << "   -> CUDA Error @ forcingParameters::copyToGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void controlParameters::copyToGPU(){

	physics->copyToGPU();
	numerics->copyToGPU();
	bed->copyToGPU();
	forcing->copyToGPU();
}

void simulationMesh::copyToGPU(){

	bool allDone = true;
	cudaError_t toGPU;

#	ifdef __STAV_MPI__
	int arrayLength = numElems - numConnectedElems;
#	else
	int arrayLength = numElems;
#	endif

	toGPU = cudaMalloc(&gpuElemState, sizeof(elementState)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemState, elemState, sizeof(elementState)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemScalars, sizeof(elementScalars)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemScalars, elemScalars, sizeof(elementScalars)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	arrayLength = numElems;

	toGPU = cudaMalloc(&gpuElemFlow, sizeof(elementFlow)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemFlow, elemFlow, sizeof(elementFlow)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemConnect, sizeof(elementConnect)*arrayLength*3);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemConnect, elemConnect, sizeof(elementConnect)*arrayLength*3, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemFluxes, sizeof(elementFluxes)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemFluxes, elemFluxes, sizeof(elementFluxes)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemBed, sizeof(elementBed)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemBed, elemBed, sizeof(elementBed)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemBedComp, sizeof(elementBedComp)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemBedComp, elemBedComp, sizeof(elementBedComp)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemForcing, sizeof(elementForcing)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemForcing, elemForcing, sizeof(elementForcing)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElem, sizeof(element)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElem, elem, sizeof(element)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElem, sizeof(element)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElem, elem, sizeof(element)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemDt, sizeof(double)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemDt, elemDt, sizeof(double)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

#	ifdef __STAV_MPI__
	arrayLength = numConnectedElems;

	toGPU = cudaMalloc(&gpuElemConnectedState, sizeof(elementState)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemConnectedState, elemConnectedState, sizeof(elementState)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemConnectedScalars, sizeof(elementScalars)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemConnectedScalars, elemConnectedScalars, sizeof(elementScalars)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	arrayLength = numRemoteElems;

	toGPU = cudaMalloc(&gpuElemRemoteState, sizeof(elementState)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemRemoteState, elemRemoteState, sizeof(elementState)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuElemRemoteScalars, sizeof(elementScalars)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemRemoteScalars, elemRemoteScalars, sizeof(elementScalars)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
#	endif

	int threads = 128;
	int blocks = (numElems + threads - 1) / threads;
	setStaticConnectivity <<< blocks, threads >>>
			(numElems, gpuElemDt, gpuElem, gpuElemFlow, gpuElemState, gpuElemScalars, gpuElemConnect, gpuElemFluxes, gpuElemBed, gpuElemBedComp, gpuElemForcing);

	std::vector<int> elemNeighbourConnectivity(numElems*6, -1);
#	ifdef __STAV_MPI__
	std::vector<int> elemDynamicConnectivity(numElems*2, -1);
#	endif

	for (unsigned i = 0; i < numElems; ++i){
		for (unsigned k = 0; k < 3; ++k)
			if (elem[i].meta->elem[k]){
				if (elem[i].meta->elem[k] >= &elem[0] && elem[i].meta->elem[k] <= &elem[numElems - 1]){
					elemNeighbourConnectivity[i*6 + k*2] = 0;
					elemNeighbourConnectivity[i*6 + k*2 + 1] = elem[i].meta->elem[k]->meta->ownerID;
				}
			}
#			ifdef __STAV_MPI__
			else if(&(elem[i].flow->flux->neighbor[k])){
				if (elem[i].flow->flux->neighbor[k].state >= &elemRemoteState[0] && elem[i].flow->flux->neighbor[k].state <= &elemRemoteState[numRemoteElems - 1]){
					elemNeighbourConnectivity[i*6 + k*2] = 1;
					elemNeighbourConnectivity[i*6 + k*2 + 1] = int(std::distance(&elemRemoteState[0], elem[i].flow->flux->neighbor[k].state));
				}
			}
#			endif
			else{
				elemNeighbourConnectivity[i*6 + k*2] = -1;
				elemNeighbourConnectivity[i*6 + k*2 + 1] = 0;
			}

#		ifdef __STAV_MPI__
		if (elem[i].flow->state >= &elemState[0] && elem[i].flow->state <= &elemState[numElems - 1]){
			elemDynamicConnectivity[i*2] = 0;
			elemDynamicConnectivity[i*2 + 1] = int(std::distance(&elemState[0], elem[i].flow->state));
		}else if (elem[i].flow->state >= &elemConnectedState[0] && elem[i].flow->state <= &elemConnectedState[numConnectedElems - 1]){
			elemDynamicConnectivity[i*2] = 1;
			elemDynamicConnectivity[i*2 + 1] = int(std::distance(&elemConnectedState[0], elem[i].flow->state));
		}else{
			elemDynamicConnectivity[i*2] = -1;
			elemDynamicConnectivity[i*2 + 1] = 0;
		}
#		endif
	}

	int* gpuElemNeighbourConnectivity;

	toGPU = cudaMalloc(&gpuElemNeighbourConnectivity, sizeof(int)*elemNeighbourConnectivity.size());
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemNeighbourConnectivity, &elemNeighbourConnectivity.front(), sizeof(int)*elemNeighbourConnectivity.size(), cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

#	ifdef __STAV_MPI__

	int* gpuElemDynamicConnectivity;

	toGPU = cudaMalloc(&gpuElemDynamicConnectivity, sizeof(int)*elemDynamicConnectivity.size());
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuElemDynamicConnectivity, &elemDynamicConnectivity.front(), sizeof(int)*elemDynamicConnectivity.size(), cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	setDynamicConnectivity <<< blocks, threads >>>
		(numElems, gpuElem, gpuElemState, gpuElemScalars, gpuElemConnectedState, gpuElemConnectedScalars, gpuElemDynamicConnectivity);
	
	setNeighbourConnectivity <<< blocks, threads >>> (numElems, gpuElem, gpuElemRemoteState, gpuElemRemoteScalars, gpuElemNeighbourConnectivity);

	toGPU = cudaFree(gpuElemDynamicConnectivity);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

#	else
	setNeighbourConnectivity <<< blocks, threads >>> (numElems, gpuElem, gpuElemState, gpuElemScalars, gpuElemNeighbourConnectivity);
#	endif

	toGPU = cudaFree(gpuElemNeighbourConnectivity);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	if (!allDone){
		std::cout << "   -> CUDA Error @ simulationMesh::copyToGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void simulationMesh::copyToCPU(){

	bool allDone = true;
	cudaError_t toGPU;

#	ifdef __STAV_MPI__
	int arrayLength = numElems - numConnectedElems;
#	else
	int arrayLength = numElems;
#	endif

	toGPU = cudaMemcpy(elemState, gpuElemState, sizeof(elementState)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemScalars, gpuElemScalars, sizeof(elementScalars)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	arrayLength = numElems;

	toGPU = cudaMemcpy(elemFlow, gpuElemFlow, sizeof(elementFlow)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemConnect, gpuElemConnect, sizeof(elementConnect)*arrayLength*3, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemFluxes, gpuElemFluxes, sizeof(elementFluxes)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemBed, gpuElemBed, sizeof(elementBed)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemBedComp, gpuElemBedComp, sizeof(elementBedComp)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemForcing, gpuElemForcing, sizeof(elementForcing)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elem, gpuElem, sizeof(element)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemDt, gpuElemDt, sizeof(double)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

#	ifdef __STAV_MPI__
	arrayLength = numConnectedElems;

	toGPU = cudaMemcpy(elemConnectedState, gpuElemState, sizeof(elementState)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemConnectedScalars, gpuElemScalars, sizeof(elementScalars)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	arrayLength = numRemoteElems;

	toGPU = cudaMemcpy(elemRemoteState, gpuElemState, sizeof(elementState)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMemcpy(elemRemoteScalars, gpuElemScalars, sizeof(elementScalars)*arrayLength, cudaMemcpyDeviceToHost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
#	endif

	if (!allDone){
		std::cout << "   -> CUDA Error @ simulationMesh::copyToCPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void simulationMesh::freeUpCPU(){

	for (unsigned i = 0; i < numElems; ++i){
		elem[i].flow->flux->neighbor = 0x0;
		elem[i].flow->flux->dt = 0x0;
		elem[i].flow->flux = 0x0;
		elem[i].forcing = 0x0;
#		ifndef __STAV_MPI__
		elem[i].meta = 0x0;
#		endif
	}

	delete[] elemConnect;
	delete[] elemDt;
	delete[] elemFluxes;
	delete[] elemForcing;
#	ifndef __STAV_MPI__
	delete[] elemMeta;
#	endif

	for (unsigned b = 0; b < cpuBoundaries.numBoundaries; ++b)
		cpuBoundaries.physical[b].elemGhost = 0x0;

	for (unsigned i = 0; i < cpuBoundaries.numElemGhosts; ++i){
		cpuBoundaries.elemGhost[i].link = 0x0;
		for (unsigned g = 0; g < 2; ++g){
			cpuBoundaries.elemGhost[i].hydroGauge[g] = 0x0;
			cpuBoundaries.elemGhost[i].sediGauge[g] = 0x0;
		}
	}

	delete[] cpuBoundaries.physical;
	delete[] cpuBoundaries.elemGhost;
	delete[] cpuBoundaries.hydroGauge;
	//delete[] cpuBoundaries.sediGauge;
}

void simulationMesh::deallocateFromGPU(){

	bool allDone = true;
	cudaError_t toGPU;

	toGPU = cudaFree(gpuElemFlow);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemState);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemScalars);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemConnect);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemFluxes);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemBed);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemBedComp);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemForcing);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElem);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemDt);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

#	ifdef __STAV_MPI__
	toGPU = cudaFree(gpuElemConnectedState);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemConnectedScalars);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemRemoteState);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuElemRemoteScalars);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
#	endif

	if (!allDone){
		std::cout << "   -> CUDA Error @ simulationMesh::deallocateFromGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void domainBoundaries::copyToGPU(){

	bool allDone = true;
	cudaError_t toGPU;

	if (numBoundaries < 1)
		return;

	int arrayLength = numElemGhosts;

	if (numElemGhosts > 0) {
		toGPU = cudaMalloc(&gpuElemGhost, sizeof(elementGhost)*arrayLength);
		allDone = (toGPU == cudaSuccess) ? allDone : false;
		toGPU = cudaMemcpy(gpuElemGhost, cpuBoundaries.elemGhost, sizeof(elementGhost)*arrayLength, cudaMemcpyHostToDevice);
		allDone = (toGPU == cudaSuccess) ? allDone : false;
	}

	arrayLength = numGauges;

	toGPU = cudaMalloc(&gpuHydroGauge, sizeof(timeseries)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuHydroGauge, cpuBoundaries.hydroGauge, sizeof(timeseries)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	/* toGPU = cudaMalloc(&gpuSediGauge, sizeof(timeseries)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuSediGauge, cpuBoundaries.sediGauge, sizeof(timeseries)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false; */

	arrayLength = numBoundaries;

	toGPU = cudaMalloc(&gpuPhysicalBoundaries, sizeof(physicalBoundary)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuPhysicalBoundaries, &cpuBoundaries.physical[0], sizeof(physicalBoundary)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuInletRefValue, sizeof(float)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuInletRefValue, &cpuInletRefValue.front(), sizeof(float)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaMalloc(&gpuInletRefFactor, sizeof(float)*arrayLength);
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuInletRefFactor, &cpuInletRefFactor.front(), sizeof(float)*arrayLength, cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	std::vector<int> boundaryConnectivity(numElemGhosts*6, -1);

	int debugVar = 0;

	for (unsigned i = 0; i < numElemGhosts; ++i){
		
		boundaryConnectivity[i*6] = int(std::distance(&cpuMesh.elem[0], cpuBoundaries.elemGhost[i].link));
		debugVar = boundaryConnectivity[i*6];
		debugVar = cpuBoundaries.elemGhost[i].link->meta->id;

		for (unsigned k = 0; k < 3; ++k)
			if (cpuBoundaries.elemGhost[i].link->flow->flux->neighbor[k].state == &cpuBoundaries.elemGhost[i].state){
				boundaryConnectivity[i*6 + 1] = k;
				break;
			}

		boundaryConnectivity[i*6 + 2] = int(std::distance(&cpuBoundaries.hydroGauge[0], cpuBoundaries.elemGhost[i].hydroGauge[0]));
		boundaryConnectivity[i*6 + 3] = int(std::distance(&cpuBoundaries.hydroGauge[0], cpuBoundaries.elemGhost[i].hydroGauge[1]));
		boundaryConnectivity[i*6 + 4] = int(std::distance(&cpuInletRefValue.front(), cpuBoundaries.elemGhost[i].inletRefValue));
		debugVar = boundaryConnectivity[i*6 + 4];
		boundaryConnectivity[i*6 + 5] = int(std::distance(&cpuInletRefFactor.front(), cpuBoundaries.elemGhost[i].inletRefFactor));
	}

	arrayLength = int(boundaryConnectivity.size());
	int* gpuBoundaryConnectivity;

	toGPU = cudaMalloc(&gpuBoundaryConnectivity, sizeof(int)*boundaryConnectivity.size());
	allDone = (toGPU == cudaSuccess) ? allDone : false;
	toGPU = cudaMemcpy(gpuBoundaryConnectivity, &boundaryConnectivity.front(), sizeof(int)*boundaryConnectivity.size(), cudaMemcpyHostToDevice);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	int threads = 128;
	int blocks = (numElemGhosts + threads - 1) / threads;

	setBoundaryConnectivity <<< blocks, threads >>>
		(numElemGhosts, numBoundaries, cpuMesh.gpuElem, gpuElemGhost, gpuHydroGauge, gpuSediGauge, gpuPhysicalBoundaries, gpuInletRefValue, gpuInletRefFactor, gpuBoundaryConnectivity);

	toGPU = cudaFree(gpuBoundaryConnectivity);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	if (!allDone){
		std::cout << "   -> CUDA Error @ domainBoundaries::copyToGPU()" << std::endl;
		exitOnKeypress(1);
	}
}

void domainBoundaries::deallocateFromGPU(){

	bool allDone = true;
	cudaError_t toGPU;

	toGPU = cudaFree(gpuElemGhost);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuHydroGauge);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	/*toGPU = cudaFree(gpuSediGauge);
	allDone = (toGPU == cudaSuccess) ? allDone : false;*/

	toGPU = cudaFree(gpuInletRefValue);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	toGPU = cudaFree(gpuInletRefFactor);
	allDone = (toGPU == cudaSuccess) ? allDone : false;

	if (!allDone){
		std::cout << "   -> CUDA Error @ domainBoundaries::deallocateFromGPU()" << std::endl;
		exitOnKeypress(1);
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// GPU Kernels for pointer setup
/////////////////////////////////////////////////////////////////////////////////////////////////


GLOBAL void setStaticConnectivity(int numElems, double* ptrElemDt, element* ptrElem, elementFlow* ptrElemFlow, elementState* ptrElemState, elementScalars* ptrElemScalars,
	elementConnect* ptrElemConnect, elementFluxes* ptrElemFluxes, elementBed* ptrElemBed, elementBedComp* ptrElemBedComp, elementForcing* ptrElemForcing){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numElems){

		ptrElemDt[i] = 999999999999.0;

		ptrElem[i].flow = &ptrElemFlow[i];

#		ifdef __STAV_MPI__
		ptrElem[i].flow->state = 0x0;
		ptrElem[i].flow->scalars = 0x0;
#		else
		ptrElem[i].flow->state = &ptrElemState[i];
		ptrElem[i].flow->scalars = &ptrElemScalars[i];
#		endif

		ptrElem[i].flow->flux = &ptrElemFluxes[i];
		ptrElem[i].flow->flux->neighbor = &ptrElemConnect[i * 3];
		ptrElem[i].flow->flux->dt = &ptrElemDt[i];

		ptrElem[i].bed = &ptrElemBed[i];
		ptrElem[i].bed->comp = &ptrElemBedComp[i];
		ptrElem[i].bed->flow = &ptrElemFlow[i];

		ptrElem[i].forcing = &ptrElemForcing[i];
	}
}

#ifdef __STAV_MPI__
GLOBAL void setDynamicConnectivity(int numElems, element* ptrElem, elementState* ptrElemState, elementScalars* ptrElemScalars, elementState* ptrElemConnectedState,
	elementScalars* ptrElemConnectedScalars, int* elemDynamicConnect){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numElems){
		
		int stateType = elemDynamicConnect[i*2];
		int stateIdx = elemDynamicConnect[i*2 + 1];

		if (stateType == 0){
			ptrElem[i].flow->state = &ptrElemState[stateIdx];
			ptrElem[i].flow->scalars = &ptrElemScalars[stateIdx];
		}else if (stateType == 1){
			ptrElem[i].flow->state = &ptrElemConnectedState[stateIdx];
			ptrElem[i].flow->scalars = &ptrElemConnectedScalars[stateIdx];
		}else{
			ptrElem[i].flow->state = 0x0;
			ptrElem[i].flow->scalars = 0x0;
		}
	}
}
#endif

GLOBAL void setNeighbourConnectivity(int numElems, element* ptrElem, elementState* ptrElemRemoteState, elementScalars* ptrElemRemoteScalars, int* elemNeighbourConnect){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numElems){

		for (unsigned k = 0; k < 3; ++k){
			
			int neighborType = elemNeighbourConnect[i*6 + k*2];
			int neighborIdx = elemNeighbourConnect[i*6 + k*2 + 1];

			if (neighborType == -1){
				ptrElem[i].flow->flux->neighbor[k].state = 0x0;
				ptrElem[i].flow->flux->neighbor[k].scalars = 0x0;
			}else if (neighborType == 0){
				ptrElem[i].flow->flux->neighbor[k].state = ptrElem[neighborIdx].flow->state;
				ptrElem[i].flow->flux->neighbor[k].scalars = ptrElem[neighborIdx].flow->scalars;
			}
#			ifdef __STAV_MPI__
			else if (neighborType == 1){
				ptrElem[i].flow->flux->neighbor[k].state = &ptrElemRemoteState[neighborIdx];
				ptrElem[i].flow->flux->neighbor[k].scalars = &ptrElemRemoteScalars[neighborIdx];
			}
#			endif
		}
	}
}

GLOBAL void setBoundaryConnectivity(int numElemGhosts, int numBoundaries, element* ptrElem, elementGhost* ptrElemGhost, timeseries* ptrHydroGauge, timeseries* ptrSediGauge,
	physicalBoundary* ptrPhysicalBoundaries, float* ptrRefValue, float* ptrRefFactor, int* boundaryConnect){

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < numElemGhosts) {

		ptrElemGhost[i].link = &ptrElem[boundaryConnect[i*6]];
		ptrElem[boundaryConnect[i*6]].flow->flux->neighbor[boundaryConnect[i*6 + 1]].state = &ptrElemGhost[i].state;
		ptrElem[boundaryConnect[i*6]].flow->flux->neighbor[boundaryConnect[i*6 + 1]].scalars = &ptrElemGhost[i].scalars;

		ptrElemGhost[i].hydroGauge[0] = &ptrHydroGauge[boundaryConnect[i*6 + 2]];
		ptrElemGhost[i].hydroGauge[1] = &ptrHydroGauge[boundaryConnect[i*6 + 3]];
		ptrElemGhost[i].inletRefValue = &ptrRefValue[boundaryConnect[i*6 + 4]];
		ptrElemGhost[i].inletRefFactor = &ptrRefFactor[boundaryConnect[i*6 + 5]];
	}

	int b = i;
	if (b < numBoundaries) {
		
		unsigned counterTotalGhosts = 0;
		for (unsigned j = 0; j < b; ++j)
			counterTotalGhosts += ptrPhysicalBoundaries[j].numElemGhosts;

		if (ptrPhysicalBoundaries[b].numElemGhosts > 0)
			ptrPhysicalBoundaries[b].elemGhost = &ptrElemGhost[counterTotalGhosts];

		ptrPhysicalBoundaries[b].inletRefValue = &ptrRefValue[b];
		ptrPhysicalBoundaries[b].inletRefFactor = &ptrRefFactor[b];
	}
}
