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
#include <vector>
#include <algorithm>

// OpenMP
#include <omp.h>

// STAV
#include "../headers/compile.hpp"
#include "../headers/control.hpp"
#include "../headers/geometry.hpp"
#include "../headers/mpiRun.hpp"
#include "../headers/numerics.hpp"
#include "../headers/mesh.hpp"
#include "../headers/sediment.hpp"
#include "../headers/boundaries.hpp"
#include "../headers/simulation.hpp"

// Definitions
#define deviceTestCPU 1
#define deviceTestGPU 2
#define deviceTestSTEPS 10

/////////////////////////////////////////////////////////////////////////////////////////////////


MPI_Comm GLOBALS;
MPI_Comm COMPUTE;

std::vector<unsigned /*short*/> globals(2, 0);
std::vector<float> chronometer;

mpiProcess::mpiProcess(){

	workCapacity = 0.0f;
	loadCapacity = 0.0f;

	cudaID = -1;

	rank = 0;
	worldSize = 0;

	firstElem = 0;
	lastElem = 0;
	messageCounter = 0;

	master = false;
	worker = false;
	hasGPU = false;

	reloadMesh = false;
	writeResults = false;
}

void mpiProcess::getRank(){

	MPI_Barrier(MPI_COMM_WORLD);

	int tempRank, tempWorldSize;
	MPI_Comm_rank(MPI_COMM_WORLD, &tempRank);
	MPI_Comm_size(MPI_COMM_WORLD, &tempWorldSize);

	rank = (unsigned /*short*/) tempRank;
	worldSize = (unsigned /*short*/) tempWorldSize;

	if (rank == 0){
		master = true;
		if (worldSize > 1){
			mySlaves.resize(worldSize - 1);
			chronometer.resize(worldSize, 0.0f);
			for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); ++pr)
				mySlaves[pr].rank = pr + 1;
		}
	}else if (rank > 0)
		worker = true;

	MPI_Barrier(MPI_COMM_WORLD);
}

void mpiProcess::getSlavesPerformance(){

	MPI_Barrier(MPI_COMM_WORLD);

	float sumTestTimeInverse = 0.0f;
	std::vector<float> testTimeInverse(worldSize, 0.0f);

	for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); pr++){

		unsigned start = deviceTestCPU;

		if(mySlaves[pr].hasGPU)
			start = deviceTestGPU;

		MPI_Ssend(&start, 1, MPI_UNSIGNED, mySlaves[pr].rank, mySlaves[pr].rank, MPI_COMM_WORLD);

		float testTime;
		MPI_Recv(&testTime, 1, MPI_FLOAT, mySlaves[pr].rank, mySlaves[pr].rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		testTimeInverse[pr] = 1.0f / testTime;
		sumTestTimeInverse += testTimeInverse[pr];
	}

	for (unsigned /*short*/ pr = 0; pr < worldSize; pr++)
		mySlaves[pr].workCapacity = testTimeInverse[pr] / sumTestTimeInverse;

	MPI_Barrier(MPI_COMM_WORLD);
}

void mpiProcess::runSlaveTest(){

	MPI_Barrier(MPI_COMM_WORLD);

	unsigned start;
	MPI_Recv(&start, 1, MPI_UNSIGNED, myProc.rank, myProc.rank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	std::vector<elementFlow> testFlow(1000000);
	std::vector<elementState> testState(1000000);
	std::vector<elementScalars> testScalars(1000000);
	std::vector<elementFluxes> testFluxes(1000000);
	std::vector<double> testDt(1000000);
	std::vector<elementConnect> testConnect(3000000);
	std::vector<elementState> testNeighState(3000000);
	std::vector<elementScalars> testNeighScalars(3000000);

	for (unsigned i = 0; i < testFlow.size(); ++i){

		std::srand(i*10 + 1);

		float random = float(std::rand() % 100) / 100.0f;
		float length = 10.0f * random;

		testFlow[i].state = &testState[i];
		testFlow[i].state->vel = vector2D((0.5f - random), -(0.5f - random));
		testFlow[i].state->h = std::max(0.01f, 2.0f + random);
		testFlow[i].state->z = -2.0f - random;

		testFlow[i].scalars = &testScalars[i];
		for (unsigned short s = 0; s < maxSCALARS; ++s)
			testFlow[i].scalars->specie[s] = random;

		for (unsigned short k = 0; k < 3; ++k){
			testNeighState[(i*3 + k)].vel = vector2D((0.5f - random*std::pow(-1.0f, float(k))), -(0.5f - random*std::pow(-1.0f, float(k))));
			testNeighState[(i*3 + k)].h = std::max(0.01f, 2.0f - random*std::pow(-1.0f, float(k)));
			testNeighState[(i*3 + k)].z = -2.0f + random*std::pow(-1.0f, float(k));
			for (unsigned short s = 0; s < maxSCALARS; ++s)
				testNeighScalars[(i*3 + k)].specie[s] = std::max(0.0f, random / float(k+1));
		}

		testFlow[i].flux = &testFluxes[i];
		testFlow[i].flux->dt = &testDt[i];
		testFlow[i].flux->neighbor = &testConnect[i * 3];

		for (unsigned k = 0; k < 3; ++k){
			testFlow[i].flux->neighbor[k].length = length + std::pow(-1.0f, float(k)) / float(k+1);
			testFlow[i].flux->neighbor[k].normal.x = random * float(k);
			testFlow[i].flux->neighbor[k].normal.y = -random * std::pow(-1.0f, float(k));
			testFlow[i].flux->neighbor[k].normal.normalize();

			testFlow[i].flux->neighbor[k].state = &testNeighState[(i*3 + k)];
			testFlow[i].flux->neighbor[k].scalars = &testNeighScalars[(i*3 + k)];
		}

		testFlow[i].area = std::sqrt(3.0f) * length * length / 4.0f;
	}

	float testTime = float(omp_get_wtime());

	if(start == deviceTestCPU){ // Slave node runs on CPU
#		pragma omp parallel
		{
			for (unsigned step = 0; step < deviceTestSTEPS; ++step){
#				pragma omp for schedule(static)
				for (unsigned i = 0; i < testFlow.size(); ++i)
					testFlow[i].computeFluxes_ReducedGudonov_1stOrder();
			}
		}
	}else if(start == deviceTestGPU){ // Slave node runs on GPU
#		ifdef __STAV_CUDA__
		// copy gpu vars
		// kernel gpu slave test
		// delete gpu vars
#		else

#		endif
	}

	testTime = float(omp_get_wtime()) - testTime;
	MPI_Ssend(&testTime, 1, MPI_FLOAT, 0, myProc.rank, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
}

mpiProcess myProc;
std::vector<mpiProcess> mySlaves;


elementConnected::elementConnected(){

	state = 0x0;
	scalars = 0x0;

	for (unsigned /*short*/ i = 0; i < bufferSIZE; ++i)
		messageBuffer[i] = 0.0f;

	neighbourOwner[0] = 0;
	neighbourOwner[1] = 0;
	neighbourOwner[2] = 0;

	id = 0;

	toSend[0] = false;
	toSend[1] = false;
	toSend[2] = false;
}

elementRemote::elementRemote(){

	state = 0x0;
	scalars = 0x0;

	for (unsigned /*short*/ i = 0; i < bufferSIZE; ++i)
		messageBuffer[i] = 0.0f;

	id = 0;
	owner = 0;
}

void simulationSTAV2D::checkBalance(){


}

void simulationSTAV2D::gatherResults(){

	MPI_Barrier(MPI_COMM_WORLD);

	if (myProc.master)
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elem[i].recvResults();
	else if (myProc.worker){
#		ifdef __STAV_CUDA__
		if(myProc.hasGPU){
			cudaMemcpy(cpuMesh.elemState, cpuMesh.gpuElemState, sizeof(elementState)*(cpuMesh.numElems-cpuMesh.numConnectedElems), cudaMemcpyDeviceToHost);
			cudaMemcpy(cpuMesh.elemConnectedState, cpuMesh.gpuElemConnectedState, sizeof(elementState)*cpuMesh.numConnectedElems, cudaMemcpyDeviceToHost);
			cudaMemcpy(cpuMesh.elemScalars, cpuMesh.gpuElemScalars, sizeof(elementScalars)*(cpuMesh.numElems-cpuMesh.numConnectedElems), cudaMemcpyDeviceToHost);
			cudaMemcpy(cpuMesh.elemConnectedScalars, cpuMesh.gpuElemConnectedScalars, sizeof(elementScalars)*cpuMesh.numConnectedElems, cudaMemcpyDeviceToHost);
			cudaMemcpy(cpuMesh.elemBedComp, cpuMesh.elemBedComp, sizeof(elementBedComp)*cpuMesh.numElems, cudaMemcpyDeviceToHost);
		}
#		endif
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elem[i].sendResults();
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void simulationSTAV2D::runMasterOnCPU(){

#	ifdef __STAV_MPI__

	MPI_Barrier(COMPUTE);
	MPI_Barrier(GLOBALS);

	stepWClockTime = float(omp_get_wtime());

	while (currentTime < finalTime){

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

		MPI_Allreduce(&cpuNumerics.dt, &cpuNumerics.dt, 1, MPI_DOUBLE, MPI_MIN, GLOBALS);

		currentTime += float(cpuNumerics.dt);
		step++;

		if (currentTime >= control->output->updateMaximaTime)
			myProc.writeResults = true;
		else if (currentTime >= control->output->writeOutputTime)
			myProc.writeResults = true;

		globals[0] = unsigned(myProc.writeResults);

		if (step % timerStepsSKIP == 0 && step > 0) {
			MPI_Gather(&chronometer.front(), 1, MPI_FLOAT, &chronometer.front(), 1, MPI_FLOAT, 0, GLOBALS);
			cpuSimulation.checkBalance();
		}

		globals[1] = unsigned(myProc.reloadMesh);

		MPI_Bcast(&globals.front(), int(globals.size()), MPI_UNSIGNED, 0, GLOBALS);
		MPI_Barrier(GLOBALS);

		if (myProc.writeResults || myProc.reloadMesh)
			gatherResults();

		if (myProc.writeResults){
			control->output->exportAll();
			myProc.writeResults = false;
		}

		if (myProc.reloadMesh){
			cpuMesh.reset();
			cpuMesh.scatterToSlaves();
			myProc.reloadMesh = false;
		}

		stepWClockTime = float(omp_get_wtime()) - stepWClockTime;
		std::cout << std::fixed << std::setprecision(6) << "  dt (s): " << cpuNumerics.dt;
		std::cout << std::fixed << std::setprecision(3) << ", CPU (s): " << stepWClockTime << ", R: " << cpuNumerics.dt / stepWClockTime;
		std::cout << std::fixed << std::setprecision(3) << ", T (s): " << currentTime << ", " << currentTime / finalTime*100.0 << "%, MPI: " << (myProc.worldSize - 1) << ", OMP: " << omp_get_max_threads() << std::endl;
		stepWClockTime = float(omp_get_wtime());
		cpuNumerics.dt = 99999999999.0;

		MPI_Barrier(COMPUTE);
	}

#	endif

}

void simulationSTAV2D::runSlaveOnCPU(){

#	ifdef __STAV_MPI__

	MPI_Barrier(COMPUTE);
	MPI_Barrier(GLOBALS);

	stepWClockTime = float(omp_get_wtime());

	while (currentTime < finalTime){

		double threadPrivateDt = 99999999999.0;

		if(step > 0){
			for (unsigned i = 0; i < cpuMesh.numRemoteElems; ++i)
				cpuMesh.elemRemote[i].recvState();
		}

		for (unsigned b = 0; b < cpuBoundaries.numBoundaries; ++b)
			if (cpuBoundaries.physical[b].isUniformInlet && cpuBoundaries.physical[b].numElemGhosts > 0)
				cpuBoundaries.physical[b].setRefValue();

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

		for (unsigned i = 0; i < cpuBoundaries.numElemGhosts; ++i)
			cpuBoundaries.elemGhost[i].getConditions();

		for (unsigned i = 0; i < cpuMesh.numElems; ++i){
			cpuMesh.elemFlow[i].computeFluxes();
			if (cpuMesh.elemDt[i] < threadPrivateDt)
				threadPrivateDt = cpuMesh.elemDt[i];
		}

		if (threadPrivateDt < cpuNumerics.dt)
			cpuNumerics.dt = threadPrivateDt;

		MPI_Allreduce(&cpuNumerics.dt, &cpuNumerics.dt, 1, MPI_DOUBLE, MPI_MIN, GLOBALS);

		step++;
		currentTime += float(cpuNumerics.dt);

		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elemFlow[i].applyFluxes();

		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elemFlow[i].applyCorrections();

		/*
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elem[i].applySourceTerms();
		*/

		if(currentTime < finalTime){
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

		cpuNumerics.dt = 99999999999.0;

		MPI_Barrier(COMPUTE);
	}

#	endif

}
