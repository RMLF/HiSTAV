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
#include <cstddef>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
#include <set>

// OpenMP
#include <omp.h>

// STAV
#include "../headers/compile.hpp"
#include "../headers/common.hpp"
#include "../headers/control.hpp"
#include "../headers/geometry.hpp"
#include "../headers/forcing.hpp"
#include "../headers/mpiIO.hpp"
#include "../headers/mpiRun.hpp"
#include "../headers/numerics.hpp"
#include "../headers/sediment.hpp"
#include "../headers/simulation.hpp"

// Definitions
#define bufferSIZE (maxCONSERVED + 2 + maxSCALARS)

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
// MPI-specific methods for already existing classes
/////////////////////////////////////////////////////////////////////////////////////////////////

#	ifdef __STAV_CUDA__
void controlParameters::readHPCControlFile(){

	std::ifstream controlFile;
	controlFile.open(controlFolder + controlHPCFileName);

	if (!controlFile.is_open() || !controlFile.good()){
		std::cerr << "   -> *Error* [H-1]: Could not open file " + controlFolder + controlHPCFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + controlFolder + controlHPCFileName << std::endl;

	unsigned /*short*/ numGPUs = 0;
	controlFile >> numGPUs;

	int /*short*/ cudaIDCounter = -1;
	for(unsigned pr = (mySlaves.size() - numGPUs); pr < mySlaves.size(); ++pr){
		mySlaves[pr].hasGPU = true;
		mySlaves[pr].cudaID = 0;//++cudaIDCounter;
	}

	controlFile.close();
}
#	endif

void controlParameters::bcast(){

	MPI_Barrier(MPI_COMM_WORLD);

	if (myProc.master){

		std::vector <float> timeControl = {cpuSimulation.initialTime, cpuSimulation.finalTime};
		MPI_Bcast(&timeControl.front(), int(timeControl.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);

		staticControlMPI staticControlToSend;
		staticControlToSend.copyFrom(*this, cpuBoundaries);
		MPI_Datatype controlCommMPI = staticControlToSend.createCommObj();
		MPI_Bcast(&staticControlToSend, 1, controlCommMPI, 0, MPI_COMM_WORLD);

		dynamicControlMPI dynamicControlToSend;
		dynamicControlToSend.copyFrom(*this, cpuBoundaries);
		unsigned dynamicControlDataSize = unsigned(dynamicControlToSend.allData.size());
		MPI_Bcast(&dynamicControlDataSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		MPI_Bcast(&dynamicControlToSend.allData.front(), int(dynamicControlToSend.allData.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);

		//myProc.getSlavesPerformance();

	}else if (myProc.worker){

		std::vector <float> timeControl(2, 0.0f);
		MPI_Bcast(&timeControl.front(), int(timeControl.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);

		cpuSimulation.initialTime = timeControl[0];
		cpuSimulation.currentTime = cpuSimulation.initialTime;
		cpuSimulation.finalTime = timeControl[1];

		staticControlMPI staticControlToRecv;
		MPI_Datatype controlCommMPI = staticControlToRecv.createCommObj();
		MPI_Bcast(&staticControlToRecv, 1, controlCommMPI, 0, MPI_COMM_WORLD);
		staticControlToRecv.copyTo(*this, cpuBoundaries);

		unsigned dynamicControlDataSize = 0;
		MPI_Bcast(&dynamicControlDataSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
		dynamicControlMPI dynamicControlToRecv(dynamicControlDataSize);
		MPI_Bcast(&dynamicControlToRecv.allData.front(), int(dynamicControlToRecv.allData.size()), MPI_FLOAT, 0, MPI_COMM_WORLD);
		dynamicControlToRecv.copyTo(*this, cpuBoundaries);

		//myProc.runSlaveTest();
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void element::sendToOwner(){

	elementMPI elemToSend;
	elemToSend.copyFrom(*this);
	MPI_Datatype elementCommMPI = elemToSend.createCommObj();
	MPI_Ssend(&elemToSend, 1, elementCommMPI, meta->owner, meta->id, MPI_COMM_WORLD);
}

void element::recvFromMaster(){

	elementMPI elemToRecv;
	MPI_Datatype elementCommMPI = elemToRecv.createCommObj();
	MPI_Recv(&elemToRecv, 1, elementCommMPI, 0, meta->id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	elemToRecv.copyTo(*this);
}

void element::sendResults(){

	std::vector<float> message(bufferSIZE + maxFRACTIONS);

	message[0] = flow->state->vel.x;
	message[1] = flow->state->vel.y;
	message[2] = flow->state->h;
	message[3] = flow->state->rho;
	message[4] = flow->state->z;

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		message[maxCONSERVED + 2 + s] = flow->scalars->specie[s];

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p)
		message[bufferSIZE + p] = bed->comp->bedPercentage[p];

	MPI_Send(&message.front(), int(message.size()), MPI_FLOAT, 0, meta->id, MPI_COMM_WORLD);
}

void element::recvResults(){

	std::vector<float> message(bufferSIZE + maxFRACTIONS);
	MPI_Recv(&message.front(), int(message.size()), MPI_FLOAT, meta->owner, meta->id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	flow->state->vel.x = message[0];
	flow->state->vel.y = message[1];
	flow->state->h = message[2];
	flow->state->rho = message[3];
	flow->state->z = message[4];

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		flow->scalars->specie[s] = message[maxCONSERVED + 2 + s];

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p)
		bed->comp->bedPercentage[p] = message[bufferSIZE + p];
}

void simulationMesh::scatterToSlaves(){

	MPI_Barrier(MPI_COMM_WORLD);

	// To Remove
	for (unsigned /*short*/ pr = 0; pr < myProc.worldSize; pr++)
		mySlaves[pr].workCapacity = 1.0f / float(myProc.worldSize);
	// To Remove

	int elemLowerBound = -1;
	int elemUpperBound = -1;

	std::vector< std::set<unsigned> > workerConnectedIdxs(mySlaves.size());
	std::vector< std::set<unsigned> > workerRemoteIdxs(mySlaves.size());

	for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); ++pr){

		mySlaves[pr].workCapacity = 1.0f / float(mySlaves.size());

		int load = unsigned(mySlaves[pr].workCapacity * float(numElems));
		elemLowerBound = std::max(0, std::min(elemUpperBound + 1, int(numElems) - 1));
		elemUpperBound = std::max(0, std::min(elemLowerBound + load, int(numElems) - 1));

		mySlaves[pr].firstElem = elemLowerBound;
		mySlaves[pr].lastElem = elemUpperBound;

		unsigned elemOwnerIDIncrementor = 0;

		for (unsigned i = mySlaves[pr].firstElem; i <= mySlaves[pr].lastElem; ++i){
			elem[i].meta->owner = (unsigned /*short*/) mySlaves[pr].rank;
			elem[i].meta->ownerID = elemOwnerIDIncrementor++;
		}
	}

	for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); ++pr)
		for (unsigned i = mySlaves[pr].firstElem; i <= mySlaves[pr].lastElem; ++i){
			bool allLocal = true;
			for (unsigned /*short*/ k = 0; k < 3; ++k)
				if (elem[i].meta->elem[k])
					if (elem[i].meta->elem[k]->meta->owner != elem[i].meta->owner)
						allLocal = false;
			if (!allLocal){
				elem[i].meta->isConnected = true;
				workerConnectedIdxs[pr].insert(elem[i].meta->id);
			}
		}

	for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); ++pr)
		for (unsigned i = mySlaves[pr].firstElem; i <= mySlaves[pr].lastElem; ++i)
			for (unsigned /*short*/ k = 0; k < 3; ++k)
				if (elem[i].meta->elem[k])
					if (elem[i].meta->elem[k]->meta->owner != mySlaves[pr].rank)
						workerRemoteIdxs[pr].insert(elem[i].meta->elem[k]->meta->id);

	exportAllProcessesMPI();
	std::cout << std::endl;

	for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); ++pr){

		std::cout << "   Sending mesh to process " << mySlaves[pr].rank << " ... ";

		std::vector<unsigned> uIntMessage(6, 0);
		uIntMessage[0] = mySlaves[pr].firstElem;
		uIntMessage[1] = mySlaves[pr].lastElem;
		uIntMessage[2] = unsigned(mySlaves[pr].cudaID);
		uIntMessage[3] = unsigned(mySlaves[pr].hasGPU);
		uIntMessage[4] = unsigned(workerConnectedIdxs[pr].size());
		uIntMessage[5] = unsigned(workerRemoteIdxs[pr].size());
		MPI_Ssend(&uIntMessage.front(), int(uIntMessage.size()), MPI_UNSIGNED, mySlaves[pr].rank, 0, MPI_COMM_WORLD);

		for (unsigned i = mySlaves[pr].firstElem; i <= mySlaves[pr].lastElem; ++i)
			elem[i].sendToOwner();

		unsigned remoteCounter = 0;
		for (std::set<unsigned>::iterator it = workerRemoteIdxs[pr].begin(); it != workerRemoteIdxs[pr].end(); ++it){

			unsigned idx = *it;

			std::vector<unsigned> IDs(2, 0);
			IDs[0] = elemMeta[idx].id;
			IDs[1] = elemMeta[idx].owner;

			MPI_Ssend(&IDs.front(), int(IDs.size()), MPI_UNSIGNED, mySlaves[pr].rank, 2*remoteCounter, MPI_COMM_WORLD);

			std::vector<float> remoteState(bufferSIZE, 0.0f);
			remoteState[0] = elemState[idx].vel.x;
			remoteState[1] = elemState[idx].vel.y;
			remoteState[2] = elemState[idx].h;
			remoteState[3] = elemState[idx].rho;
			remoteState[4] = elemState[idx].z;

			for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
				remoteState[maxCONSERVED + 2 + s] = elemScalars[idx].specie[s];

			MPI_Ssend(&remoteState.front(), int(remoteState.size()), MPI_FLOAT, mySlaves[pr].rank, (2*remoteCounter + 1), MPI_COMM_WORLD);
			++remoteCounter;
		}

		unsigned connectedCounter = 0;
		for (std::set<unsigned>::iterator it = workerConnectedIdxs[pr].begin(); it != workerConnectedIdxs[pr].end(); ++it){

			unsigned idx = *it;

			std::vector<unsigned> remoteFlags(7, 0);
			remoteFlags[3] = idx;

			std::set<unsigned /*short*/> neighbourProcess;

			for (unsigned /*short*/ k = 0; k < 3; ++k)
				if (elem[idx].meta->elem[k])
					if (elem[idx].meta->elem[k]->meta->owner != elem[idx].meta->owner)
						neighbourProcess.insert(elem[idx].meta->elem[k]->meta->owner);

			unsigned pr2 = 0;
			for (std::set<unsigned /*short*/>::iterator it = neighbourProcess.begin(); it != neighbourProcess.end(); ++it){
				remoteFlags[pr2] = *it;
				remoteFlags[4+pr2] = 1;
				++pr2;
			}

			MPI_Ssend(&remoteFlags.front(), int(remoteFlags.size()), MPI_UNSIGNED, mySlaves[pr].rank, connectedCounter, MPI_COMM_WORLD);
			++connectedCounter;
		}

		for (unsigned i = mySlaves[pr].firstElem; i <= mySlaves[pr].lastElem; ++i){
			std::vector<int> elemConnectivity(6, -1);
			for (unsigned /*short*/ k = 0; k < 3; ++k)
				if (elem[i].meta->elem[k]){
					if (elem[i].meta->elem[k]->meta->owner == mySlaves[pr].rank){
						elemConnectivity[k*2] = 0;
						elemConnectivity[k*2 + 1] = elem[i].meta->elem[k]->meta->ownerID;
					}else{
						elemConnectivity[k*2] = 1;
						elemConnectivity[k*2 + 1] = int(std::distance(workerRemoteIdxs[pr].begin(), workerRemoteIdxs[pr].find(elem[i].meta->elem[k]->meta->id)));
					}
				}
			MPI_Ssend(&elemConnectivity.front(), int(elemConnectivity.size()), MPI_INT, mySlaves[pr].rank, elem[i].meta->id, MPI_COMM_WORLD);
		}

		std::cout << " Done. " << std::endl;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (cpuBoundaries.numBoundaries){

		std::vector< std::vector<unsigned> > slavesNumElemGhosts(mySlaves.size(), std::vector<unsigned>(cpuBoundaries.numBoundaries, 0));
		for (unsigned /*short*/ b = 0; b < cpuBoundaries.numBoundaries; ++b)
			for (unsigned i = 0; i < cpuBoundaries.physical[b].numElemGhosts; ++i){
				unsigned elemGhostSlave = unsigned(cpuBoundaries.physical[b].elemGhost[i].link->meta->owner - 1);
				slavesNumElemGhosts[elemGhostSlave][b]++;
			}

		for (unsigned /*short*/ pr = 0; pr < mySlaves.size(); ++pr){
			MPI_Ssend(&slavesNumElemGhosts[pr].front(), int(slavesNumElemGhosts[pr].size()), MPI_UNSIGNED, mySlaves[pr].rank, 0, MPI_COMM_WORLD);
			for (unsigned /*short*/ b = 0; b < cpuBoundaries.numBoundaries; ++b){
				physicalBoundaryMPI boundaryToSend;
				boundaryToSend.copyFrom(cpuBoundaries.physical[b]);
				MPI_Datatype boundaryCommMPI = boundaryToSend.createCommObj();
				MPI_Ssend(&boundaryToSend, 1, boundaryCommMPI, mySlaves[pr].rank, b, MPI_COMM_WORLD);
				unsigned ghostCounter = 0;
				for (unsigned i = 0; i < cpuBoundaries.physical[b].numElemGhosts; ++i)
					if (cpuBoundaries.physical[b].elemGhost[i].link->meta->owner == mySlaves[pr].rank){
						elementGhostMPI elementGhostToSend;
						elementGhostToSend.copyFrom(cpuBoundaries.physical[b].elemGhost[i]);
						MPI_Datatype elementGhostCommMPI = elementGhostToSend.createCommObj();
						MPI_Ssend(&elementGhostToSend, 1, elementGhostCommMPI, mySlaves[pr].rank, ghostCounter++, MPI_COMM_WORLD);
					}
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void simulationMesh::recvFromMaster(){

	MPI_Barrier(MPI_COMM_WORLD);

	std::vector<unsigned> uIntMessage(6, 0);
	MPI_Recv(&uIntMessage.front(), int(uIntMessage.size()), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	myProc.firstElem = uIntMessage[0];
	myProc.lastElem = uIntMessage[1];
	myProc.cudaID = (unsigned /*short*/) uIntMessage[2];
	myProc.hasGPU = bool(uIntMessage[3]);
	numConnectedElems = uIntMessage[4];
	numRemoteElems = uIntMessage[5];

#	ifdef __STAV_CUDA__
	unsigned ok = 0;
	if(myProc.hasGPU){
	cudaError_t toGPU;
	toGPU = cudaSetDevice(myProc.cudaID);
	if(toGPU == cudaSuccess)
		ok = 1;
	}
#	endif

	numElems = myProc.lastElem - myProc.firstElem + 1;

	allocateOnCPU();

	for (unsigned i = 0; i < numElems; ++i){
		elem[i].meta->id = myProc.firstElem + i;
		elem[i].meta->ownerID = i;
		elem[i].meta->owner = myProc.rank;
	}

	for (unsigned i = 0; i < numElems; ++i)
		elem[i].recvFromMaster();

	unsigned remoteCounter = 0;
	for (unsigned i = 0; i < numRemoteElems; ++i){

		std::vector<unsigned> remoteIDs(2, 0);
		std::vector<float> remoteState(bufferSIZE, 0.0f);

		MPI_Recv(&remoteIDs.front(), int(remoteIDs.size()), MPI_UNSIGNED, 0, (2*remoteCounter), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		elemRemote[i].id = remoteIDs[0];
		elemRemote[i].owner = (unsigned /*short*/) remoteIDs[1];

		MPI_Recv(&remoteState.front(), int(remoteState.size()), MPI_FLOAT, 0, (2*remoteCounter + 1), MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		elemRemote[i].state->vel.x = remoteState[0];
		elemRemote[i].state->vel.y = remoteState[1];
		elemRemote[i].state->h = remoteState[2];
		elemRemote[i].state->rho = remoteState[3];
		elemRemote[i].state->z = remoteState[4];

		for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
			cpuMesh.elemRemote[i].scalars->specie[s] = remoteState[maxCONSERVED + 2 + s];

		++remoteCounter;
	}

	unsigned connectedCounter = 0;
	for (unsigned i = 0; i < numConnectedElems; ++i){

		std::vector<unsigned> remoteFlags(7, 0);
		MPI_Recv(&remoteFlags.front(), int(remoteFlags.size()), MPI_UNSIGNED, 0, connectedCounter, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		elemConnected[i].neighbourOwner[0] = remoteFlags[0];
		elemConnected[i].neighbourOwner[1] = remoteFlags[1];
		elemConnected[i].neighbourOwner[2] = remoteFlags[2];

		elemConnected[i].id = remoteFlags[3];

		elemConnected[i].toSend[0] = bool(remoteFlags[4]);
		elemConnected[i].toSend[1] = bool(remoteFlags[5]);
		elemConnected[i].toSend[2] = bool(remoteFlags[6]);

		++connectedCounter;
	}

	for (unsigned i = 0; i < numElems; ++i){

		std::vector<int> elemConnectivity(6, -1);
		MPI_Recv(&elemConnectivity.front(), int(elemConnectivity.size()), MPI_INT, 0, elem[i].meta->id, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (unsigned /*short*/ k = 0; k < 3; ++k)
			if (elemConnectivity[k*2] == -1){
				elem[i].meta->elem[k] = 0x0;
				elem[i].flow->flux->neighbor[k].state = 0x0;
				elem[i].flow->flux->neighbor[k].scalars = 0x0;
			}else if (elemConnectivity[k*2] == 0){
				unsigned idx = elemConnectivity[k*2 + 1];
				elem[i].meta->elem[k] = &elem[idx];
				elem[i].flow->flux->neighbor[k].state = elem[idx].flow->state;
				elem[i].flow->flux->neighbor[k].scalars = elem[idx].flow->scalars;
			}else if (elemConnectivity[k*2] == 1){
				unsigned idx = elemConnectivity[k*2 + 1];
				elem[i].meta->elem[k] = 0x0;
				elem[i].flow->flux->neighbor[k].state = elemRemote[idx].state;
				elem[i].flow->flux->neighbor[k].scalars = elemRemote[idx].scalars;
			}
	}

	MPI_Barrier(MPI_COMM_WORLD);

	int debugVar = cpuBoundaries.numBoundaries;

	if(cpuBoundaries.numBoundaries){

		std::vector<unsigned> myNumElemGhosts(cpuBoundaries.numBoundaries, 0);
		MPI_Recv(&myNumElemGhosts.front(), int(myNumElemGhosts.size()), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for (unsigned i = 0; i < myNumElemGhosts.size(); ++i)
			cpuBoundaries.numElemGhosts += myNumElemGhosts[i];

		debugVar = cpuBoundaries.numElemGhosts;

		cpuBoundaries.elemGhost = new elementGhost[cpuBoundaries.numElemGhosts];

		for (unsigned /*short*/ b = 0; b < cpuBoundaries.numBoundaries; ++b){
			debugVar = myNumElemGhosts[b];
			cpuBoundaries.physical[b].numElemGhosts = myNumElemGhosts[b];
			if(cpuBoundaries.physical[b].numElemGhosts > 0){
				unsigned counterTotalGhosts = 0;
				for (unsigned /*short*/ j = 0; j < b; ++j)
					counterTotalGhosts += cpuBoundaries.physical[j].numElemGhosts;
				debugVar = counterTotalGhosts;
				cpuBoundaries.physical[b].elemGhost = &cpuBoundaries.elemGhost[counterTotalGhosts];
			}
		}

		for (unsigned /*short*/ b = 0; b < cpuBoundaries.numBoundaries; ++b){
			physicalBoundaryMPI boundaryToRecv;
			MPI_Datatype boundaryCommMPI = boundaryToRecv.createCommObj();
			MPI_Recv(&boundaryToRecv, 1, boundaryCommMPI, 0, b, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			boundaryToRecv.copyTo(cpuBoundaries.physical[b]);
			for (unsigned i = 0; i < cpuBoundaries.physical[b].numElemGhosts; ++i){
				elementGhostMPI elementGhostToRecv;
				MPI_Datatype elementGhostCommMPI = elementGhostToRecv.createCommObj();
				MPI_Recv(&elementGhostToRecv, 1, elementGhostCommMPI, 0, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				elementGhostToRecv.copyTo(cpuBoundaries.physical[b].elemGhost[i]);
				cpuBoundaries.physical[b].elemGhost[i].inletRefValue = cpuBoundaries.physical[b].inletRefValue;
				cpuBoundaries.physical[b].elemGhost[i].inletRefFactor = cpuBoundaries.physical[b].inletRefFactor;
			}
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
}

void simulationMesh::reset(){

	if (myProc.master)
#		pragma omp parallel for
		for (unsigned i = 0; i < numElems; ++i){
			elem[i].meta->ownerID = 0;
			elem[i].meta->owner = 0;
			elem[i].meta->isConnected = false;
		}
	else if (myProc.worker){
		myProc.firstElem = 0;
		myProc.lastElem = 0;
		deallocateFromCPU();
#		ifdef __STAV_CUDA__
		if (myProc.hasGPU)
			deallocateFromGPU();
#		endif
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// MPI I/O communication classes, datatypes and (respective) constructors
/////////////////////////////////////////////////////////////////////////////////////////////////


staticControlMPI::staticControlMPI(){

	dt = 0.0;

	gravity = 0.0f;
	waterDensity = 0.0f;
	waterDinVis = 0.0f;
	waterTemp = 0.0f;

	CFL = 0.0f;
	minDepthFactor = 0.0f;

	maxConc = 0.0f;
	mobileBedTimeTrigger = 0.0f;

	numRainGauges = 0;

	frictionOption = 0;
	depEroOption = 0;
	adaptLenOption = 0;

	numBoundaries = 0;
	numBndGauges = 0;

	useFriction = false;
	useMobileBed = false;
	useRainfall = false;
}

MPI_Datatype staticControlMPI::createCommObj(){

	int blockLengths[] = { 1, 8, 9 };

	MPI_Aint blockOffsets[] = { 0, 0, 0 };
	blockOffsets[0] = offsetof(staticControlMPI, dt);
	blockOffsets[1] = offsetof(staticControlMPI, gravity);
	blockOffsets[2] = offsetof(staticControlMPI, numRainGauges);

	MPI_Datatype blockTypes[] = { MPI_DOUBLE, MPI_FLOAT, MPI_UNSIGNED/*_SHORT*/};

	MPI_Datatype newControlMPIStruct;
	MPI_Type_create_struct(3, blockLengths, blockOffsets, blockTypes, &newControlMPIStruct);
	MPI_Type_commit(&newControlMPIStruct);

	return newControlMPIStruct;
}

void staticControlMPI::copyFrom(const controlParameters& control, const domainBoundaries& boundaries){

	dt = control.numerics->dt;

	gravity = control.physics->gravity;
	waterDensity = control.physics->waterDensity;
	waterDinVis = control.physics->waterDinVis;
	waterTemp = control.physics->waterTemp;

	CFL = control.numerics->CFL;
	minDepthFactor = control.numerics->minDepthFactor;

	maxConc = control.bed->maxConc;
	mobileBedTimeTrigger = control.bed->mobileBedTimeTrigger;

	numRainGauges = control.forcing->numRainGauges;

	frictionOption = control.bed->frictionOption;
	depEroOption = control.bed->depEroOption;
	adaptLenOption = control.bed->adaptLenOption;

	numBndGauges = boundaries.numGauges;
	numBoundaries = boundaries.numBoundaries;

	useFriction = (unsigned /*short*/) control.bed->useFriction;
	useMobileBed = (unsigned /*short*/) control.bed->useMobileBed;
	useRainfall = (unsigned /*short*/) control.forcing->useRainfall;
}

void staticControlMPI::copyTo(controlParameters& control, domainBoundaries& boundaries){

	control.numerics->dt = dt;

	control.physics->gravity = gravity;
	control.physics->waterDensity = waterDensity;
	control.physics->waterDinVis = waterDinVis;
	control.physics->waterTemp = waterTemp;

	control.numerics->CFL = CFL;
	control.numerics->minDepthFactor = minDepthFactor;

	control.bed->maxConc = maxConc;
	control.bed->mobileBedTimeTrigger = mobileBedTimeTrigger;

	control.forcing->numRainGauges = numRainGauges;
	control.forcing->rainGauge = new timeseries[numRainGauges];

	control.bed->frictionOption = frictionOption;
	control.bed->depEroOption = depEroOption;
	control.bed->adaptLenOption = adaptLenOption;

	boundaries.numBoundaries = numBoundaries;
	boundaries.physical = new physicalBoundary[numBoundaries];

	boundaries.numGauges = numBndGauges;
	boundaries.hydroGauge = new timeseries[numBndGauges];
	boundaries.sediGauge = new timeseries[numBndGauges];

	control.bed->useFriction = bool(useFriction);
	control.bed->useMobileBed = bool(useMobileBed);
	control.forcing->useRainfall = bool(useRainfall);

	cpuInletRefValue.resize(boundaries.numBoundaries, 0.0f);
	cpuInletRefFactor.resize(boundaries.numBoundaries, 0.0f);
	cpuInletRefValueBuffer.resize(boundaries.numBoundaries*myProc.worldSize, 0.0f);
	cpuInletRefFactorBuffer.resize(boundaries.numBoundaries*myProc.worldSize, 0.0f);

	for (unsigned b = 0; b < boundaries.numBoundaries; ++b){
		boundaries.physical[b].inletRefValue = &cpuInletRefValue[b];
		boundaries.physical[b].inletRefFactor = &cpuInletRefFactor[b];
	}
}

dynamicControlMPI::dynamicControlMPI(){}

dynamicControlMPI::dynamicControlMPI(const unsigned dynamicControlDataSize){
	allData.resize(dynamicControlDataSize, 0.0f);
}

void dynamicControlMPI::copyFrom(const controlParameters& control, const domainBoundaries& boundaries){

	unsigned /*short*/ numRainGauges = control.forcing->numRainGauges;
	unsigned /*short*/ numHydroGauges = boundaries.numGauges;

	std::vector<float> allSeriesTimes(timeseriesMAX * (numHydroGauges + numRainGauges), 0.0f);
	std::vector<float> allSeriesValues(timeseriesMAX * (numHydroGauges + numRainGauges), 0.0f);
	std::vector<float> allGrains(maxFRACTIONS * 12, 0.0f);

	for (unsigned /*short*/ g = 0; g < numHydroGauges; ++g)
		for (unsigned t = 0; t < timeseriesMAX; ++t){
			allSeriesTimes[g * timeseriesMAX + t] = boundaries.hydroGauge[g].time[t];
			allSeriesValues[g * timeseriesMAX + t] = boundaries.hydroGauge[g].data[t];
		}

	for (unsigned /*short*/ g = 0; g < numRainGauges; ++g)
		for (unsigned t = 0; t < timeseriesMAX; ++t){
			allSeriesTimes[(g + numHydroGauges) * timeseriesMAX + t] = control.forcing->rainGauge[g].time[t];
			allSeriesValues[(g + numHydroGauges) * timeseriesMAX + t] = control.forcing->rainGauge[g].data[t];
		}

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		allGrains[p*12 + 0] = float(control.bed->grain[p].color);
		allGrains[p*12 + 1] = control.bed->grain[p].diam;
		allGrains[p*12 + 2] = control.bed->grain[p].specGrav;
		allGrains[p*12 + 3] = control.bed->grain[p].poro;
		allGrains[p*12 + 4] = control.bed->grain[p].rest;
		allGrains[p*12 + 5] = control.bed->grain[p].alpha;
		allGrains[p*12 + 6] = control.bed->grain[p].beta;
		allGrains[p*12 + 7] = control.bed->grain[p].tanPhi;
		allGrains[p*12 + 8] = control.bed->grain[p].adaptLenMinMult;
		allGrains[p*12 + 9] = control.bed->grain[p].adaptLenMaxMult;
		allGrains[p*12 + 10] = control.bed->grain[p].adaptLenRefShields;
		allGrains[p*12 + 11] = control.bed->grain[p].adaptLenShapeFactor;
	}

	allData.insert(allData.end(), allSeriesTimes.begin(), allSeriesTimes.end());
	allData.insert(allData.end(), allSeriesValues.begin(), allSeriesValues.end());
	allData.insert(allData.end(), allGrains.begin(), allGrains.end());
}

void dynamicControlMPI::copyTo(controlParameters& control, domainBoundaries& boundaries){

	unsigned /*short*/ numRainGauges = control.forcing->numRainGauges;
	unsigned /*short*/ numHydroGauges = boundaries.numGauges;

	unsigned gaugeDataLength = timeseriesMAX * (unsigned(numHydroGauges) + unsigned(numRainGauges));
	unsigned /*short*/ grainDataLength = maxFRACTIONS * 12;

	std::vector<float> allSeriesTimes(gaugeDataLength, 0.0f);
	for (unsigned t = 0; t < gaugeDataLength; ++t)
		allSeriesTimes[t] = allData[t];

	std::vector<float> allSeriesValues(gaugeDataLength, 0.0f);
	for (unsigned t = 0; t < gaugeDataLength; ++t)
		allSeriesValues[t] = allData[gaugeDataLength + t];

	std::vector<float> allGrains(gaugeDataLength, 0.0f);
	for (unsigned /*short*/ p = 0; p < grainDataLength; ++p)
		allGrains[p] = allData[gaugeDataLength*2 + p];

	for (unsigned /*short*/ g = 0; g < numHydroGauges; ++g)
		for (unsigned t = 0; t < timeseriesMAX; ++t){
			float time = allSeriesTimes[g * timeseriesMAX + t];
			float data = allSeriesValues[g * timeseriesMAX + t];
			boundaries.hydroGauge[g].addData(time, data);
		}

	for (unsigned /*short*/ g = 0; g < numRainGauges; ++g)
		for (unsigned t = 0; t < timeseriesMAX; ++t){
			float time =  allSeriesTimes[(g + numHydroGauges) * timeseriesMAX + t];
			float data = allSeriesValues[(g + numHydroGauges) * timeseriesMAX + t];
			control.forcing->rainGauge[g].addData(time, data);
		}

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		control.bed->grain[p].color = (unsigned /*short*/) allGrains[p*12 + 0];
		control.bed->grain[p].diam = allGrains[p*12 + 1];
		control.bed->grain[p].specGrav = allGrains[p*12 + 2];
		control.bed->grain[p].poro = allGrains[p*12 + 3];
		control.bed->grain[p].rest = allGrains[p*12 + 4];
		control.bed->grain[p].alpha = allGrains[p*12 + 5];
		control.bed->grain[p].beta = allGrains[p*12 + 6];
		control.bed->grain[p].tanPhi = allGrains[p*12 + 7];
		control.bed->grain[p].adaptLenMinMult = allGrains[p*12 + 8];
		control.bed->grain[p].adaptLenMaxMult = allGrains[p*12 + 9];
		control.bed->grain[p].adaptLenRefShields = allGrains[p*12 + 10];
		control.bed->grain[p].adaptLenShapeFactor = allGrains[p*12 + 11];
	}
}

elementMPI::elementMPI(){

	for (unsigned /*short*/ i = 0; i < (maxCONSERVED + 2 + maxSCALARS); ++i)
		state[i] = 0.0f;

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		bedComp[p] = 0.0f;
		subComp[p] = 0.0f;
	}

	for (unsigned /*short*/ k = 0; k < 14; ++k)
		geometry[k] = 0.0f;

	connectFlag[0] = 0;

	for (unsigned /*short*/ nn = 0; nn < maxNN; ++nn)
		rainGaugeIdx[nn] = 0;
}

MPI_Datatype elementMPI::createCommObj(){

	int blockLengths[] = { (maxCONSERVED + 2 + maxSCALARS) + 2*maxFRACTIONS + 14, 1 + maxNN };

	MPI_Aint blockOffsets[] = { 0, 0, 0 };
	blockOffsets[0] = offsetof(elementMPI, state);
	blockOffsets[1] = offsetof(elementMPI, connectFlag);

	MPI_Datatype blockTypes[] = { MPI_FLOAT, MPI_INT};

	MPI_Datatype newElementCommMPIStruct;
	MPI_Type_create_struct(2, blockLengths, blockOffsets, blockTypes, &newElementCommMPIStruct);
	MPI_Type_commit(&newElementCommMPIStruct);

	return newElementCommMPIStruct;
}

void elementMPI::copyFrom(const element& elem){

	state[0] = elem.flow->state->vel.x;
	state[1] = elem.flow->state->vel.y;
	state[2] = elem.flow->state->h;
	state[3] = elem.flow->state->rho;
	state[4] = elem.flow->state->z;

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		state[maxCONSERVED + 2 + s] = elem.flow->scalars->specie[s];

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		bedComp[p] = elem.bed->comp->bedPercentage[p];
		subComp[p] = elem.bed->comp->subPercentage[p];
	}

	for (unsigned /*short*/ k = 0; k < 3; ++k){
		geometry[k*3] = elem.flow->flux->neighbor[k].length;
		geometry[k*3 + 1] = elem.flow->flux->neighbor[k].normal.x;
		geometry[k*3 + 2] = elem.flow->flux->neighbor[k].normal.y;
	}

	geometry[9] = elem.flow->area;

	geometry[10] = elem.bed->slope.x;
	geometry[11] = elem.bed->slope.y;

	geometry[12] = elem.meta->center.x;
	geometry[13] = elem.meta->center.y;

	if (elem.meta->isConnected)
		connectFlag[0] = 1;
	else
		connectFlag[0] = 0;

	if (cpuForcing.numRainGauges > 0 && cpuForcing.useRainfall)
		for (unsigned /*short*/ nn = 0; nn < maxNN; ++nn){
			int idx = int(std::distance(&cpuForcing.rainGauge[0], elem.forcing->rainGauge[nn]));
			if (idx >= 0 && idx <= int(cpuForcing.numRainGauges))
				rainGaugeIdx[nn] = idx;
			else
				rainGaugeIdx[nn] = -1;
		}
}

void elementMPI::copyTo(element& elem){

	static unsigned localCounter = 0;
	static unsigned connectedCounter = 0;

	elem.meta->isConnected = bool(connectFlag[0]);

	if (elem.meta->isConnected){
		elem.flow->state = &cpuMesh.elemConnectedState[connectedCounter];
		elem.flow->scalars = &cpuMesh.elemConnectedScalars[connectedCounter];
		++connectedCounter;
	}else{
		elem.flow->state = &cpuMesh.elemState[localCounter];
		elem.flow->scalars = &cpuMesh.elemScalars[localCounter];
		++localCounter;
	}

	elem.flow->state->vel.x = state[0];
	elem.flow->state->vel.y = state[1];
	elem.flow->state->h = state[2];
	elem.flow->state->rho = state[3];
	elem.flow->state->z = state[4];

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		elem.flow->scalars->specie[s] = state[maxCONSERVED + 2 + s];

	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		elem.bed->comp->bedPercentage[p] = bedComp[p];
		elem.bed->comp->subPercentage[p] = subComp[p];
	}

	for (unsigned /*short*/ k = 0; k < 3; ++k){
		elem.flow->flux->neighbor[k].length = geometry[k*3];
		elem.flow->flux->neighbor[k].normal.x = geometry[k*3 + 1];
		elem.flow->flux->neighbor[k].normal.y = geometry[k*3 + 2];
	}

	elem.flow->area = geometry[9];

	elem.bed->slope.x = geometry[10];
	elem.bed->slope.y = geometry[11];

	elem.meta->center.x = geometry[12];
	elem.meta->center.y = geometry[13];

	if (cpuForcing.numRainGauges > 0 && cpuForcing.useRainfall)
		for (unsigned /*short*/ nn = 0; nn < maxNN; ++nn)
			if (rainGaugeIdx[nn] != -1)
				elem.forcing->rainGauge[nn] = &cpuForcing.rainGauge[rainGaugeIdx[nn]];
}

physicalBoundaryMPI::physicalBoundaryMPI(){

	geometry[0] = { 0.0f };

	for (unsigned /*short*/ t = 0; t < maxDISCHCURLENGTH * 2; ++t)
		inletCurve[t] = 0.0f;

	flags[0] = { 0 };
}

MPI_Datatype physicalBoundaryMPI::createCommObj(){

	int blockLengths[] = { 2 + 2*maxDISCHCURLENGTH, 5 };

	MPI_Aint blockOffsets[] = { 0, 0 };
	blockOffsets[0] = offsetof(physicalBoundaryMPI, geometry);
	blockOffsets[1] = offsetof(physicalBoundaryMPI, flags);

	MPI_Datatype blockTypes[] = { MPI_FLOAT, MPI_UNSIGNED };

	MPI_Datatype newPhysicalBoundaryMPI;
	MPI_Type_create_struct(2, blockLengths, blockOffsets, blockTypes, &newPhysicalBoundaryMPI);
	MPI_Type_commit(&newPhysicalBoundaryMPI);

	return newPhysicalBoundaryMPI;
}

void physicalBoundaryMPI::copyFrom(const physicalBoundary& boundary){

	geometry[0] = boundary.referenceSlope;
	geometry[1] = boundary.bufferRadious;

	for (unsigned /*short*/ t = 0; t < maxDISCHCURLENGTH; ++t){
		inletCurve[t] = boundary.inletCurveLevel[t];
		inletCurve[maxDISCHCURLENGTH + t] = boundary.inletCurveDischarge[t];
	}

	flags[0] = unsigned(boundary.numGauges);
	flags[1] = unsigned(boundary.conditionType);
	flags[2] = unsigned(boundary.id);

	flags[3] = unsigned(boundary.isAtEquilibrium);
	flags[4] = unsigned(boundary.isUniformInlet);
}

void physicalBoundaryMPI::copyTo(physicalBoundary& boundary){

	boundary.referenceSlope = geometry[0];
	boundary.bufferRadious = geometry[1];

	for (unsigned /*short*/ t = 0; t < maxDISCHCURLENGTH; ++t){
		boundary.inletCurveLevel[t] = inletCurve[t];
		boundary.inletCurveDischarge[t] = inletCurve[maxDISCHCURLENGTH + t];
	}

	boundary.numGauges = (unsigned /*short*/) flags[0];
	boundary.conditionType = (unsigned /*short*/) flags[1];
	boundary.id = (unsigned /*short*/) flags[2];

	boundary.isAtEquilibrium = bool(flags[3]);
	boundary.isUniformInlet = bool(flags[4]);
}

elementGhostMPI::elementGhostMPI(){

	geometry[0] = { 0.0f };
	gaugeWeights[0] = { 0.0f };

	gaugeIdxs[0] = { 0 };
	flags[0] = { 0 };
}

MPI_Datatype elementGhostMPI::createCommObj(){

	int blockLengths[] = { 5, 6 };

	MPI_Aint blockOffsets[] = { 0, 0 };
	blockOffsets[0] = offsetof(elementGhostMPI, geometry);
	blockOffsets[1] = offsetof(elementGhostMPI, gaugeIdxs);

	MPI_Datatype blockTypes[] = { MPI_FLOAT, MPI_UNSIGNED };

	MPI_Datatype newElementGhostMPI;
	MPI_Type_create_struct(2, blockLengths, blockOffsets, blockTypes, &newElementGhostMPI);
	MPI_Type_commit(&newElementGhostMPI);

	return newElementGhostMPI;
}

void elementGhostMPI::copyFrom(const elementGhost& elemGhost){

	geometry[0] = elemGhost.normal.x;
	geometry[1] = elemGhost.normal.y;
	geometry[2] = elemGhost.edgeLength;

	gaugeWeights[0] = elemGhost.gaugeWeight[0];
	gaugeWeights[1] = elemGhost.gaugeWeight[1];

	gaugeIdxs[0] = unsigned(std::distance(&cpuBoundaries.hydroGauge[0], elemGhost.hydroGauge[0]));
	gaugeIdxs[1] = unsigned(std::distance(&cpuBoundaries.hydroGauge[0], elemGhost.hydroGauge[1]));

	flags[0] = elemGhost.link->meta->ownerID;

	for (unsigned /*short*/ k = 0; k < 3; ++k)
		if (&(elemGhost.link->flow->flux->neighbor[k]))
			if (elemGhost.link->flow->flux->neighbor[k].state == &elemGhost.state)
				flags[1] = k;

	flags[2] = unsigned(elemGhost.conditionType);
	flags[3] = unsigned(elemGhost.isAtEquilibrium);
}

void elementGhostMPI::copyTo(elementGhost& elemGhost){

	elemGhost.normal.x = geometry[0];
	elemGhost.normal.y = geometry[1];
	elemGhost.edgeLength = geometry[2];

	elemGhost.gaugeWeight[0] = gaugeWeights[0];
	elemGhost.gaugeWeight[1] = gaugeWeights[1];

	elemGhost.hydroGauge[0] = &cpuBoundaries.hydroGauge[gaugeIdxs[0]];
	elemGhost.hydroGauge[1] = &cpuBoundaries.hydroGauge[gaugeIdxs[1]];

	int debugVar = flags[0];
	debugVar = flags[1];

	elemGhost.link = &cpuMesh.elem[flags[0]];
	elemGhost.link->flow->flux->neighbor[flags[1]].state = &elemGhost.state;
	elemGhost.link->flow->flux->neighbor[flags[1]].scalars = &elemGhost.scalars;

	elemGhost.conditionType = (unsigned /*short*/) flags[2];
	elemGhost.isAtEquilibrium = bool(flags[3]);
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void exportAllProcessesMPI(){

	std::ofstream mpiProcessesOut;
	std::ostringstream mpiProcessesOutSstream;
	mpiProcessesOutSstream << "./output/control/mpiProcesses-" << cpuSimulation.step << ".vtk";
	std::string mpiProcessesOutFile = mpiProcessesOutSstream.str();

	std::cout << "   Writing " << mpiProcessesOutFile.c_str() << " ... ";

	mpiProcessesOut.open(mpiProcessesOutFile.c_str());
	mpiProcessesOut << "# vtk DataFile Version 2.0" << std::endl;
	mpiProcessesOut << "STAV-2D MPI distribution at " << cpuSimulation.currentTime << " secs." << std::endl;
	mpiProcessesOut << "ASCII" << std::endl;
	mpiProcessesOut << "DATASET UNSTRUCTURED_GRID" << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "POINTS " << cpuMesh.numNodes << " FLOAT" << std::endl;

	for (unsigned i = 0; i < cpuMesh.numNodes; ++i){
		mpiProcessesOut << std::fixed << cpuMesh.elemNode[i].coord.x << "	" << std::fixed << cpuMesh.elemNode[i].coord.y << "	" << std::fixed << 0.0f << std::endl;
	}

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "CELLS " << cpuMesh.numElems << "	" << cpuMesh.numElems * 4 << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		mpiProcessesOut << 3 << "	" << cpuMesh.elem[i].meta->node[0]->id << "	" << cpuMesh.elem[i].meta->node[1]->id << "	" << cpuMesh.elem[i].meta->node[2]->id << std::endl;
	}

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "CELL_TYPES " << cpuMesh.numElems << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << 5 << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "CELL_DATA " << cpuMesh.numElems << std::endl;
	mpiProcessesOut << "SCALARS Owner INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->owner << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS OwnerID INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->ownerID << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS GlobalID INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->id << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS isConnected INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << int(cpuMesh.elem[i].meta->isConnected) << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS Area FLOAT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].flow->area << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS EdgeLengthSum FLOAT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		float edgeLengthSum = 0.0f;
		for (unsigned /*short*/ k = 0; k < 3; ++k)
			edgeLengthSum += cpuMesh.elem[i].flow->flux->neighbor[k].length;
		mpiProcessesOut << std::fixed << edgeLengthSum << std::endl;
	}

	mpiProcessesOut.close();
	std::cout << " Done. " << std::endl;
}

void exportCurrentProcessMPI(){

	std::ofstream mpiProcessesOut;
	std::ostringstream mpiProcessesOutSstream;
	mpiProcessesOutSstream << "./output/control/mpiCurrentProcess-" << myProc.rank << ".vtk";
	std::string mpiProcessesOutFile = mpiProcessesOutSstream.str();

	std::cout << "   Writing " << mpiProcessesOutFile.c_str() << " ... ";

	mpiProcessesOut.open(mpiProcessesOutFile.c_str());
	mpiProcessesOut << "# vtk DataFile Version 2.0" << std::endl;
	mpiProcessesOut << "STAV-2D MPI distribution at " << cpuSimulation.currentTime << " secs." << std::endl;
	mpiProcessesOut << "ASCII" << std::endl;
	mpiProcessesOut << "DATASET UNSTRUCTURED_GRID" << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "POINTS " << cpuMesh.numElems << " FLOAT" << std::endl;

	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->center.x << "	" << std::fixed << cpuMesh.elem[i].meta->center.y << "	" << std::fixed << 0.0f << std::endl;
	}

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "POINT_DATA " << cpuMesh.numElems << std::endl;
	mpiProcessesOut << "SCALARS Owner INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->owner << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS OwnerID INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->ownerID << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS GlobalID INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].meta->id << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS isConnected INT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << int(cpuMesh.elem[i].meta->isConnected) << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS Area FLOAT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		mpiProcessesOut << std::fixed << cpuMesh.elem[i].flow->area << std::endl;

	mpiProcessesOut << std::endl;

	mpiProcessesOut << "SCALARS EdgeLengthSum FLOAT" << std::endl;
	mpiProcessesOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		float edgeLengthSum = 0.0f;
		for (unsigned /*short*/ k = 0; k < 3; ++k)
			edgeLengthSum += cpuMesh.elem[i].flow->flux->neighbor[k].length;
		mpiProcessesOut << std::fixed << edgeLengthSum << std::endl;
	}

	mpiProcessesOut.close();
	std::cout << " Done. " << std::endl;
}
