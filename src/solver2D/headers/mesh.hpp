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
#pragma once

// STAV
#include "compile.hpp"
#include "common.hpp"
#include "geometry.hpp"
#include "numerics.hpp"
#include "forcing.hpp"
#include "sediment.hpp"
#include "output.hpp"
#ifdef __STAV_MPI__
#include "mpiRun.hpp"
#endif

// Forward Declarations
class elementOutput;

/////////////////////////////////////////////////////////////////////////////////////////////////


class elementNode{
public:

	elementNode();

	point coord;

	float fricPar;
	float bedrockOffset;
	float landslideDepth;

	unsigned id;

	unsigned /*short*/ curveID;
	unsigned /*short*/ landslideID;
};

class elementMeta{
public:

	elementMeta();

	elementNode* node[3];
	element* elem[3];
	point center;

	unsigned id;
	unsigned ownerID;
	unsigned /*short*/ owner;

#	ifdef __STAV_MPI__
	bool isConnected;
#	endif
};

class element{
public:

	CPU GPU element();

	CPU GPU INLINE void applySourceTerms();
	void computeGeometry();

#	ifdef __STAV_MPI__
	void sendToOwner();
	void recvFromMaster();
	void sendResults();
	void recvResults();
#	endif

	elementFlow* flow;
	elementBed* bed;
	elementForcing* forcing;

	elementMeta* meta;
};

class simulationMesh{
public:

	CPU GPU simulationMesh();

	void allocateOnCPU();
	void deallocateFromCPU();

	void readTopologyFiles(std::string&, std::string&, std::string&);

#	ifdef __STAV_CUDA__
	void copyToGPU();
	void copyToCPU();
	void deallocateFromGPU();
	void freeUpCPU();
#	endif

#	ifdef __STAV_MPI__
	void scatterToSlaves();
	void recvFromMaster();
	void reset();
#	endif

	unsigned numNodes;
	unsigned numElems;

#	ifdef __STAV_MPI__
	unsigned numConnectedElems;
	unsigned numRemoteElems;
#	endif

	elementNode* elemNode;

	elementFlow* elemFlow;
	elementState* elemState;
	elementScalars* elemScalars;
	elementConnect* elemConnect;
	elementFluxes* elemFluxes;
	double* elemDt;

	elementBed* elemBed;
	elementBedComp* elemBedComp;
	elementForcing* elemForcing;

	elementMeta* elemMeta;
	element* elem;
	
	elementOutput* elemOutput;

#	ifdef __STAV_CUDA__
	elementFlow* gpuElemFlow;
	elementState* gpuElemState;
	elementScalars* gpuElemScalars;
	elementConnect* gpuElemConnect;
	elementFluxes* gpuElemFluxes;
	double* gpuElemDt;

	elementBed* gpuElemBed;
	elementBedComp* gpuElemBedComp;
	elementForcing* gpuElemForcing;

	element* gpuElem;
#	endif

#	ifdef __STAV_MPI__
	elementConnected* elemConnected;
	elementState* elemConnectedState;
	elementScalars* elemConnectedScalars;
	
	elementRemote* elemRemote;
	elementState* elemRemoteState;
	elementScalars* elemRemoteScalars;
#	endif

#	ifdef __STAV_MPI__
#	ifdef __STAV_CUDA__
	elementState* gpuElemConnectedState;
	elementScalars* gpuElemConnectedScalars;

	elementState* gpuElemRemoteState;
	elementScalars* gpuElemRemoteScalars;
#	endif
#	endif
};

extern simulationMesh cpuMesh;



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU INLINE void element::applySourceTerms() {

#	ifdef __CUDA_ARCH__
	bool useFriction = gpuBed.useFriction;
	bool useMobileBed = gpuBed.useMobileBed;
	//bool useRainfall = gpuForcing.useRainfall;
#	else
	bool useFriction = cpuBed.useFriction;
	bool useMobileBed = cpuBed.useMobileBed;
	//bool useRainfall = cpuForcing.useRainfall;
#	endif

	/*if(useRainfall)
		forcing->applyRainfall();*/

	if (bed->flow && flow->state->isWet()){
		
		if (useFriction)
			bed->applyBedFriction();

		if (useMobileBed)
			bed->applyMobileBed();

		//bed->validate();
		flow->validate();
	}
}
