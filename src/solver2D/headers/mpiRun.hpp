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

// STL
#include <vector>

// STAV
#include "compile.hpp"
#include "numerics.hpp"
#include "forcing.hpp"
#include "sediment.hpp"

// Definitions
#define bufferSIZE (maxCONSERVED + 2 + maxSCALARS)
#define timerStepsSKIP 2
#define testStepLimit 250000

/////////////////////////////////////////////////////////////////////////////////////////////////


extern MPI_Comm GLOBALS;
extern MPI_Comm COMPUTE;

extern std::vector<unsigned /*short*/> globals;
extern std::vector<float> chronometer;

class mpiProcess{
public:

	mpiProcess();

	void getRank();
	void getSlavesPerformance();
	void runSlaveTest();

	float workCapacity;
	float loadCapacity;

	int /*short*/ cudaID;

	unsigned firstElem;
	unsigned lastElem;
	unsigned messageCounter;
	
	unsigned /*short*/ rank;
	unsigned /*short*/ worldSize;

	bool master;
	bool worker;
	bool hasGPU;

	bool reloadMesh;
	bool writeResults;
};

extern mpiProcess myProc;
extern std::vector<mpiProcess> mySlaves;


class elementConnected{
public:

	elementConnected();

	INLINE void sendState();

	elementState* state;
	elementScalars* scalars;

	float messageBuffer[bufferSIZE];
	
	unsigned /*short*/ neighbourOwner[3];
	unsigned id;

	bool toSend[3];
};

class elementRemote{
public:

	elementRemote();

	INLINE void recvState();

	elementState* state;
	elementScalars* scalars;

	float messageBuffer[bufferSIZE];

	unsigned id;
	unsigned /*short*/ owner;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


INLINE void elementConnected::sendState(){

	messageBuffer[0] = state->vel.x;
	messageBuffer[1] = state->vel.y;
	messageBuffer[2] = state->h;
	messageBuffer[3] = state->rho;
	messageBuffer[4] = state->z;

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		messageBuffer[maxCONSERVED + 2 + s] = scalars->specie[s];

	for (unsigned /*short*/ pr = 0; pr < 3; ++pr)
		if (toSend[pr])
			MPI_Send(&messageBuffer[0], bufferSIZE, MPI_FLOAT, neighbourOwner[pr], id, COMPUTE);
}

INLINE void elementRemote::recvState(){

	MPI_Recv(&messageBuffer[0], bufferSIZE, MPI_FLOAT, owner, id, COMPUTE, MPI_STATUS_IGNORE);

	state->vel.x = messageBuffer[0];
	state->vel.y = messageBuffer[1];
	state->h = messageBuffer[2];
	state->rho = messageBuffer[3];
	state->z = messageBuffer[4];

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		scalars->specie[s] = messageBuffer[maxCONSERVED + 2 + s];
}
