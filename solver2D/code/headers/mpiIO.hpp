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

// STAV's Headers
#include "../headers/compile.hpp"
#include "../headers/control.hpp"
#include "../headers/boundaries.hpp"
#include "../headers/forcing.hpp"
#include "../headers/mesh.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////
// MPI I/O communication classes, datatypes and (respective) constructors
/////////////////////////////////////////////////////////////////////////////////////////////////


class staticControlMPI{
public:

	staticControlMPI();
	MPI_Datatype createCommObj();

	void copyFrom(const controlParameters&, const domainBoundaries&);
	void copyTo(controlParameters&, domainBoundaries&);

	double dt;

	float gravity;
	float waterDensity;
	float waterDinVis;
	float waterTemp;

	float CFL;
	float minDepthFactor;

	float maxConc;
	float mobileBedTimeTrigger;

	unsigned short numRainGauges;

	unsigned short frictionOption;
	unsigned short depEroOption;
	unsigned short adaptLenOption;

	unsigned short numBoundaries;
	unsigned short numBndGauges;

	unsigned short useFriction;
	unsigned short useMobileBed;
	unsigned short useRainfall;
};

class dynamicControlMPI{
public:

	dynamicControlMPI();
	dynamicControlMPI(const unsigned);
	
	void copyFrom(const controlParameters&, const domainBoundaries&);
	void copyTo(controlParameters&, domainBoundaries&);

	std::vector<float> allData;
};

class elementMPI{
public:

	elementMPI();
	MPI_Datatype createCommObj();

	void copyFrom(const element&);
	void copyTo(element&);

	float state[(maxCONSERVED + 2 + maxSCALARS)];
	float bedComp[maxFRACTIONS];
	float subComp[maxFRACTIONS];
	float geometry[14];

	int connectFlag[1]; 		// Step (1) Leave it as array[1] 	Step (2) Don't ask.
	int rainGaugeIdx[maxNN];
};

class physicalBoundaryMPI{
public:

	physicalBoundaryMPI();
	MPI_Datatype createCommObj();

	void copyFrom(const physicalBoundary&);
	void copyTo(physicalBoundary&);

	float geometry[2];
	float inletCurve[maxDISCHCURLENGTH*2];

	unsigned flags[5];
};

class elementGhostMPI{
public:

	elementGhostMPI();
	MPI_Datatype createCommObj();

	void copyFrom(const elementGhost&);
	void copyTo(elementGhost&);

	float geometry[3];
	float gaugeWeights[2];
	
	unsigned gaugeIdxs[2];
	unsigned flags[4];
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void exportAllProcessesMPI();
void exportCurrentProcessMPI();
