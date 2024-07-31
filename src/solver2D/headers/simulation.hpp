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
#include <string>

// STAV
#include "compile.hpp"
#include "control.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

class controlParameters;

class simulationSTAV2D{
public:

	simulationSTAV2D();

	controlParameters* control;
	
	std::string controlFolder;
	std::string timeControlFileName;

	float stepWClockTime;
	float totalWClockTime;

	float initialTime;
	float finalTime;
	float currentTime;
	unsigned step;
	bool isRunning;

	void init();
	void readControlFiles();
	void readMeshFiles();

	void runOnCPU();

#	ifdef __STAV_CUDA__
	double reduceDtOnGPU();
	void runOnGPU();
#	endif

#	ifdef __STAV_MPI__
	void runMasterOnCPU();
	void runSlaveOnCPU();
#		ifdef __STAV_CUDA__
		void runSlaveOnGPU();
#		endif
	void gatherResults();
	void checkBalance();
#	endif
};

extern simulationSTAV2D cpuSimulation;
#ifdef __STAV_CUDA__
extern GPU float gpuCurrentTime;
#endif
