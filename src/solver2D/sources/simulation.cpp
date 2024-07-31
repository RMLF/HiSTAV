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
#include <fstream>
#include <string>
#include <vector>

// OpenMP
#include <omp.h>

// STAV
#include "../headers/simulation.hpp"
#include "../headers/numerics.hpp"
#include "../headers/mesh.hpp"
#include "../headers/boundaries.hpp"

// Definitions
#define ompBlockSIZE 12500

/////////////////////////////////////////////////////////////////////////////////////////////////

simulationSTAV2D::simulationSTAV2D(){

	controlFolder = "./control/";
	timeControlFileName = "time.cnt";

	stepWClockTime = 0.0f;
	totalWClockTime = 0.0f;

	initialTime = 0.0f;
	finalTime = 0.0f;
	currentTime = 0.0f;
	step = 0;
	isRunning = false;

	control = 0;
}

simulationSTAV2D cpuSimulation;
#ifdef __STAV_CUDA__
GPU float gpuCurrentTime;
#endif


void simulationSTAV2D::init(){

#	ifdef __STAV_MPI__
	myProc.getRank();

	MPI_Comm_dup(MPI_COMM_WORLD, &GLOBALS);
	MPI_Comm_dup(MPI_COMM_WORLD, &COMPUTE);
#	endif

	control = &cpuControl;

	control->physics = &cpuPhysics;
	control->numerics = &cpuNumerics;
	control->bed = &cpuBed;
	control->forcing = &cpuForcing;
	//control->lagrangian = &cpuLagrangian;
	control->initial = &cpuInitial;
	control->output = &cpuOutput;

#	ifdef __STAV_MPI__
	if (myProc.master){
#	endif

	std::cout << std::endl;

	std::cout << "            _____ _______  __      __    ___  _____" << std::endl;
	std::cout << "           / ____|__   __|/\\ \\    / /   |__ \\|  __ \\ " << std::endl;
	std::cout << "          | (___    | |  /  \\ \\  / /       ) | |  | |" << std::endl;
	std::cout << "           \\___ \\   | | / /\\ \\ \\/ /       / /| |  | |" << std::endl;
	std::cout << "           ____) |  | |/ ____ \\  /       / /_| |__| |" << std::endl;
	std::cout << "          |_____/   |_/_/    \\_\\/       |____|_____/" << std::endl;

	std::cout << std::endl << std::endl;

#	ifdef __STAV_MPI__
	}
#	endif
}

void simulationSTAV2D::runOnCPU(){

	std::cout << std::endl << std::endl;

	stepWClockTime = float(omp_get_wtime());
	totalWClockTime = float(omp_get_wtime());

#	pragma omp parallel
	{
		while (currentTime <= finalTime) {

			double threadPrivateDt = 99999999999.0;
			
			if (cpuBoundaries.hasUniformInlets)
#				pragma omp for schedule(static,1)
				for (unsigned b = 0; b < cpuBoundaries.numBoundaries; ++b)
					if (cpuBoundaries.physical[b].isUniformInlet)
						cpuBoundaries.physical[b].setRefValue();

#			pragma omp for schedule(static, ompBlockSIZE)
			for (unsigned i = 0; i < cpuBoundaries.numElemGhosts; ++i)
				cpuBoundaries.elemGhost[i].getConditions();

#			pragma omp for schedule(static, ompBlockSIZE)
			for (unsigned i = 0; i < cpuMesh.numElems; ++i){
				cpuMesh.elemFlow[i].computeFluxes();
				if (cpuMesh.elemDt[i] < threadPrivateDt)
					threadPrivateDt = cpuMesh.elemDt[i];
			}

#			pragma omp critical
			{
				if (threadPrivateDt < cpuNumerics.dt)
					cpuNumerics.dt = threadPrivateDt;
			}
#			pragma omp barrier

#			pragma omp for schedule(static, ompBlockSIZE)
			for (unsigned i = 0; i < cpuMesh.numElems; ++i)
				cpuMesh.elemFlow[i].applyFluxes();

#			pragma omp for schedule(static, ompBlockSIZE)
			for (unsigned i = 0; i < cpuMesh.numElems; ++i)
				cpuMesh.elemFlow[i].applyCorrections();

#			pragma omp for schedule(static, ompBlockSIZE)
			for (unsigned i = 0; i < cpuMesh.numElems; ++i)
				cpuMesh.elem[i].applySourceTerms();

#			pragma omp barrier
#			pragma omp single
			{
				step++;
				currentTime += float(cpuNumerics.dt);
				stepWClockTime = float(omp_get_wtime()) - stepWClockTime;

				std::cout << std::fixed << std::setprecision(6) << "  dt (s): " << cpuNumerics.dt;
				std::cout << std::fixed << std::setprecision(3) << ",  CPU (s): " << stepWClockTime << ",  Ratio: ";
				std::cout << cpuNumerics.dt / stepWClockTime << ",  Time (s): " << currentTime << ",  Comp. (%): ";
				std::cout << currentTime / finalTime * 100.0f << ",  OMP: " << omp_get_num_threads() << std::endl;

				stepWClockTime = float(omp_get_wtime());
				cpuNumerics.dt = 99999999999.0;

				cpuOutput.exportAll();
			}
		}
	}

	cpuOutput.exportMaxima();

	std::cout << std::endl;
	std::cout << std::fixed << std::setprecision(3) << "  Total Simulation Time (s): " << (totalWClockTime = float(omp_get_wtime()) - totalWClockTime) << std::endl;
	std::cout << std::endl;
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void simulationSTAV2D::readControlFiles(){

	std::cout << "  -> Importing control files ..." << std::endl;
	
	std::ifstream timeControlFile;
	timeControlFile.open(controlFolder + timeControlFileName);
	
	if (!timeControlFile.is_open() || !timeControlFile.good()){
		std::cout << "   -> Err [C-1]: Could not open file " + controlFolder + timeControlFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}
	
	std::cout << "   -> Reading " + controlFolder + timeControlFileName << std::endl;

	timeControlFile >> initialTime;
	currentTime = initialTime;
	timeControlFile >> finalTime;

	timeControlFile.close();

	std::vector<std::string> fileNames;
	std::vector<std::string> fileFolders;

	fileNames.push_back(control->controlFolder + control->controlPhysicsFileName);
	cpuPhysics.readControlFile(fileNames[0]);

	fileNames.push_back(control->controlFolder + control->controlNumericsFileName);
	cpuNumerics.readControlFile(fileNames[1]);

	fileNames.push_back(control->forcingFolder + control->forceControlFileName);
	fileFolders.push_back(control->forcingFolder + control->forceGaugesFolder);
	cpuForcing.readControlFile(fileNames[2], fileFolders[0]);

	//control->lagrangian->readControlFile();
	cpuInitial.readControlFile();
	cpuOutput.readControlFile();

#	ifdef __STAV_CUDA__
#	ifdef __STAV_MPI__
	cpuControl.readHPCControlFile();
#	endif
#	endif

	fileNames.push_back(control->boundaryFolder + control->boundaryMeshFolder + control->boundaryDimFileName);
	fileNames.push_back(control->boundaryFolder + control->boundaryControlFileName);
	fileFolders.push_back(control->boundaryFolder + control->boundaryGaugesFolder);
	cpuBoundaries.readControlFiles(fileNames[3], fileNames[4], fileFolders[1]);
}

void simulationSTAV2D::readMeshFiles(){

	std::vector<std::string> fileNames;
	std::vector<std::string> fileFolders;

	fileNames.push_back(control->meshFolder + control->meshDimFileName);
	fileNames.push_back(control->meshFolder + control->nodesFileName);
	fileNames.push_back(control->meshFolder + control->elementsFileName);
	cpuMesh.readTopologyFiles(fileNames[0], fileNames[1], fileNames[2]);

	fileNames.push_back(control->bedFolder + control->bedControlFileName);
	fileNames.push_back(control->bedFolder + control->bedLayersFolder + control->bedLayersFileName);
	fileFolders.push_back(control->bedFolder + control->bedLayersFolder);
	fileFolders.push_back(control->bedFolder + control->bedFractionFolder);
	fileFolders.push_back(control->bedFolder + control->bedGradCurveFolder);
	cpuBed.readAllFiles(fileNames[3], fileNames[4], fileFolders[0], fileFolders[1], fileFolders[2]);
	cpuInitial.setInitialConditions();

	fileNames.push_back(control->boundaryFolder + control->boundaryMeshFolder + control->boundaryIdxFileName);
	if (cpuBoundaries.numBoundaries > 0)
		cpuBoundaries.readMeshFiles(fileNames[5]);
}
