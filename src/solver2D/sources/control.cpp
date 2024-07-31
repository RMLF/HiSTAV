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
#include <fstream>
#include <vector>
#include <string>

// STAV
#include "../headers/control.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU physicsParameters::physicsParameters(){
#	ifndef __CUDA_ARCH__
	gravity = 9.81f;
	waterDensity = 1000.0f;
	waterDinVis = 0.001f;
	waterTemp = 20.0f;
#	endif
}

CPU GPU numericsParameters::numericsParameters(){
#	ifndef __CUDA_ARCH__
	dt = 999999999999.0;
	CFL = 0.5f;
	minDepthFactor = 0.005f;
#	endif
}

CPU GPU sedimentType::sedimentType(){
#	ifndef __CUDA_ARCH__
	color = 0;
	diam = 0.00161f;
	specGrav = 2.65f;
	poro = 0.40f;
	rest = 0.90f;
	alpha = 3.2f;
	beta = 3.0f;
	tanPhi = 0.5f;
	adaptLenMinMult = 20.0f;
	adaptLenMaxMult = 8000.0f;
	adaptLenRefShields = 1.4f;
	adaptLenShapeFactor = 14.0f;
#	endif
}

CPU GPU bedParameters::bedParameters(){
#	ifndef __CUDA_ARCH__
	landslide = 0x0;

	maxConc = 0.7f;
	frictionCoef = 5.0f;
	bedrockOffset = 0.0f;
	mobileBedTimeTrigger = 0.0f;

	numGradingCurves = 0;
	frictionOption = 1;
	depEroOption = 0;
	adaptLenOption = 0;

	useFriction = true;
	useMobileBed = false;

	useFrictionMap = false;
	useBedrockMap = false;
	useGradeIDMap = false;

	/*lastLandslideTime = 0.0f;
	firstLandslideTime = 0.0f;
	useLandslides;
	useLandslidesID;
	importLandslidesMap();
	importLandslidesIdMap();
	readLandslidesTriggerFile();*/
#	endif
}

CPU GPU forcingParameters::forcingParameters(){
#	ifndef __CUDA_ARCH__
	rainGauge = 0x0;
	numRainGauges = 0;
	useRainfall = false;
#	endif
}

initialParameters::initialParameters(){

	resumeFolder = "./initial/";
	resumeControlFileName = "initial.cnt";

	iniValue = 0.0f;
	iniValueType = 0;
	resumeSimulation = false;
}

outputParameters::outputParameters(){

	outputFolder = "./output/";
	outputControlFileName = "output.cnt";
	outputHydroFileName = "hydro";
	outputScalarsFileName = "scalars";
	outputBedFileName = "bed";
	outputForcingFileName = "force";
	outputMaximaFileName = "maxima";

	writeOutputTime = 0.0f;
	updateMaximaTime = 0.0f;
	outputWriteFreq = 0.0f;
	maximaUpdateFreq = 0.0f;
	writeTimeStep = false;
}

controlParameters::controlParameters(){

	controlFolder = "./control/";
	controlPhysicsFileName = "physics.cnt";
	controlNumericsFileName = "numerics.cnt";

	bedFolder = "./bed/";
	bedLayersFolder = "activeLayers/";
	bedFractionFolder = "fractionData/";
	bedGradCurveFolder = "gradCruves/";
	bedControlFileName = "bed.cnt";
	bedLayersFileName = "bedLayers.sed";

	forcingFolder = "./forcing/";
	forceGaugesFolder = "gauges/";
	forceControlFileName = "forcing.cnt";

	//lagrangianFolder;
	//lagrControlFileName;

	meshFolder = "./mesh/";
	meshDimFileName = "info.mesh";
	nodesFileName = "nodes.mesh";
	elementsFileName = "elements.mesh";

	boundaryFolder = "./boundary/";
	boundaryMeshFolder = "meshData/";
	boundaryGaugesFolder = "gauges/";
	boundaryDimFileName = "boundaryDim.bnd";
	boundaryIdxFileName = "boundaryIdx";
	boundaryControlFileName = "boundary.cnt";

	initial = 0x0;
	physics = 0x0;
	numerics = 0x0;
	bed = 0x0;
	forcing = 0x0;
	//lagrangian = 0x0;
	output = 0x0;

#	ifdef __STAV_MPI__
#	ifdef __STAV_CUDA__
	controlHPCFileName = "hpc.cnt";
#	endif
#	endif
}

physicsParameters cpuPhysics;
numericsParameters cpuNumerics;
bedParameters cpuBed;
forcingParameters cpuForcing;
//lagrangianParameters cpuLagrangian;

initialParameters cpuInitial;
outputParameters cpuOutput;
controlParameters cpuControl;

#ifdef __STAV_CUDA__
GPU physicsParameters gpuPhysics;
GPU numericsParameters gpuNumerics;
GPU bedParameters gpuBed;
GPU forcingParameters gpuForcing;
//GPU lagrangianParameters gpuLagrangian;
#endif



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void physicsParameters::readControlFile(std::string& controlFileName){

	std::ifstream controlFile;
	controlFile.open(controlFileName);

	if (!controlFile.is_open() || !controlFile.good()){
		std::cerr << "   -> *Error* [C-2]: Could not open file " + controlFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + controlFileName << std::endl;

	controlFile >> gravity;
	controlFile >> waterDensity;
	controlFile >> waterDinVis;

	controlFile.close();
}

void numericsParameters::readControlFile(std::string& controlFileName){

	std::ifstream controlFile;
	controlFile.open(controlFileName);

	if (!controlFile.is_open() || !controlFile.good()){
		std::cerr << "   -> *Error* [C-3]: Could not open file " + controlFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + controlFileName << std::endl;

	controlFile >> CFL;
	controlFile >> minDepthFactor;

	controlFile.close();
}
