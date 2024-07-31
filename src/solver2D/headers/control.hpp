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
#include <string>

// STAV
#include "compile.hpp"
#include "common.hpp"

// Forward Declarations
class bedLandslide;

/////////////////////////////////////////////////////////////////////////////////////////////////


class physicsParameters{
public:

	CPU GPU physicsParameters();

	void readControlFile(std::string&);

	float gravity;
	float waterDensity;
	float waterDinVis;
	float waterTemp;

#	ifdef __STAV_CUDA__
	void copyToGPU();
#	endif

#	ifdef __STAV_MPI__
	void bcast();
#	endif
};

class numericsParameters{
public:

	CPU GPU numericsParameters();

	void readControlFile(std::string&);

	double dt;
	float CFL;
	float minDepthFactor;

#	ifdef __STAV_CUDA__
	void copyToGPU();
#	endif

#	ifdef __STAV_MPI__
	void bcast();
#	endif
};

class sedimentType{
public:

	CPU GPU sedimentType();
	void readData(std::string&);

	unsigned /*short*/ color;
	float diam;
	float specGrav;
	float poro;
	float rest;
	float alpha;
	float beta;
	float tanPhi;
	float adaptLenMinMult;
	float adaptLenMaxMult;
	float adaptLenRefShields;
	float adaptLenShapeFactor;
};

class bedParameters{
public:

	CPU GPU bedParameters();

	void readAllFiles(std::string&, std::string&, std::string&, std::string&, std::string&);
	void importFrictionCoef(std::string&);
	void importBedrockOffset(std::string&);
	void importBedComposition(std::string&, std::vector<std::string>&);

	sedimentType grain[maxFRACTIONS];
	bedLandslide* landslide;

	float maxConc;
	float frictionCoef;
	float bedrockOffset;
	float mobileBedTimeTrigger;

	unsigned /*short*/ numGradingCurves;
	unsigned /*short*/ frictionOption;
	unsigned /*short*/ depEroOption;
	unsigned /*short*/ adaptLenOption;

	bool useFriction;
	bool useMobileBed;

	bool useFrictionMap;
	bool useBedrockMap;
	bool useGradeIDMap;

	/*
	float lastLandslideTime;
	float firstLandslideTime;
	bool useLandslides;
	bool useLandslidesID;
	void importLandslidesMap();
	void readLandslidesTriggerFile();
	*/

#	ifdef __STAV_CUDA__
	void copyToGPU();
#	endif

#	ifdef __STAV_MPI__
	void bcast();
#	endif
};

class forcingParameters{
public:

	CPU GPU forcingParameters();

	void readControlFile(std::string&, std::string&);

	timeseries* rainGauge;
	unsigned /*short*/ numRainGauges;
	bool useRainfall;

	void readPrecipitationFile(std::string&);

#	ifdef __STAV_CUDA__
	void copyToGPU();
#	endif

#	ifdef __STAV_MPI__
	void bcast();
#	endif
};

class initialParameters{
public:

	initialParameters();

	void readControlFile();
	void setInitialConditions();
	void loadInitialConditions();

	std::string resumeFolder;
	std::string resumeControlFileName;

	std::string hydroResumeFileName;
	std::string bedResumeFileName;
	std::string scalarsResumeFileName;
	std::string maximaResumeFileName;

	float iniValue;
	unsigned /*short*/ iniValueType;
	bool resumeSimulation;
};

class outputParameters{
public:

	outputParameters();

	void readControlFile();
	void exportHydrodynamics();
	void exportScalars();
	void exportBedComposition();
	//void exportForcing();
	void exportMaxima();
	//void exportTimeStep();
	void exportAll();

	std::string outputFolder;
	std::string outputControlFileName;
	std::string outputHydroFileName;
	std::string outputScalarsFileName;
	std::string outputBedFileName;
	std::string outputForcingFileName;
	std::string outputMaximaFileName;

	float writeOutputTime;
	float updateMaximaTime;

	float outputWriteFreq;
	float maximaUpdateFreq;

	bool writeTimeStep;

#	ifdef __STAV_MPI__
	void bcast();
#	endif
};

class controlParameters{
public:

	controlParameters();

	std::string controlFolder;
	std::string controlPhysicsFileName;
	std::string controlNumericsFileName;

	std::string bedFolder;
	std::string bedLayersFolder;
	std::string bedFractionFolder;
	std::string bedGradCurveFolder;
	std::string bedControlFileName;
	std::string bedLayersFileName;

	std::string forcingFolder;
	std::string forceGaugesFolder;
	std::string forceControlFileName;

	//std::string lagrangianFolder;
	//std::string lagrControlFileName;

	std::string meshFolder;
	std::string meshDimFileName;
	std::string nodesFileName;
	std::string elementsFileName;

	std::string boundaryFolder;
	std::string boundaryMeshFolder;
	std::string boundaryGaugesFolder;
	std::string boundaryDimFileName;
	std::string boundaryIdxFileName;
	std::string boundaryControlFileName;

	physicsParameters* physics;
	numericsParameters* numerics;
	bedParameters* bed;
	forcingParameters* forcing;
	//lagrangianParameters* lagrangian;

	initialParameters* initial;
	outputParameters* output;

#	ifdef __STAV_CUDA__
	void copyToGPU();
#	endif

#	ifdef __STAV_MPI__
	void bcast();
#	ifdef __STAV_CUDA__
	void readHPCControlFile();
	std::string controlHPCFileName;
#	endif
#	endif
};

extern physicsParameters cpuPhysics;
extern numericsParameters cpuNumerics;
extern bedParameters cpuBed;
extern forcingParameters cpuForcing;
//extern lagrangianParameters cpuLagrangian;

extern initialParameters cpuInitial;
extern outputParameters cpuOutput;
extern controlParameters cpuControl;

#ifdef __STAV_CUDA__
extern GPU physicsParameters gpuPhysics;
extern GPU numericsParameters gpuNumerics;
extern GPU bedParameters gpuBed;
extern GPU forcingParameters gpuForcing;
//extern GPU lagrangianParameters gpuLagrangian;
#endif
