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
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>
#include <algorithm>

// STAV
#include "../headers/output.hpp"
#include "../headers/common.hpp"
#include "../headers/control.hpp"
#include "../headers/numerics.hpp"
#include "../headers/sediment.hpp"
#include "../headers/simulation.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


elementOutput::elementOutput(){

	link = 0;

	h = 0;
	rho = 0;
	z = 0;
	zIni = 0;

	for (unsigned short s = 0; s < maxSCALARS; ++s)
		scalars[s] = 0.0f;

	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		perc[p] = (1.0f / ((float) maxFRACTIONS));
}

void elementOutput::update(){

	vel = link->flow->state->vel;
	h = link->flow->state->h;
	rho = link->flow->state->rho;
	z = link->flow->state->z;

	for (unsigned short s = 0; s < maxSCALARS; ++s){
		scalars[s] = link->flow->scalars->specie[s];
		if (s == relTemp)
			scalars[s] = link->flow->scalars->getTemp();
	}

	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		perc[p] = link->bed->comp->bedPercentage[p];
}

elementMaximum::elementMaximum(){

	link = 0;

	maxVel = 0.0f;
	maxQ = 0.0f;
	maxQTime = -1.0f;
	maxDepth = 0.0f;
	maxDepthTime = -1.0f;
	wettingTime = -1.0f;
	maxDep = 0.0f;
	maxEro = 0.0f;
	zIni = 0.0f;
}

void elementMaximum::update(){

	if (link->flow->state->vel.norm() > maxVel){
		maxVel = link->flow->state->vel.norm();
		maxDepthTime = cpuSimulation.currentTime;
	}

	// APA only, remove after
	float discharge = (link->flow->state->vel.norm() + 0.5f) * link->flow->state->h;
	if (discharge > maxQ){
		maxQ = discharge;
		maxQTime = cpuSimulation.currentTime;
	}

	if (link->flow->state->h > maxDepth)
		maxDepth = link->flow->state->h;

	if (isValid(link->flow->state->h) && wettingTime <= 0.0f)
		wettingTime = cpuSimulation.currentTime;

	if ((link->flow->state->z - zIni) > maxEro)
		maxEro = (link->flow->state->z - zIni);

	if ((link->flow->state->z - zIni) < maxDep)
		maxDep = (link->flow->state->z - zIni);
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void outputParameters::readControlFile(){

	std::string inputText;
	std::ifstream outputControlFile;
	outputControlFile.open(outputFolder + outputControlFileName);

	if (!outputControlFile.is_open() || !outputControlFile.good()){
		std::cerr << "   -> *Error* [O-1]: Could not open file " + outputFolder + outputControlFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}else{
		std::cout << "   -> Reading " + outputFolder + outputControlFileName << std::endl;
		outputControlFile >> cpuOutput.outputWriteFreq;
		outputControlFile >> cpuOutput.maximaUpdateFreq;
	}

	outputControlFile.close();

	writeOutputTime = cpuSimulation.initialTime;
	updateMaximaTime = cpuSimulation.initialTime;
}

void outputParameters::exportAll(){

	if (cpuSimulation.currentTime < writeOutputTime && cpuSimulation.currentTime < updateMaximaTime)
		return;
	
	if (cpuSimulation.currentTime >= updateMaximaTime){
		updateMaximaTime += 1.0f / maximaUpdateFreq;

#		ifdef __STAV_CUDA__
#		ifndef __STAV_MPI__
		cudaMemcpy(cpuMesh.elemState, cpuMesh.gpuElemState, sizeof(elementState)*cpuMesh.numElems, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMesh.elemScalars, cpuMesh.gpuElemScalars, sizeof(elementScalars)*cpuMesh.numElems, cudaMemcpyDeviceToHost);
#		endif
#		endif

#		pragma omp parallel for schedule(static)
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elemOutput[i].max.update();
	}

	if (cpuSimulation.currentTime >= writeOutputTime){

		std::cout << std::endl;
		writeOutputTime += 1.0f / outputWriteFreq;

#		ifdef __STAV_CUDA__
#		ifndef __STAV_MPI__
		cudaMemcpy(cpuMesh.elemState, cpuMesh.gpuElemState, sizeof(elementState)*cpuMesh.numElems, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuMesh.elemScalars, cpuMesh.gpuElemScalars, sizeof(elementScalars)*cpuMesh.numElems, cudaMemcpyDeviceToHost);
#		endif
#		endif

#		pragma omp parallel for schedule(static)
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elemOutput[i].update();

		exportHydrodynamics();

		if (false)
			exportScalars();

		if (false)
			exportMaxima();

		/*if (cpuBed.useMobileBed){
		
#			ifdef __STAV_CUDA__
#			ifndef __STAV_MPI__
			cudaMemcpy(cpuMesh.elemBedComp, cpuMesh.gpuElemBedComp, sizeof(elementBedComp)*cpuMesh.numElems, cudaMemcpyDeviceToHost);
#			endif
#			endif

			exportBedComposition();
		}*/

		std::cout << std::endl;
	}
}

void outputParameters::exportHydrodynamics(){

	std::ofstream hydroOut;
	std::ostringstream hydroOutSstream;
	hydroOutSstream << "./output/hydrodynamics/hydro-" << cpuSimulation.step << ".vtk";
	std::string hydroOutfile = hydroOutSstream.str();

	std::cout << "   Writing " << hydroOutfile.c_str() << " ... ";

	hydroOut.open(hydroOutfile.c_str());
	hydroOut << "# vtk DataFile Version 2.0" << std::endl;
	hydroOut << "STAV-2D solver at " << cpuSimulation.currentTime << " secs." << std::endl;
	hydroOut << "ASCII" << std::endl;
	hydroOut << "DATASET UNSTRUCTURED_GRID" << std::endl;

	hydroOut << std::endl;

	hydroOut << "POINTS " << cpuMesh.numNodes << " FLOAT" << std::endl;

	for (unsigned i = 0; i < cpuMesh.numNodes; ++i)
		hydroOut << std::fixed << cpuMesh.elemNode[i].coord.x << "	" << std::fixed << cpuMesh.elemNode[i].coord.y << "	" << std::fixed << 0.0f << std::endl;

	hydroOut << std::endl;

	hydroOut << "CELLS " << cpuMesh.numElems << "	" << cpuMesh.numElems * 4 << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		hydroOut << 3 << "	" << cpuMesh.elem[i].meta->node[0]->id << "	" << cpuMesh.elem[i].meta->node[1]->id << "	" << cpuMesh.elem[i].meta->node[2]->id << std::endl;

	hydroOut << std::endl;

	hydroOut << "CELL_TYPES " << cpuMesh.numElems << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		hydroOut << 5 << std::endl;

	hydroOut << std::endl;

	hydroOut << "CELL_DATA " << cpuMesh.numElems << std::endl;
	hydroOut << "SCALARS Bed FLOAT" << std::endl;
	hydroOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		hydroOut << std::fixed << cpuMesh.elemOutput[i].z << std::endl;

	hydroOut << std::endl;

	hydroOut << "SCALARS Level FLOAT" << std::endl;
	hydroOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		hydroOut << std::fixed << (cpuMesh.elemOutput[i].h + cpuMesh.elemOutput[i].z) << std::endl;

	hydroOut << std::endl;

    if (false){
        hydroOut << "SCALARS Density FLOAT" << std::endl;
        hydroOut << "LOOKUP_TABLE default" << std::endl;
        for (unsigned i = 0; i < cpuMesh.numElems; ++i)
            hydroOut << std::fixed << cpuMesh.elemOutput[i].rho << std::endl;

        hydroOut << std::endl;
    }

	if (false){
		hydroOut << "SCALARS Dt FLOAT" << std::endl;
		hydroOut << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			hydroOut << std::fixed << cpuMesh.elemDt[i] << std::endl;

		hydroOut << std::endl;
	}

	hydroOut << "VECTORS Velocity FLOAT" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		hydroOut << std::fixed << cpuMesh.elemOutput[i].vel.x << "	" << std::fixed << cpuMesh.elemOutput[i].vel.y << "	" << 0.0f << std::endl;

	hydroOut.close();
	std::cout << " Done. " << std::endl;
}

void outputParameters::exportScalars(){

	std::ofstream scalarsOut;
	std::ostringstream hydroOutSstream;
	hydroOutSstream << "./output/scalars/scalars-" << cpuSimulation.step << ".vtk";
	std::string hydroOutfile = hydroOutSstream.str();

	std::cout << "   Writing " << hydroOutfile.c_str() << " ... ";

	scalarsOut.open(hydroOutfile.c_str());
	scalarsOut << "# vtk DataFile Version 2.0" << std::endl;
	scalarsOut << "STAV-2D solver at " << cpuSimulation.currentTime << " secs." << std::endl;
	scalarsOut << "ASCII" << std::endl;
	scalarsOut << "DATASET UNSTRUCTURED_GRID" << std::endl;

	scalarsOut << std::endl;

	scalarsOut << "POINTS " << cpuMesh.numNodes << " FLOAT" << std::endl;

	for (unsigned i = 0; i < cpuMesh.numNodes; ++i)
		scalarsOut << std::fixed << cpuMesh.elemNode[i].coord.x << "	" << std::fixed << cpuMesh.elemNode[i].coord.y << "	" << std::fixed << 0.0f << std::endl;

	scalarsOut << std::endl;

	scalarsOut << "CELLS " << cpuMesh.numElems << "	" << cpuMesh.numElems * 4 << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		scalarsOut << 3 << "	" << cpuMesh.elem[i].meta->node[0]->id << "	" << cpuMesh.elem[i].meta->node[1]->id << "	" << cpuMesh.elem[i].meta->node[2]->id << std::endl;

	scalarsOut << std::endl;

	scalarsOut << "CELL_TYPES " << cpuMesh.numElems << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		scalarsOut << 5 << std::endl;

	scalarsOut << std::endl;

	scalarsOut << "CELL_DATA " << cpuMesh.numElems << std::endl;

	for (unsigned scl = 0; scl < maxSCALARS; ++scl){

		if (scl < maxFRACTIONS){
			scalarsOut << "SCALARS Conc" << (scl + 1) << " FLOAT" << std::endl;
			scalarsOut << "LOOKUP_TABLE default" << std::endl;
			for (unsigned i = 0; i < cpuMesh.numElems; ++i)
				scalarsOut << std::fixed << cpuMesh.elemOutput[i].scalars[scl] << std::endl;

			scalarsOut << std::endl;
		}else if (scl == relTemp){
			for (unsigned p = 0; p <= maxFRACTIONS; ++p){
				scalarsOut << "SCALARS Temperature FLOAT" << std::endl;
				scalarsOut << "LOOKUP_TABLE default" << std::endl;
				for (unsigned i = 0; i < cpuMesh.numElems; ++i)
					scalarsOut << std::fixed << cpuMesh.elemOutput[i].scalars[relTemp] << std::endl;

				scalarsOut << std::endl;
			}
		}
	}

	scalarsOut.close();
	std::cout << " Done. " << std::endl;
}

void outputParameters::exportBedComposition(){

	std::ofstream sediOut;
	std::ostringstream sedOutSstream;
	sedOutSstream << "./output/sediment/bedComp-" << cpuSimulation.step << ".vtk";
	std::string sediOutFile = sedOutSstream.str();

	std::cout << "   Writing " << sediOutFile.c_str() << " ... ";

	sediOut.open(sediOutFile.c_str());
	sediOut << "# vtk DataFile Version 2.0" << std::endl;
	sediOut << "STAV-2D solver at " << cpuSimulation.currentTime << " secs." << std::endl;
	sediOut << "ASCII" << std::endl;
	sediOut << "DATASET UNSTRUCTURED_GRID" << std::endl;

	sediOut << std::endl;

	sediOut << "POINTS " << cpuMesh.numNodes << " FLOAT" << std::endl;

	for (unsigned i = 0; i < cpuMesh.numNodes; ++i)
		sediOut << std::fixed << cpuMesh.elemNode[i].coord.x << "	" << std::fixed << cpuMesh.elemNode[i].coord.y << "	" << std::fixed << 0.0f << std::endl;

	sediOut << std::endl;

	sediOut << "CELLS " << cpuMesh.numElems << "	" << cpuMesh.numElems * 4 << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		sediOut << 3 << "	" << cpuMesh.elem[i].meta->node[0]->id << "	" << cpuMesh.elem[i].meta->node[1]->id << "	" << cpuMesh.elem[i].meta->node[2]->id << std::endl;

	sediOut << std::endl;

	sediOut << "CELL_TYPES " << cpuMesh.numElems << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		sediOut << 5 << std::endl;

	sediOut << std::endl;

	sediOut << "CELL_DATA " << cpuMesh.numElems << std::endl;
	sediOut << "SCALARS Bed_Var FLOAT" << std::endl;
	sediOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		sediOut << std::fixed << (cpuMesh.elemOutput[i].z - cpuMesh.elemOutput[i].zIni) << std::endl;

	sediOut << std::endl;

	for (unsigned short p = 0; p < maxFRACTIONS; ++p){
		sediOut << "SCALARS Frac_" << p << " FLOAT" << std::endl;
		sediOut << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			sediOut << std::fixed << cpuMesh.elemOutput[i].perc[p] << std::endl;

		sediOut << std::endl;
	}

	sediOut << "SCALARS D_eq FLOAT" << std::endl;
	sediOut << "LOOKUP_TABLE default" << std::endl;
	float outputDeq = 0.0f;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		outputDeq = 0.0f;
		for (unsigned short p = 0; p < maxFRACTIONS; ++p)
			outputDeq += cpuMesh.elemOutput[i].perc[p] * cpuBed.grain[p].diam;

		sediOut << std::fixed << outputDeq << std::endl;
	}

	sediOut.close();
	std::cout << " Done. " << std::endl;
}

void outputParameters::exportMaxima(){

	std::ofstream maxiOut;
	std::ostringstream maxiOutSstream;
	maxiOutSstream << "./output/maxima/maxi-" << cpuSimulation.step << ".vtk";
	std::string maxiOutFile = maxiOutSstream.str();

	std::cout << "   Writing " << maxiOutFile.c_str() << " ... ";

	maxiOut.open(maxiOutFile.c_str());
	maxiOut << "# vtk DataFile Version 2.0" << std::endl;
	maxiOut << "STAV-2D solver at " << cpuSimulation.currentTime << " secs." << std::endl;
	maxiOut << "ASCII" << std::endl;
	maxiOut << "DATASET UNSTRUCTURED_GRID" << std::endl;

	maxiOut << std::endl;

	maxiOut << "POINTS " << cpuMesh.numNodes << " FLOAT" << std::endl;

	for (unsigned i = 0; i < cpuMesh.numNodes; ++i){
		maxiOut << std::fixed << cpuMesh.elemNode[i].coord.x << "	" << std::fixed << cpuMesh.elemNode[i].coord.y << "	" << std::fixed << 0.0f << std::endl;
	}

	maxiOut << std::endl;

	maxiOut << "CELLS " << cpuMesh.numElems << "	" << cpuMesh.numElems * 4 << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		maxiOut << 3 << "	" << cpuMesh.elem[i].meta->node[0]->id << "	" << cpuMesh.elem[i].meta->node[1]->id << "	" << cpuMesh.elem[i].meta->node[2]->id << std::endl;
	}

	maxiOut << std::endl;

	maxiOut << "CELL_TYPES " << cpuMesh.numElems << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << 5 << std::endl;

	maxiOut << std::endl;

	maxiOut << "CELL_DATA " << cpuMesh.numElems << std::endl;
	maxiOut << "SCALARS Max_Depth FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.maxDepth << std::endl;

    maxiOut << std::endl;

    maxiOut << "SCALARS Max_Level FLOAT" << std::endl;
    maxiOut << "LOOKUP_TABLE default" << std::endl;
    for (unsigned i = 0; i < cpuMesh.numElems; ++i){
        float level = 0.0f;
        if(cpuMesh.elemOutput[i].max.maxDepth > cpuControl.numerics->lowDepth){
            float depth = cpuMesh.elemOutput[i].max.maxDepth;
            float bed = cpuMesh.elemOutput[i].max.zIni;
            level = depth + bed;
        }
        maxiOut << std::fixed << level << std::endl;
    }

	maxiOut << std::endl;

	maxiOut << "SCALARS Max_Vel FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.maxVel << std::endl;

	maxiOut << std::endl;

	maxiOut << "SCALARS Max_Q FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.maxQ << std::endl;

	maxiOut << std::endl;

	maxiOut << "SCALARS Time_to_Wet FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.wettingTime << std::endl;

	maxiOut << std::endl;

	maxiOut << "SCALARS  Time_to_Max_Q FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.maxQTime << std::endl;

	maxiOut << std::endl;

	maxiOut << "SCALARS Max_dZb FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.maxEro << std::endl;

	maxiOut << std::endl;

	maxiOut << "SCALARS Min_dZb FLOAT" << std::endl;
	maxiOut << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < cpuMesh.numElems; ++i)
		maxiOut << std::fixed << cpuMesh.elemOutput[i].max.maxDep << std::endl;

	maxiOut.close();
	std::cout << " Done. " << std::endl;
}
