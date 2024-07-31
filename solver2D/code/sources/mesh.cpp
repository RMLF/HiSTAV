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
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

// STAV
#include "../headers/mesh.hpp"
#include "../headers/geometry.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


elementNode::elementNode(){

	fricPar = 0.0f;
	bedrockOffset = 0.0f;
	landslideDepth = 0.0f;

	id = 0;
	curveID = 0;
	landslideID = 0;
}

elementMeta::elementMeta(){

	node[0] = 0x0;
	node[1] = 0x0;
	node[2] = 0x0;

	elem[0] = 0x0;
	elem[1] = 0x0;
	elem[2] = 0x0;

	id = 0;
	ownerID = 0;
	owner = 0;

#	ifdef __STAV_MPI__
	isConnected = false;
#	endif
}

CPU GPU element::element(){

	flow = 0x0;
	bed = 0x0;
	forcing = 0x0;

	meta = 0x0;
}

CPU GPU simulationMesh::simulationMesh(){

    numNodes = 0;
    numElems = 0;

#	ifdef __STAV_MPI__
    numRemoteElems = 0;
	numConnectedElems = 0;
#	endif

	elemNode = 0x0;

	elemFlow = 0x0;
	elemState = 0x0;
	elemFluxes = 0x0;
	elemConnect = 0x0;
	elemScalars = 0x0;

 	elemBed = 0x0;
 	elemBedComp = 0x0;
 	elemForcing = 0x0;
 	elemMeta = 0x0;

 	elemDt = 0x0;
 	elem = 0x0;

 	elemOutput = 0x0;

#	ifdef __STAV_CUDA__
	gpuElemFlow = 0x0;
	gpuElemState = 0x0;
	gpuElemScalars = 0x0;
	gpuElemConnect = 0x0;
	gpuElemFluxes = 0x0;
	gpuElemDt = 0x0;

	gpuElemBed = 0x0;
	gpuElemBedComp = 0x0;
	gpuElemForcing = 0x0;

	gpuElem = 0x0;
#	endif

#	ifdef __STAV_MPI__
	elemConnected = 0x0;
	elemConnectedState = 0x0;
	elemConnectedScalars = 0x0;

	elemRemote = 0x0;
	elemRemoteState = 0x0;
	elemRemoteScalars = 0x0;
#	endif

#	ifdef __STAV_MPI__
#	ifdef __STAV_CUDA__
	gpuElemConnectedState = 0x0;
	gpuElemConnectedScalars = 0x0;

	gpuElemRemoteState = 0x0;
	gpuElemRemoteScalars = 0x0;
#	endif
#	endif
}

void element::computeGeometry(){

	// Area
	double x[3] = { 0.0 };
	double y[3] = { 0.0 };
	for (unsigned short n = 0; n < 3; ++n){
		x[n] = meta->node[n]->coord.x;
		y[n] = meta->node[n]->coord.y;
	}

	double area = std::abs(x[0]*(y[1] - y[2]) + x[1]*(y[2] - y[0]) + x[2]*(y[0] - y[1])) / 2.0;
	flow->area = float(area);

	// Center
	meta->center = (meta->node[0]->coord + meta->node[1]->coord + meta->node[2]->coord) * (1.0f / 3.0f);

	// Edge lengths and outer normals
	for (unsigned short n = 0; n < 3; ++n){
		vector2D edgeVector(meta->node[n]->coord, meta->node[(n + 1) % 3]->coord);
		flow->flux->neighbor[n].length = edgeVector.norm();
		edgeVector.normalize();

		flow->flux->neighbor[n].normal = vector2D(edgeVector.y, -edgeVector.x);
		flow->flux->neighbor[n].normal.normalize();
	}

	// Bed slope
	point A = meta->node[0]->coord;
	point B = meta->node[1]->coord;
	point C = meta->node[2]->coord;

	// Plane equation Z = a.x + b.y + c
	float a = (B.y - A.y)*(C.z - A.z) - (C.y - A.y)*(B.z - A.z);
	float b = (B.z - A.z)*(C.x - A.x) - (C.z - A.z)*(B.x - A.x);
	float c = (B.x - A.x)*(C.y - A.y) - (C.x - A.x)*(B.y - A.y);

	bed->slope = vector2D(-a/c, -b/c);
}

void simulationMesh::allocateOnCPU(){

	unsigned numLocalElems = 0;

#	ifdef __STAV_MPI__
	numLocalElems = numElems - numConnectedElems;
	if (myProc.master){
#	endif
		numLocalElems = numElems;
		elemNode = new elementNode[numNodes];
#	ifdef __STAV_MPI__
	}
#	endif

	elemFlow = new elementFlow[numElems];
	elemState = new elementState[numLocalElems];
	elemScalars = new elementScalars[numLocalElems];
	elemConnect = new elementConnect[numElems*3];
	elemFluxes = new elementFluxes[numElems];
	elemDt = new double[numElems];

	elemBed = new elementBed[numElems];
	elemBedComp = new elementBedComp[numElems];
	elemForcing = new elementForcing[numElems];

	elemMeta = new elementMeta[numElems];
	elem = new element[numElems];

#	ifdef __STAV_MPI__
	if (myProc.master)
#	endif
		elemOutput = new elementOutput[numElems];

#	ifdef __STAV_MPI__
	if (myProc.worker){
		if (numConnectedElems > 0){
			elemConnected = new elementConnected[numConnectedElems];
			elemConnectedState = new elementState[numConnectedElems];
			elemConnectedScalars = new elementScalars[numConnectedElems];
		}

		if (numRemoteElems > 0){
			elemRemote = new elementRemote[numRemoteElems];
			elemRemoteState = new elementState[numRemoteElems];
			elemRemoteScalars = new elementScalars[numRemoteElems];
		}
	}
#	endif

	for (unsigned i = 0; i < numElems; ++i){

		elem[i].flow = &elemFlow[i];

#		ifdef __STAV_MPI__
		if (myProc.master){
#		endif
			elem[i].flow->state = &elemState[i];
			elem[i].flow->scalars = &elemScalars[i];
#		ifdef __STAV_MPI__
		}
#		endif

		elem[i].flow->flux = &elemFluxes[i];
		elem[i].flow->flux->neighbor = &elemConnect[i*3];
		elem[i].flow->flux->dt = &elemDt[i];

		elem[i].bed = &elemBed[i];
		elem[i].bed->comp = &elemBedComp[i];
		elem[i].bed->flow = &elemFlow[i];

		elem[i].forcing = &elemForcing[i];
        elem[i].forcing->state = &elemState[i];
		elem[i].meta = &elemMeta[i];

		elemDt[i] = 999999999999.0;

#		ifdef __STAV_MPI__
		if (myProc.master){
#		endif
			elemOutput[i].link = &elem[i];
			elemOutput[i].max.link = &elem[i];
#		ifdef __STAV_MPI__
		}
#		endif
	}

#	ifdef __STAV_MPI__
	if (myProc.worker){
		for (unsigned i = 0; i < numConnectedElems; ++i){
			elemConnected[i].state = &elemConnectedState[i];
			elemConnected[i].scalars = &elemConnectedScalars[i];
		}

		for (unsigned i = 0; i < numRemoteElems; ++i){
			elemRemote[i].state = &elemRemoteState[i];
			elemRemote[i].scalars = &elemRemoteScalars[i];
		}
	}
#	endif
}

void simulationMesh::deallocateFromCPU(){

	delete[] elemFlow;
	delete[] elemState;
	delete[] elemScalars;
	delete[] elemConnect;
	delete[] elemFluxes;
	delete[] elemDt;

	delete[] elemBed;
	delete[] elemBedComp;
	delete[] elemForcing;
	delete[] elemMeta;
	delete[] elem;
	

#	ifdef __STAV_MPI__
	if (myProc.worker){
		if (numConnectedElems > 0){
			delete[] elemConnected;
			delete[] elemConnectedState;
			delete[] elemConnectedScalars;
		}

		if (numRemoteElems > 0){
			delete[] elemRemote;
			delete[] elemRemoteState;
			delete[] elemRemoteScalars;
		}
	}

	numConnectedElems = 0;
	numRemoteElems = 0;

#	endif

	numElems = 0;
}

simulationMesh cpuMesh;



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void simulationMesh::readTopologyFiles(std::string& meshDimFileName, std::string& nodesFileName, std::string& elementsFileName){

	std::ifstream meshInformationFile, nodesFile, elementsFile;
	std::string inputText;

	std::cout << std::endl;

	meshInformationFile.open(meshDimFileName);
	std::cout << "  -> Importing mesh ..." << std::endl;

	if (!meshInformationFile.is_open() || !meshInformationFile.good()){
		std::cerr << "   -> *Error* [M-1]: Could not open file: " + meshDimFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading mesh.info" << std::endl;

	meshInformationFile >> numNodes;
	meshInformationFile >> numElems;
	meshInformationFile.close();

	allocateOnCPU();

	nodesFile.open(nodesFileName);

	if (!nodesFile.is_open() || !nodesFile.good()){
		std::cerr << "    -> *Error* [M-2]: Could not open file " + nodesFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	for (unsigned n = 0; n < numNodes; ++n){
		elemNode[n].id = n;
		nodesFile >> elemNode[n].coord.x >> elemNode[n].coord.y >> elemNode[n].coord.z;
		showProgress(int(n), int(numNodes), "   -> Reading", "nodes	");
	}

	nodesFile.close();
	std::cout << std::endl;

	elementsFile.open(elementsFileName);

	if (!elementsFile.is_open() || !elementsFile.good()){
		std::cerr << "    -> *Error* [M-3]: Could not open file " + elementsFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	for (unsigned i = 0; i < numElems; ++i){

		unsigned elemNode0, elemNode1, elemNode2;
		int elemNeigh0, elemNeigh1, elemNeigh2;

		elem[i].meta->id = i;

		elementsFile >> elemNode0 >> elemNode1 >> elemNode2 >> elemNeigh0 >> elemNeigh1 >> elemNeigh2;

#		ifndef __STAV_MPI__
		elem[i].meta->ownerID = elem[i].meta->id;
#		endif

		elem[i].meta->node[0] = &elemNode[elemNode0];
		elem[i].meta->node[1] = &elemNode[elemNode1];
		elem[i].meta->node[2] = &elemNode[elemNode2];

		elem[i].computeGeometry();
		elem[i].flow->state->z = (1.0f / 3.0f)*(elem[i].meta->node[0]->coord.z + elem[i].meta->node[1]->coord.z + elem[i].meta->node[2]->coord.z);

		if (elemNeigh0 != -1){
			elem[i].meta->elem[0] = &elem[elemNeigh0];
			elemFlow[i].flux->neighbor[0].state = &elemState[elemNeigh0];
			elemFlow[i].flux->neighbor[0].scalars = &elemScalars[elemNeigh0];
		}

		if (elemNeigh1 != -1){
			elem[i].meta->elem[1] = &elem[elemNeigh1];
			elemFlow[i].flux->neighbor[1].state = &elemState[elemNeigh1];
			elemFlow[i].flux->neighbor[1].scalars = &elemScalars[elemNeigh1];
		}

		if (elemNeigh2 != -1){
			elem[i].meta->elem[2] = &elem[elemNeigh2];
			elemFlow[i].flux->neighbor[2].state = &elemState[elemNeigh2];
			elemFlow[i].flux->neighbor[2].scalars = &elemScalars[elemNeigh2];
		}

		showProgress(int(i), int(numElems), "   -> Reading", "elements");
	}

	elementsFile.close();
	std::cout << std::endl;

	std::cout << std::endl;
	std::cout << "   -> Mesh dimensions: " << numElems << " elements and " << numNodes << " nodes." << std::endl;

	unsigned totalMemory = unsigned(sizeof(element) + sizeof(elementFlow) + sizeof(elementState) + sizeof(elementScalars) + sizeof(elementFluxes) + sizeof(elementConnect) * 3
		+ sizeof(elementBed) + sizeof(elementBedComp) + sizeof(elementForcing) + sizeof(elementMeta) + sizeof(double) + sizeof(elementOutput))*numElems
		+ unsigned(sizeof(elementNode))*numNodes;

	std::cout << "   -> Allocated memory: " << totalMemory / (1e6) << " MB of RAM." << std::endl;
	std::cout << std::endl;
}
