/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira
Instituto Superior Técnico - Universidade de Lisboa
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

// Pre-Processor Headers
#include "meshControl.hpp"
#include "meshInterpolation.hpp"
#include "meshEntities.hpp"
#include "meshBoundary.hpp"
#include "meshRead.hpp"
#include "meshGen.hpp"

// Forward Declarations
class regularGrid;

/////////////////////////////////////////////////////////////////////////////////////////////////

class simulationMesh{
public:

	simulationMesh();

	meshControl control;
	meshGen geometry;

	std::vector<node> nodes;
	std::vector<facet> facets;
	std::vector<edge> edges;
	std::vector<edge> physEdges;
	std::vector<meshBoundary> boundaries;

	Polyhedron polyCGAL;
	
	node lowerLeft;
	node upperRight;
	double minLength;
	double avgLength;

	void setBox();

	void readMeshFile();
	void setConnectivity();
	void setFacetsCcWise();
	void assignPhysicalEdges();
	void applySpfOrderingToNodes();
	void applySpfOrderingToFacets();
	void setInterpolatedData(const regularGrid&, const std::string&);

	void writeMeshInfo();
	void writeMeshNodes();
	void writeMeshFacets();
	void writeMeshBoundaries();
	void writeMeshQualityVTK();
	void writeMeshDebugVTK();
};

extern simulationMesh mesh;