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
#include <string>
#include <vector>

// Pre-Processor Headers
#include "../headers/meshEntities.hpp"
#include "../headers/meshGlobals.hpp"
#include "../headers/meshRead.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

simulationMesh::simulationMesh(){

	lowerLeft = node(999999999.0, 999999999.0, 0.0);
	upperRight = node(-999999999.0, -999999999.0, 0.0);
	minLength = 999999999.0;
	avgLength = 999999999.0;
};

void simulationMesh::setBox(){

	for (unsigned n = 0; n < geometry.domain[0].nodes.size(); ++n){
		if (geometry.domain[0].nodes[n].x < mesh.lowerLeft.x)
			mesh.lowerLeft.x = geometry.domain[0].nodes[n].x;
		if (geometry.domain[0].nodes[n].y < mesh.lowerLeft.y)
			mesh.lowerLeft.y = geometry.domain[0].nodes[n].y;
		if (geometry.domain[0].nodes[n].x > mesh.upperRight.x)
			mesh.upperRight.x = geometry.domain[0].nodes[n].x;
		if (geometry.domain[0].nodes[n].y > mesh.upperRight.y)
			mesh.upperRight.y = geometry.domain[0].nodes[n].y;
	}

	double totalEdgeLength = 0.0;

	for (unsigned k = 0; k < edges.size(); ++k){
		double edgeLength = edges[k].getLength();
		totalEdgeLength += edgeLength;
		if (edgeLength < minLength)
			minLength = edgeLength;
	}

	avgLength = totalEdgeLength / (double(edges.size()));
};

void simulationMesh::setFacetsCcWise(){

	for (unsigned i = 0; i < facets.size(); ++i){
		facets[i].setCcWise();
		showProgress((int)i, facets.size(), "Orienting", "Facets");
	}

	std::cout << std::endl;
};

simulationMesh mesh;