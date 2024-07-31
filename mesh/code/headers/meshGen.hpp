/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira
Instituto Superior Tecnico - Universidade de Lisboa
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
#include "meshEntities.hpp"
#include "meshBoundary.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


class meshGeometryType{
public:

	meshGeometryType();

	std::vector<node> nodes;
	double charLength;
	double auxField;
};

class meshGen{
public:

	meshGen();
	
	void importShapeFile(const std::string&, std::vector<meshGeometryType>&);
	void importBoundaryShapeFiles();

	void writeGmshFile();
	void callGmsh();

	std::string shapesFolder;

	std::string domainFileName;
	std::string refinementsFileName;
	std::string alignmentsFileName;
	std::string voidsFileName;
	std::string boundariesFileName;
	std::string boundaryPointsFileName;

	std::string gmshFolder;

	std::string gmshExe;
	std::string gmshGeoFile;
	std::string gmshMeshFile;

	std::vector<meshGeometryType> domain;
	std::vector<meshGeometryType> refinements;
	std::vector<meshGeometryType> alignments;
	std::vector<meshGeometryType> voids;
	std::vector<meshBoundary> boundaries;
};