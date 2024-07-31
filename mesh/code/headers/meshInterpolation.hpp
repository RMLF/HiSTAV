/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira
Instituto Superior Tecnico - Universidade de Lisboa
Av. Rovisco Pais 1, 1049-001 Lisboa, Portugal

This file is part of STAV-2D

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
#include "meshGlobals.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


class regularGrid{
public:

	regularGrid();

	void clear();
	void allocateRaster();
	void importRaster(const std::string&, const std::string&);

	double dx;
	double dy;

	double rootX;
	double rootY;

	unsigned nLin;
	unsigned nCol;

	std::vector< std::vector<double> > rasterVal;
};