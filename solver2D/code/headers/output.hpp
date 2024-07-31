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
#include "geometry.hpp"
#include "mesh.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


class elementMaximum{
public:

	elementMaximum();
	void update();

	element* link;

	float maxVel;
	float maxQ;
	float maxQTime;
	float maxDepth;
	float maxDepthTime;
	float wettingTime;
	float maxDep;
	float maxEro;
	float zIni;
};

class elementOutput{
public:

	elementOutput();
	void update();

	elementMaximum max;
	element* link;
	
	vector2D vel;
	float h;
	float rho;
	float z;
	float zIni;

	float scalars[maxSCALARS];
	float perc[maxFRACTIONS];
};
