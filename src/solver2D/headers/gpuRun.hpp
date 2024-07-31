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

// STAV
#include "compile.hpp"
#include "boundaries.hpp"
#include "numerics.hpp"
#include "forcing.hpp"
#include "sediment.hpp"
#include "mesh.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


class kernelLaunch{
public:

	kernelLaunch();

	void setKernel(void*, int);

	int gridSize;
	int blockSize;
	int minGridSize;
};


extern kernelLaunch setBndRefValues;
extern kernelLaunch setBndConditions;

extern kernelLaunch getFluxes;
extern kernelLaunch applyFluxes;
extern kernelLaunch applyCorrections;
extern kernelLaunch applySources;


GLOBAL void setBndRefValuesKernel(physicalBoundary*, int);
GLOBAL void setBndConditionsKernel(elementGhost*, int);

GLOBAL void getFluxesKernel(elementFlow*, int);
GLOBAL void applyFluxesKernel(elementFlow*, int);
GLOBAL void applyCorrectionsKernel(elementFlow*, int);
GLOBAL void applySourcesKernel(element*, int);

GLOBAL void reduceDtKernel(int, int, const double*, double*);