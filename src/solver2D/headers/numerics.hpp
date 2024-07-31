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
#include <cmath>
#include <algorithm>

// STAV
#include "compile.hpp"
#include "common.hpp"
#include "control.hpp"
#include "geometry.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


class elementScalars{
public:

	CPU GPU elementScalars();
	
	CPU GPU INLINE void setTemp(const float);
	CPU GPU INLINE float getTemp();
	CPU GPU void validate();

	float specie[maxSCALARS];
};

class elementState{
public:

	CPU GPU elementState();

	CPU GPU INLINE bool isWet();

	vector2D vel;
	float h;
	float rho;
	float z;
};

class elementConnect{
public:

	CPU GPU elementConnect();

	elementState* state;
	elementScalars* scalars;
	vector2D normal;
	float length;
};

class elementFluxes{
public:

	CPU GPU elementFluxes();
	
	CPU GPU INLINE void reset();
	CPU GPU INLINE bool hasWetNeighbors();
	CPU GPU INLINE bool hasFluxes();

	elementConnect* neighbor;
	double flux[maxCONSERVED + maxSCALARS];
	double* dt;
};

class elementFlow{
public:

	CPU GPU elementFlow();

	CPU GPU INLINE void computeFluxes();
	CPU GPU INLINE void applyFluxes();
	CPU GPU INLINE void applyCorrections();

	CPU GPU INLINE void updateDensity();
	CPU GPU INLINE float getMu();
	
	CPU GPU void computeFluxes_ReducedGudonov_1stOrder();
	CPU GPU void applyFluxes_ReducedGudonov_1stOrder();

	CPU GPU void applyVelocityCorrections();
	CPU GPU void validate();

	elementState* state;
	elementScalars* scalars;
	elementFluxes* flux;

	float area;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU INLINE float elementScalars::getTemp(){

#	ifdef  __CUDA_ARCH__
	float refTemp = gpuPhysics.waterTemp;
#	else
	float refTemp = cpuPhysics.waterTemp;
#	endif

	if (relTemp < maxSCALARS)
		return refTemp * (specie[relTemp] + 1.0f);
	else
		return -273.15f;
}

CPU GPU INLINE void elementScalars::setTemp(const float absTemp){

#	ifdef  __CUDA_ARCH__
	float refTemp = gpuPhysics.waterTemp;
#	else
	float refTemp = cpuPhysics.waterTemp;
#	endif

	if (relTemp < maxSCALARS)
		specie[relTemp] = (absTemp - refTemp) / refTemp;
}

CPU GPU INLINE bool elementState::isWet(){

	if (isValid(h))
		return true;
	else
		return false;
}

CPU GPU INLINE bool elementFluxes::hasFluxes(){

	if (isValid(flux[0]))
		return true;
	else
		return false;
}

CPU GPU INLINE bool elementFluxes::hasWetNeighbors(){

	bool isWetNeighbor = false;

	if (neighbor[0].state != 0x0)
		if (isValid(neighbor[0].state->h))
			isWetNeighbor = true;

	if (neighbor[1].state != 0x0)
		if (isValid(neighbor[1].state->h))
			isWetNeighbor = true;

	if (neighbor[2].state != 0x0)
		if (isValid(neighbor[2].state->h))
			isWetNeighbor = true;

	return isWetNeighbor;
}

CPU GPU INLINE void elementFluxes::reset(){

	for (unsigned /*short*/ f = 0; f < (maxCONSERVED + maxSCALARS); ++f)
		flux[f] = 0.0f;

	*dt = 999999999999.0;
}

CPU GPU INLINE void elementFlow::computeFluxes(){

	flux->reset();
	if (state->isWet() || flux->hasWetNeighbors()){
		computeFluxes_ReducedGudonov_1stOrder();
	}
}

CPU GPU INLINE void elementFlow::applyFluxes(){

	if (state->isWet() || flux->hasFluxes()) {
		applyFluxes_ReducedGudonov_1stOrder();
	}
	validate();
}

CPU GPU INLINE void elementFlow::applyCorrections(){

	if (state->isWet()) {
		applyVelocityCorrections();
	}
	validate();
}

CPU GPU INLINE void elementFlow::updateDensity(){

#	ifdef  __CUDA_ARCH__
	float rhoWater = gpuPhysics.waterDensity;
#	else
	float rhoWater = cpuPhysics.waterDensity;
#	endif

	float totalSolidMass = 0.0f;
	for(unsigned /*short*/ p = 0; p < maxFRACTIONS; p++)
		totalSolidMass += scalars->specie[sedCp]*2.65f;

	state->rho =  rhoWater*(1.0f - totalSolidMass*(2.65f - 1.0f));
}

CPU GPU INLINE float elementFlow::getMu(){

#	ifdef  __CUDA_ARCH__
	float muWater = gpuPhysics.waterDinVis;
	float rhoWater = gpuPhysics.waterDensity;
#	else
	float muWater = cpuPhysics.waterDensity;
	float rhoWater = cpuPhysics.waterDensity;
#	endif

	return muWater * state->rho / rhoWater;
}
