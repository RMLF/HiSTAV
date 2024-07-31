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
#include <string>

// STAV
#include "compile.hpp"
#include "common.hpp"
#include "control.hpp"
#include "numerics.hpp"
#include "simulation.hpp"

#define maxNN 5

/////////////////////////////////////////////////////////////////////////////////////////////////


class elementForcing{
public:

	CPU GPU elementForcing();

	CPU GPU INLINE void applyRainfall();

	elementState* state;
	point* center;

	timeseries* rainGauge[maxNN];
	unsigned numRainGauges;
	float rainInt;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU INLINE void elementForcing::applyRainfall(){

#	ifdef __CUDA_ARCH__
	float dt = float(gpuNumerics.dt);
#	else
	float dt = float(cpuNumerics.dt);
#	endif

	rainInt = rainGauge[0]->getData();
	if(isValid(rainInt*dt) && rainInt*dt > 0.0f)
		state->h += rainInt*dt;
}

CPU GPU INLINE float timeseries::getData(){

#	ifdef __CUDA_ARCH__
	float simTime = gpuCurrentTime;
#	else
	float simTime = cpuSimulation.currentTime;
#	endif

	if (simTime <= time[0])
		return data[0];
	else if (simTime >= time[length - 1])
		return data[length - 1];

	for (unsigned /*short*/ t = present; t < length; ++t){
		if (time[t] >= simTime){
			present = (unsigned /*short*/) (t - 1);
			break;
		}
	}

	unsigned /*short*/ prevTime = present;
	unsigned /*short*/ nextTime = (unsigned /*short*/) (present + 1);
	return (data[prevTime] + (simTime - time[prevTime])*(data[nextTime] - data[prevTime]) / (time[nextTime] - time[prevTime]));
}
