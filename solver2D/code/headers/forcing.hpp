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
    CPU GPU INLINE void applyInfiltration();

	elementState* state;
	point* center;

	timeseries* rainGauge[maxNN];
	unsigned numRainGauges;
	float rainInt;

    float infilPar;
    float infilTotal;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU INLINE void elementForcing::applyRainfall(){

#	ifdef __CUDA_ARCH__
    float dt = float(gpuNumerics.dt);
	float constantRainfall = gpuForcing.constantRainfall;
	unsigned short numRainGauges = gpuForcing.numRainGauges;
	unsigned short infiltrationOption = gpuForcing.infiltrationOption;
	bool useInfiltration = gpuForcing.useInfiltration;
#	else
	float dt = float(cpuNumerics.dt);
    float constantRainfall = cpuForcing.constantRainfall;
    unsigned short numRainGauges = cpuForcing.numRainGauges;
    unsigned short infiltrationOption = cpuForcing.infiltrationOption;
    bool useInfiltration = cpuForcing.useInfiltration;
#	endif

    if(numRainGauges == 1)
        rainInt = rainGauge[0]->getData();
    else
        rainInt = constantRainfall/(1000.0f*3600.0f);

    if(useInfiltration) {
        if (infiltrationOption == 1)
            this->applyInfiltration();
        if (infiltrationOption == 2)
            rainInt *= (1.0f - infilPar/100.0f);
    }

    if(isValid(rainInt*dt))
        state->h += rainInt*dt;
}

CPU GPU INLINE void elementForcing::applyInfiltration(){

#	ifdef __CUDA_ARCH__
    float dt = float(gpuNumerics.dt);
	float infilParameter1 = gpuForcing.infilParameter1;
	float infilParameter2 = gpuForcing.infilParameter2;
	float waterDinVis = gpuPhysics.waterDinVis;
    float waterDensity = gpuPhysics.waterDensity;
    float gravity = gpuPhysics.gravity;
#	else
    float dt = float(cpuNumerics.dt);
    float infilParameter1 = cpuForcing.infilParameter1;
    float infilParameter2 = cpuForcing.infilParameter2;
    float waterDinVis = cpuPhysics.waterDinVis;
    float waterDensity = cpuPhysics.waterDensity;
    float gravity = cpuPhysics.gravity;
#	endif

    float deltaWaterContent = infilPar * (1.0f - infilParameter1);
    float infilDepth = infilTotal / (deltaWaterContent);

    float hydroConductivity = infilPar * gravity * waterDensity / waterDinVis;
    float infilRate =
            hydroConductivity * (1.0f + (abs(infilParameter2) + state->h) * deltaWaterContent / infilDepth);

    if (isValid(infilRate * dt)) {
        state->h -= infilRate * dt;
        infilTotal += infilRate * dt;
    }
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

	for (unsigned short t = present; t < length; ++t){
		if (time[t] >= simTime){
			present = (unsigned short) (t - 1);
			break;
		}
	}

    unsigned short prevTime = present;
	unsigned short nextTime = (unsigned short) (present + 1);
	return (data[prevTime] + (simTime - time[prevTime])*(data[nextTime] - data[prevTime]) / (time[nextTime] - time[prevTime]));
}
