//region >> Copyright, doxygen, includes and definitions
/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde & Rui M. L. Ferreira
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
#include <utility>
#include <complex>

// CUDA
#ifdef __STAV_CUDA__
#include <thrust/complex.h>
#endif

// STAV
#include "compile.hpp"
#include "common.hpp"
#include "geometry.hpp"
#include "control.hpp"
#include "numerics.hpp"
#include "sediment.hpp"
#include "mesh.hpp"

// Definitions
#define bndNOACTIVE 0
#define bndDISCHCUR 1
#define bndDISCHRIV 2
#define bndLEVELRIV 3
#define bndCRITICAL 4
#define bndZEROGRAD 5

// Not working
#define bndDISCHRHG 998
#define bndLEVELRHG 999
// Not working

#define bndTOLERANCE 1.0e-6f
#define bndWGHTPOWER 1.5f

#define maxDISCHCURLENGTH 5000

#ifdef __CUDA_ARCH__
#define localCOMPLEX thrust::complex<double>
#else
#define localCOMPLEX std::complex<double>
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////
//endregion


class  elementGhost{
public:

	CPU GPU elementGhost();

	CPU GPU INLINE void getConditions();
	
	CPU GPU INLINE void getWallReflection();
	CPU GPU INLINE void getDischargesStageCurve();
	CPU GPU INLINE void getDischargesRiemannInv();
	CPU GPU INLINE void getLevelsRiemannInv();
	CPU GPU INLINE void getMinimalEnergy();
	CPU GPU INLINE void getZeroGradient();

	CPU GPU INLINE void getDischargesRankineHugoniot();
	CPU GPU INLINE void getLevelsRankineHugoniot();

#	ifdef __STAV_MPI__
	void sendToOwner();
	void recvFromMaster();
#	endif

	elementState state;
	elementScalars scalars;
	element* link;

	timeseries* hydroGauge[2];
	timeseries* sediGauge[2];

	float* inletRefValue;
	float* inletRefFactor;

	vector2D normal;
	float gaugeWeight[2];
	float edgeLength;

	unsigned short conditionType;

	bool isAtEquilibrium;
};

class physicalBoundary{
public:

	physicalBoundary();
	void setUniformInlet();
	CPU GPU void setRefValue();

	elementGhost* elemGhost;

	float* inletRefValue;
	float* inletRefFactor;

	float inletCurveLevel[maxDISCHCURLENGTH];
	float inletCurveDischarge[maxDISCHCURLENGTH];

	float referenceSlope;
	float bufferRadious;
	
	unsigned numElemGhosts;

	unsigned short numGauges;
	unsigned short conditionType;
	unsigned short id;

	bool isAtEquilibrium;
	bool isUniformInlet;
};

class domainBoundaries{
public:

	domainBoundaries();

	void readControlFiles(std::string&, std::string&, std::string&);
	void readMeshFiles(std::string&);

#	ifdef __STAV_CUDA__
	void copyToGPU();
	void deallocateFromGPU();
#	endif

	timeseries* hydroGauge;
	timeseries* sediGauge;

	physicalBoundary* physical;
	elementGhost* elemGhost;

	unsigned numElemGhosts;

	unsigned short numGauges;
	unsigned short numBoundaries;

	bool hasUniformInlets;
};

extern domainBoundaries cpuBoundaries;

extern std::vector<float> cpuInletRefValue;
extern std::vector<float> cpuInletRefFactor;

#ifdef __STAV_MPI__
extern std::vector<float> cpuInletRefValueBuffer;
extern std::vector<float> cpuInletRefFactorBuffer;
#endif

#ifdef __STAV_CUDA__
extern physicalBoundary* gpuPhysicalBoundaries;
extern elementGhost* gpuElemGhost;
extern timeseries* gpuHydroGauge;
extern timeseries* gpuSediGauge;
extern float* gpuInletRefValue;
extern float* gpuInletRefFactor;
#endif



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////

CPU GPU void getRealRoots3rdOrder(double*, double, double, double, double); // Awfull... big no!

CPU GPU INLINE void elementGhost::getConditions(){

#	ifdef __CUDA_ARCH__
		bool useMobileBed = gpuBed.useMobileBed;
		unsigned depEroOption = gpuBed.depEroOption;
		float maxConc = gpuBed.maxConc;
		float lowDepth = gpuNumerics.lowDepth;
#	else
		bool useMobileBed = cpuBed.useMobileBed;
		unsigned depEroOption = cpuBed.depEroOption;
		float maxConc = cpuBed.maxConc;
		float lowDepth = cpuNumerics.lowDepth;
		using std::max;
		using std::min;
#	endif

	if (conditionType == bndNOACTIVE)
		getWallReflection();
	else if (conditionType == bndDISCHCUR)
		getDischargesStageCurve();
	else if (conditionType == bndDISCHRHG)
		getDischargesRankineHugoniot();
	else if (conditionType == bndDISCHRIV)
		getDischargesRiemannInv();
	else if (conditionType == bndLEVELRHG)
		getLevelsRankineHugoniot();
	else if (conditionType == bndLEVELRIV)
		getLevelsRiemannInv();
	else if (conditionType == bndCRITICAL)
		getMinimalEnergy();
	else if (conditionType == bndZEROGRAD)
		getZeroGradient();

	if (isAtEquilibrium && useMobileBed && state.h >= lowDepth && conditionType != bndNOACTIVE){
		for (unsigned short p = 0; p < maxFRACTIONS; ++p){
			
			float equilibriumQs = 0.0f;
			
			float innerVel = link->flow->state->vel.norm();
			float innerDepth = link->flow->state->h;
			float innerTau = link->bed->tau.norm();
			float innerRho = link->flow->state->rho;
			float innerMu = link->flow->getMu();

			if (depEroOption == 1)
				equilibriumQs = getEqDischFerreira(p, innerDepth, innerVel, innerTau, innerRho);
			else if (depEroOption == 2)
				equilibriumQs = getEqDischBagnold(p, innerDepth, innerVel, innerTau, innerRho, innerMu);
			else if (depEroOption == 3)
				equilibriumQs = getEqDischMPM(p, innerDepth, innerVel, innerTau, innerRho);
			else if (depEroOption == 4)
				equilibriumQs = getEqDischSmart(p, innerDepth, innerVel, innerTau, innerRho);

			scalars.specie[sedCp] = max(0.0f, min(maxConc, equilibriumQs / (innerDepth * innerVel)));
		}
	}
}

CPU GPU INLINE void elementGhost::getWallReflection(){
    state.h = link->flow->state->h;
    state.z = link->flow->state->z;
    state.vel.setNull();
}

CPU GPU INLINE void elementGhost::getDischargesStageCurve(){

#	ifndef __CUDA_ARCH__
	using std::max;
	using std::pow;
	using std::sqrt;
#	endif

	state.h = max(0.0f, *inletRefValue - state.z);

	float localWeight = 1.0f;
	if (isValid(*inletRefFactor))
		localWeight = std::pow(state.h*edgeLength, bndWGHTPOWER) / (*inletRefFactor);

	if (state.h > bndTOLERANCE){
		float localDischarge = (hydroGauge[0]->getData()*gaugeWeight[0] + hydroGauge[1]->getData()*gaugeWeight[1]) * localWeight / edgeLength;
		state.vel = normal*(localDischarge/state.h);
	}else
		state.vel.setNull();
}

CPU GPU INLINE void elementGhost::getDischargesRankineHugoniot(){

}

CPU GPU INLINE void elementGhost::getDischargesRiemannInv(){

#	ifdef __CUDA_ARCH__
        float lowDepth = gpuNumerics.lowDepth;
        float highDepth = gpuNumerics.highDepth;
        float gravity = gpuPhysics.gravity;
#	else
        float lowDepth = cpuNumerics.lowDepth;
        float highDepth = cpuNumerics.highDepth;
        float gravity = cpuPhysics.gravity;
        using std::max;
        using std::sqrt;
        using std::cbrt;
#	endif

    if(state.h > bndTOLERANCE){

        float gaugeData = (hydroGauge[0]->getData() * gaugeWeight[0] + hydroGauge[1]->getData() * gaugeWeight[1]);
        float innerDepth = link->flow->state->h;
        float innerZ = link->flow->state->z;
        float innerVel = 0.0;
        float innerQ = 0.0;

        if(gaugeData < 0.006f) {
            innerQ = gaugeData / 0.7f;
        } else {
            innerQ = gaugeData / 0.4f;
        }

        float froude = 0.6f;
        float temp = innerQ * innerQ / (gravity * froude * froude);
        float outerDepth = cbrt(temp);

        state.vel = normal * innerQ / outerDepth;
        link->flow->state->vel = state.vel;
        state.h = outerDepth;
        link->flow->state->h = outerDepth;

    }else{
        state.h = 2.0f*bndTOLERANCE;
        state.vel.setNull();
    }

    /*if(state.h > bndTOLERANCE){

        state.vel.setNull();

        float localWeight = std::pow(state.h*edgeLength, bndWGHTPOWER) / (*inletRefFactor);
        float gaugeData = (hydroGauge[0]->getData() * gaugeWeight[0] + hydroGauge[1]->getData() * gaugeWeight[1]);
        float outerDischarge = gaugeData / edgeLength * localWeight;

        if(outerDischarge >= 0.0){

            float innerDepth = max(bndTOLERANCE, link->flow->state->h);
            float innerVel = link->flow->state->vel.dot(normal);
            float innerCel = sqrt(gravity*innerDepth);
            float innerInvariant = innerVel - 2.0f*innerCel;

            float outerDepth = innerDepth;
            float newDepth = innerDepth;

            float error =  9999999999999.0f;
            while(error > 0.0001f){
                newDepth = outerDepth -
                           (outerDischarge - 2.0f*sqrt(gravity*pow(outerDepth, 3.0f)) - innerInvariant*outerDepth)
                           / ( -3.0f*sqrt(gravity*outerDepth) - innerInvariant);
                error = abs(newDepth - outerDepth);
                outerDepth = newDepth;
            }

            float outerVel = outerDischarge / outerDepth ;
            float froude = outerVel/sqrt(gravity * outerDepth);
            if(froude >= 0.999f){
                froude = 0.999f;
                outerDepth = cbrt(pow(outerDischarge, 2.0f) / (gravity * pow(froude, 2.0f)));
                outerVel = outerDischarge / outerDepth;
                if(outerVel >= 2.5f){
                    outerVel = 2.5f;
                    outerDepth = outerDischarge/outerVel;
                }
            }

            if(outerDepth > lowDepth) {
                state.h = outerDepth;
                float wetDryCoef = getWetCoef(state.h, lowDepth, highDepth);
                state.vel = normal * outerVel * wetDryCoef;
            } else {
                state.h = outerDepth;
                state.vel.setNull();
            }
        } else {
            state.h = 2.0f*bndTOLERANCE;
            state.vel.setNull();
        }
    }else{
        state.h = 2.0f*bndTOLERANCE;
        state.vel.setNull();
    }*/
}

CPU GPU INLINE void elementGhost::getLevelsRankineHugoniot(){

}

CPU GPU INLINE void elementGhost::getLevelsRiemannInv(){

#	ifdef __CUDA_ARCH__
	float gravity = gpuPhysics.gravity;
#	else
	float gravity = cpuPhysics.gravity;
	using std::max;
	using std::sqrt;
#	endif

	state.vel.setNull();
	state.h = link->flow->state->h;
	state.rho = link->flow->state->rho;
	state.z = link->flow->state->z;

	float outerDepth = max(0.0f, (hydroGauge[0]->getData()*gaugeWeight[0] + hydroGauge[1]->getData()*gaugeWeight[1]) - state.z);
	float innerDepth = link->flow->state->h;
    float innerVel = -link->flow->state->vel.dot(normal);

	if (outerDepth >= bndTOLERANCE){

		/*float outerCel = sqrt(gravity*outerDepth);
		float outerVel = 0.0f;
		float innerVel = -link->flow->state->vel.dot(normal);
		float innerCel = sqrt(gravity*innerDepth);

		outerVel = innerVel + 2.0f*innerCel - 2.0f*outerCel;*/

		//state.vel = normal * (-outerVel);
        state.vel.y = -abs(innerVel) * innerDepth / outerDepth;
		state.h = outerDepth;

        link->flow->state->vel = state.vel;
        link->flow->state->h = state.h;
	}
}

CPU GPU INLINE void elementGhost::getMinimalEnergy(){

	if (link->flow->state->h < bndTOLERANCE){
	    // dry
		state.vel.setNull();
		state.h = 0.0f;
		state.rho = link->flow->state->rho;
		state.z = link->flow->state->z;
	}else{
        // Forcing outflow
        if(-link->flow->state->vel.dot(normal) >= 0.0f)
            state.vel = link->flow->state->vel;
        else
            state.vel.setNull();

        // Set big fall
        state.h = link->flow->state->h;
		state.z = link->flow->state->z - 100.0f;
        //state.rho = link->flow->state->rho;
	}
}

CPU GPU INLINE void elementGhost::getZeroGradient(){

	if (link->flow->state->h < bndTOLERANCE){
        // dry
		state.vel.setNull();
		state.h = 0.0f;
		state.rho = link->flow->state->rho;
		state.z = link->flow->state->z;
	}else{
	    // Forcing outflow
	    if(-link->flow->state->vel.dot(normal) >= 0.0f)
            state.vel = link->flow->state->vel;
	    else
            state.vel.setNull();

        // Copy levels
		state.h = link->flow->state->h;
		state.z = link->flow->state->z;
        //state.rho = link->flow->state->rho;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU void getRoots3rdOrder(localCOMPLEX(&y)[3], double, double, double, double);
CPU GPU void getRoots4thOrder(localCOMPLEX(&x)[4], double, double, double, double, double);
