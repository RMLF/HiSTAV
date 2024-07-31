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


class elementGhost{
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
		float minDepthFactor = gpuNumerics.minDepthFactor;
#	else
		bool useMobileBed = cpuBed.useMobileBed;
		unsigned depEroOption = cpuBed.depEroOption;
		float maxConc = cpuBed.maxConc;
		float minDepthFactor = cpuNumerics.minDepthFactor;
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

	if (isAtEquilibrium && useMobileBed && state.h >= minDepthFactor && conditionType != bndNOACTIVE){
		for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
			
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
	float gravity = gpuPhysics.gravity;
#	else
	float gravity = cpuPhysics.gravity;
	using std::max;
	using std::sqrt;
#	endif

	state.vel = link->flow->state->vel;
	state.h = link->flow->state->h;
	state.rho = link->flow->state->rho;
	state.z = link->flow->state->z;

	float localWeight = 1.0f;
	if (isValid(*inletRefFactor))
		localWeight = std::pow(state.h*edgeLength, bndWGHTPOWER) / (*inletRefFactor);

	float outerDischarge = (hydroGauge[0]->getData()*gaugeWeight[0] + hydroGauge[1]->getData()*gaugeWeight[1]) * localWeight / edgeLength;

	if (outerDischarge >= bndTOLERANCE){

		float innerDepth = link->flow->state->h;
		float innerVel = link->flow->state->vel.dot(normal);
		//float innerCel = sqrt(gravity*innerDepth);

		float outerDepth = innerDepth;
		float outerVel = innerVel;

		double invSignal = -1.0;
		double innerInvariant = double(innerVel) + 2.0*invSignal*sqrt(double(gravity)*double(innerDepth));

		double aCoef = 4.0*double(gravity);
		double bCoef = -(innerInvariant*innerInvariant);
		double cCoef = 2.0*innerInvariant*double(outerDischarge);
		double dCoef = -double(outerDischarge)*double(outerDischarge);

		double realRoots3rdOrder[3] = { 0.0 };
		getRealRoots3rdOrder(realRoots3rdOrder, aCoef, bCoef, cCoef, dCoef);

		float innerHead = (innerVel*innerVel) / 2.0f + gravity*(innerDepth);
		float headLoss = 99999999999.0f;

		for (unsigned short k = 0; k < 3; ++k)
			if (realRoots3rdOrder[k] >= 0.0){
				float outerDepthTemp = float(realRoots3rdOrder[k]);
				float outerVelTemp = outerDischarge / outerDepthTemp;
				float outerHead = (outerVelTemp*outerVelTemp) / 2.0f + gravity*(outerDepthTemp);
				if (abs(outerHead - innerHead) < headLoss){
					headLoss = abs(outerHead - innerHead);
					outerDepth = outerDepthTemp;
					outerVel = outerVelTemp;
				}
			}

		state.vel = normal*outerVel;
		state.h = outerDepth;
	}
}

CPU GPU INLINE void elementGhost::getLevelsRankineHugoniot(){

}

CPU GPU INLINE void elementGhost::getLevelsRiemannInv(){

#	ifdef __CUDA_ARCH__
	//float gravity = gpuPhysics.gravity;
#	else
	//float gravity = cpuPhysics.gravity;
	using std::max;
	using std::sqrt;
#	endif

	state.vel.setNull();
	state.h = link->flow->state->h;
	state.rho = link->flow->state->rho;
	state.z = link->flow->state->z;

	float outerDepth = max(0.0f, (hydroGauge[0]->getData()*gaugeWeight[0] + hydroGauge[1]->getData()*gaugeWeight[1]) - state.z);
	//float innerDepth = link->flow->state->h;

	if (outerDepth >= bndTOLERANCE){

		//float outerCel = sqrt(gravity*outerDepth);
		//float outerVel = 0.0f;
		//float innerVel = -link->flow->state->vel.dot(normal);
		//float innerCel = sqrt(gravity*innerDepth);

		//outerVel = innerVel + 2.0f*innerCel - 2.0f*outerCel;

		// Tsunami Benchmarks Conical Island
		//float velocity = std::sqrt(cpuPhysics.gravity / (0.78f + eta))*eta;
		//state.vel = normal * (-outerVel);
		state.h = outerDepth;
	}
}

CPU GPU INLINE void elementGhost::getMinimalEnergy(){

	if (link->flow->state->h < bndTOLERANCE){
		state.vel.setNull();
		state.h = 0.0f;
		state.rho = link->flow->state->rho;
		state.z = link->flow->state->z;
	}else{
		state.vel.setNull();
		state.h = 0.001f;
		state.rho = link->flow->state->rho;
		state.z = link->flow->state->z - 100.0f;
	}
}

CPU GPU INLINE void elementGhost::getZeroGradient(){

	if (link->flow->state->h < bndTOLERANCE){
		state.vel.setNull();
		state.h = 0.0f;
		state.rho = link->flow->state->rho;
		state.z = link->flow->state->z;
	}else{
		state.vel = link->flow->state->vel;
		state.h = link->flow->state->h;
		state.rho = link->flow->state->rho;
		state.z = link->flow->state->z;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU void getRoots3rdOrder(localCOMPLEX(&y)[3], double, double, double, double);
CPU GPU void getRoots4thOrder(localCOMPLEX(&x)[4], double, double, double, double, double);
