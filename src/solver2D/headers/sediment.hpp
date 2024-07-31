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
#include <algorithm>

// STAV
#include "compile.hpp"
#include "common.hpp"
#include "control.hpp"
#include "geometry.hpp"
#include "numerics.hpp"

// Forward Declarations
class element;

/////////////////////////////////////////////////////////////////////////////////////////////////


class elementBedComp{
public:

	CPU GPU elementBedComp();

	float bedPercentage[maxFRACTIONS];
	float subPercentage[maxFRACTIONS];
};

class elementBed{
public:

	CPU GPU elementBed();

	CPU GPU void applyMobileBed();
	CPU GPU void computeBedFriction();
	CPU GPU void applyBedFriction();
	CPU GPU float getShields();

	elementBedComp* comp;
	elementFlow* flow;
	
	vector2D tau;
	vector2D slope;

	float fricPar;
	float bedrock;
};

class bedLandslide{
public:

	CPU GPU bedLandslide();

	CPU GPU void applyLandslide();

	element* elemLink;

	float landslideTimeTrigger;
	float landslideDepth;

	unsigned numElems;
	bool isTriggered;

#	ifdef __STAV_CUDA__
		void copyToGPU();
		void deleteFromGPU();
#	endif
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU INLINE float getAvgDiam(const float* sedConc){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
#	endif

	float avgDiameter = 0.0f;
	float avgDiameterConcSum = 0.0f;
	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		avgDiameter += grain[p].diam*sedConc[p];
		avgDiameterConcSum += sedConc[p];
	}

	return (avgDiameter / avgDiameterConcSum);
}

CPU GPU INLINE float getAvgBedDiam(const float* bedSedPercentage){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
#	endif

	float avgBedDiameter = 0.0f;
	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p)
		avgBedDiameter += grain[p].diam*bedSedPercentage[p];

	return avgBedDiameter;
}

CPU GPU INLINE float getAvgConc(const float* sedConc){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
#	endif

	float avgConcentration = 0.0f;
	float avgConcDiameterSum = 0.0f;
	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		avgConcentration += grain[p].diam*sedConc[p];
		avgConcDiameterSum += grain[p].diam;
	}

	return (avgConcentration / avgConcDiameterSum);
}

CPU GPU INLINE float getActiveDepth(const float* bedSedPercentage, float percentile){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
#	endif

	float activeDepth = 0.0f;
	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p)
		if (bedSedPercentage[p] >= (percentile / 100.0f) && grain[p].diam >= activeDepth){
			float activeDepth = 2.0f*grain[p].diam;
			return activeDepth;
		}else
			return 0.0f;

	return activeDepth;
}

CPU GPU INLINE float getFallVel(unsigned p, float stateRho, float stateMu){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	float gravity = gpuPhysics.gravity;
#	else
	sedimentType* grain = cpuBed.grain;
	float gravity = cpuPhysics.gravity;
	using std::abs;
	using std::max;
	using std::sqrt;
#	endif

	float fallVelPar1 = abs(grain[p].diam*stateRho) / (4.0f*stateMu)*sqrt((grain[p].specGrav - 1.0f)*gravity*grain[p].diam);
	float fallVelPar2 = 0.0f;

	if(fallVelPar1 < 1.0f)
		fallVelPar2 = fallVelPar1 / 4.5f;
	else if(fallVelPar1 <= 150.0f)
		fallVelPar2 = 1.0f / (0.954f + 5.121f / fallVelPar1);
	else
		fallVelPar2 = 1.83f;
	
	return max(1.0e-3f, sqrt((grain[p].specGrav - 1.0f)*gravity*grain[p].diam)*fallVelPar2);
}

CPU GPU INLINE float getAvgFallVel(const float* sedConc, float stateRho, float stateMu){

	float avgFallVelocity = 0.0f;
	float avgFallVelocityConcSum = 0.0f;
	for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p){
		avgFallVelocity += getFallVel(p, stateRho, stateMu)*sedConc[sedCp];
		avgFallVelocityConcSum += sedConc[sedCp];
	}

	return (avgFallVelocity / avgFallVelocityConcSum);
}

CPU GPU INLINE float getTauYield(){
	return 0.0f;
}

CPU GPU INLINE float getShieldsParameter(unsigned p, float tauNorm, float stateRho){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	float gravity = gpuPhysics.gravity;
#	else
	sedimentType* grain = cpuBed.grain;
	float gravity = cpuPhysics.gravity;
	using std::abs;
#	endif

	return abs(tauNorm) / (stateRho*(grain[p].specGrav - 1.0f)*gravity*grain[sedCp].diam);
}


CPU GPU INLINE float getContactLayerDepthFerreira(unsigned p, float stateDepth, float stateVelNorm, float tauNorm, float stateRho){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	unsigned depEroOption = gpuBed.depEroOption;
#	else
	sedimentType* grain = cpuBed.grain;
	unsigned depEroOption = cpuBed.depEroOption;
	using std::abs;
#	endif

	float contactTheta = getShieldsParameter(sedCp, tauNorm, stateRho);
	float contactCoefB = (0.25f*grain[sedCp].beta + grain[sedCp].alpha) / 0.25f;
	float contactCoefA = grain[p].beta - contactCoefB;

	float contactDepth = 0.0f;

	if (stateVelNorm <= 1e-4f || depEroOption == 2){
		contactDepth = 0.0f;
	}else{
		if(contactTheta <= 0.5f)
			contactDepth = grain[p].diam*(contactCoefA*(contactTheta*contactTheta) + contactCoefB*abs(contactTheta));
		else if(contactTheta > 0.5f)
			contactDepth = grain[p].diam*(grain[sedCp].beta*contactTheta + grain[sedCp].alpha);
	}

	if(contactDepth >= stateDepth)
		return stateDepth;
	else
		return contactDepth;
}

CPU GPU INLINE float getContactLayerVelocityFerreira(float stateDepth, float stateVelNorm, float contactDepth){

#	ifndef __CUDA_ARCH__
	using std::abs;
	using std::pow;
#	endif

	if(abs(stateVelNorm) <= 1.0e-4f)
		return 0.0f;
	else
		if(abs(stateDepth) <= 1.0e-8f)
			return stateVelNorm;
		else
			return stateVelNorm*pow(abs(contactDepth / stateDepth), (1.0f / 6.0f));
}

CPU GPU INLINE float getEquilibriumConcentrationFerreira(unsigned p, float shieldsParameter, float stateVelNorm){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
	using std::abs;
#	endif

	if(abs(stateVelNorm) <= 1.0e-4f)
		return 0.0f;
	else
		return abs(shieldsParameter) / (grain[p].tanPhi*(grain[p].alpha + grain[p].beta*abs(shieldsParameter)));
}

CPU GPU INLINE float getAdaptLenCanelas(unsigned p, float tauNorm, float stateRho){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
	using std::abs;
	using std::atan;
	using std::min;
	using std::max;
#	endif

	float canelasTheta = getShieldsParameter(p, tauNorm, stateRho);
	float canelasAux1 = (grain[p].adaptLenMaxMult*grain[p].diam - grain[p].adaptLenMinMult*grain[p].diam) / (3.1415f - abs(atan(-grain[p].adaptLenRefShields*grain[p].adaptLenShapeFactor)));
	float canelasAux2 = canelasAux1*(atan((canelasTheta - grain[p].adaptLenRefShields)*grain[p].adaptLenShapeFactor) + abs(atan(-grain[p].adaptLenRefShields*grain[p].adaptLenShapeFactor))) + grain[p].adaptLenMinMult*grain[p].diam;
	float adaptLenCanelas = min(grain[p].adaptLenMaxMult*grain[p].diam, max(grain[p].adaptLenMinMult*grain[p].diam, canelasAux2));
	denoise(adaptLenCanelas);
	return adaptLenCanelas;
}

CPU GPU INLINE float getAdaptLenArmini(unsigned p, float stateDepth, float stateVelNorm, float tauNorm, float stateRho, float stateMu){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
#	else
	sedimentType* grain = cpuBed.grain;
	using std::abs;
	using std::max;
	using std::sqrt;
	using std::pow;
	using std::exp;
#	endif

	float uStarArmanini = sqrt(tauNorm / stateRho);
	float contactDepthArmanini = getContactLayerDepthFerreira(p, stateDepth, stateVelNorm, tauNorm, stateRho);
	float adaptLenArmanini = contactDepthArmanini / stateDepth + (1.0f - contactDepthArmanini / stateDepth)*exp(-1.5f*getFallVel(p, stateRho, stateMu) / uStarArmanini*pow(contactDepthArmanini / stateDepth, -1.0f / 6.0f));
	adaptLenArmanini = max(20.0f*grain[p].diam, abs(stateDepth*tauNorm*adaptLenArmanini / getFallVel(p, stateRho, stateMu)));
	denoise(adaptLenArmanini);
	return adaptLenArmanini;
}

CPU GPU INLINE float getEqDischFerreira(unsigned p, float stateDepth, float stateVelNorm, float tauNorm, float stateRho){

#	ifdef __CUDA_ARCH__
	float maxConc = gpuBed.maxConc;
#	else
	float maxConc = cpuBed.maxConc;
	using std::max;
	using std::min;
#	endif

	float shieldsPar = getShieldsParameter(p, tauNorm, stateRho);
	float contactLayerDepth = getContactLayerDepthFerreira(p, stateDepth, stateVelNorm, tauNorm, stateRho);
	float contactLayerVelocity = getContactLayerVelocityFerreira(stateDepth, stateVelNorm, contactLayerDepth);
	float equilibriumQs = max(0.0f, min(maxConc, getEquilibriumConcentrationFerreira(p, shieldsPar, stateVelNorm)))*contactLayerDepth*contactLayerVelocity;
	denoise(equilibriumQs);
	return equilibriumQs;
}

CPU GPU INLINE float getEqDischBagnold(unsigned p, float stateDepth, float stateVelNorm, float tauNorm, float stateRho, float stateMu){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	float maxConc = gpuBed.maxConc;
#	else
	sedimentType* grain = cpuBed.grain;
	float maxConc = cpuBed.maxConc;
	using std::max;
	using std::min;
#	endif

	float shieldsPar = getShieldsParameter(p, tauNorm, stateRho);
	float equilibriumQs = (grain[p].specGrav - 1.0f)*grain[p].diam*stateVelNorm*shieldsPar*(0.17f + 0.01f*stateVelNorm / getFallVel(p, stateRho, stateMu));
	equilibriumQs = max(0.0f, min(maxConc * stateVelNorm * stateDepth, equilibriumQs));

	if((equilibriumQs != equilibriumQs) || !isValid(equilibriumQs))
		equilibriumQs = 0.0f;

	denoise(equilibriumQs);
	return equilibriumQs;
}

CPU GPU INLINE float getEqDischMPM(unsigned p, float stateDepth, float stateVelNorm, float tauNorm, float stateRho){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	float gravity = gpuPhysics.gravity;
	float maxConc = gpuBed.maxConc;
#	else
	sedimentType* grain = cpuBed.grain;
	float gravity = cpuPhysics.gravity;
	float maxConc = cpuBed.maxConc;
	using std::max;
	using std::min;
	using std::sqrt;
	using std::pow;
#	endif

	float shieldsPar = getShieldsParameter(p, tauNorm, stateRho);
	float equilibriumQs = 0.0f;
	float shieldsParCritical = 0.05f;

	if(shieldsPar >= shieldsParCritical){
		equilibriumQs = 8.0f*pow((shieldsPar - shieldsParCritical), (3.0f / 2.0f))*grain[p].diam*sqrt(grain[p].diam*(grain[p].specGrav - 1.0f)*gravity);
		equilibriumQs = max(0.0f, min(maxConc*stateVelNorm*stateDepth, equilibriumQs));
	}else
		equilibriumQs = 0.0f;

	if((equilibriumQs != equilibriumQs) || !isValid(equilibriumQs))
		equilibriumQs = 0.0f;

	denoise(equilibriumQs);
	return equilibriumQs;
}

CPU GPU INLINE float getEqDischSmart(unsigned p, float stateDepth, float stateVelNorm, float tauNorm, float stateRho){

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	float gravity = gpuPhysics.gravity;
	float maxConc = gpuBed.maxConc;
#	else
	sedimentType* grain = cpuBed.grain;
	float gravity = cpuPhysics.gravity;
	float maxConc = cpuBed.maxConc;
	using std::max;
	using std::min;
	using std::sqrt;
	using std::pow;
#	endif

	float shieldsPar = getShieldsParameter(p, tauNorm, stateRho);
	float equilibriumQs = 0.0f;
	float shieldsParCritical = 0.05f;

	if(shieldsPar > (shieldsParCritical + 1.0e-8f)){
		equilibriumQs = 8.0f*pow((shieldsPar - shieldsParCritical), (3.0f / 2.0f))*grain[p].diam*sqrt(grain[p].diam*(grain[p].specGrav - 1.0f)*gravity);
		equilibriumQs = max(0.0f, min(maxConc*stateVelNorm*stateDepth, equilibriumQs));
	}else
		equilibriumQs = 0.0f;

	if((equilibriumQs != equilibriumQs) || !isValid(equilibriumQs))
		equilibriumQs = 0.0f;

	denoise(equilibriumQs);
	return equilibriumQs;
}
