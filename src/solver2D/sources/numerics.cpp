/*	STAV-2D Hydrodynamic Model

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
along with this program. If not, see http://www.gnu.org/licenses/.	*/


/////////////////////////////////////////////////////////////////////////////////////////////////

// STL
#include <algorithm>

// STAV
#include "../headers/numerics.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU elementScalars::elementScalars(){
	for (unsigned s = 0; s < maxSCALARS; ++s)
		specie[s] = 0.0f;
}

CPU GPU elementState::elementState(){
	h = 0.0f;
	rho = 1000.0f;
	z = 0.0f;
}

CPU GPU elementConnect::elementConnect(){
	state = 0x0;
	scalars = 0x0;
	length = 0.0f;
}

CPU GPU elementFluxes::elementFluxes(){
	neighbor = 0x0;
	dt = 0x0;
	for (unsigned /*short*/ f = 0; f < (maxCONSERVED + maxSCALARS); ++f)
		flux[f] = 0.0f;
}

CPU GPU elementFlow::elementFlow(){
	state = 0x0;
	scalars = 0x0;
	flux = 0x0;
	area = 0.0f;
}

CPU GPU void elementFlow::computeFluxes_ReducedGudonov_1stOrder(){

#	ifdef __CUDA_ARCH__
	float localGRAV = gpuPhysics.gravity;
	float localMinDEPTH = gpuNumerics.minDepthFactor;
	float CFL = gpuNumerics.CFL;
#	else
	float localGRAV = cpuPhysics.gravity;
	float localMinDEPTH = cpuNumerics.minDepthFactor;
	float CFL = cpuNumerics.CFL;
	using std::abs;
	using std::min;
	using std::max;
	using std::sqrt;
#	endif

	double myDt = 999999999999.0;
	float myArea = area;

	vector2D myVel = state->vel;
	float myDepth = state->h;
	//float myDensity = state->rho;
	float myZb = state->z;

	float myScalars[maxSCALARS];
	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		myScalars[s] = scalars->specie[s];

	for (unsigned /*short*/ k = 0; k < 3; ++k){

		bool isWall = false;

		vector2D adjNormal = flux->neighbor[k].normal;
		float adjLength = flux->neighbor[k].length;

		float myNormalVel = max(0.0f, myVel.dot(adjNormal));
		float myEnergy = myZb + myDepth + (myNormalVel * myNormalVel) / (2.0f*localGRAV);

		vector2D adjVel;
		float adjDepth;
		//float adjDensity;
		float adjZb;

		if (flux->neighbor[k].state != 0x0){
			adjVel = flux->neighbor[k].state->vel;
			adjDepth = flux->neighbor[k].state->h;
			//adjDensity = flux->neighbor[k].state->rho;
			adjZb = flux->neighbor[k].state->z;
			if (adjDepth < localMinDEPTH && (myEnergy <= adjZb))
				isWall = true;
		}else{
			isWall = true;
		}

		if (isWall){
			vector2D adjTangent = vector2D(adjNormal.y, -adjNormal.x);
			float adjVelTangent = myVel.dot(adjTangent);
			if (adjVelTangent < 0.0){
				adjTangent = -adjTangent;
				adjVelTangent = -adjVelTangent;
			}
			adjVel = adjTangent*(2.0f*adjVelTangent) - myVel;
			adjDepth = myDepth;
			//adjDensity = myDensity;
			adjZb = myZb;
		}

		vector2D approxVel = (myVel * sqrt(myDepth) + adjVel*sqrt(adjDepth)) / (sqrt(myDepth) + sqrt(adjDepth));

		float approxVelNormal = approxVel.dot(adjNormal);
		float approxCel = sqrt(localGRAV * (myDepth + adjDepth) / 2.0f);

		float lambda[maxCONSERVED] = { 0.0f };
		lambda[0] = approxVelNormal - approxCel;
		lambda[1] = approxVelNormal;
		lambda[2] = approxVelNormal + approxCel;

		for (unsigned /*short*/ i = 0; i < maxCONSERVED; ++i)
			if (isValid(lambda[i]) && lambda[i] < 0)
				myDt = min(double(myDt), double(CFL) * double(myArea) * (2.0/3.0) / (double(adjLength) * double(abs(lambda[i]))));

		float lambdaLeft0 = myVel.dot(adjNormal) - sqrt(localGRAV*myDepth);
		float lambdaRight0 = adjVel.dot(adjNormal) - sqrt(localGRAV*adjDepth);

		float lambdaLeft2 = myVel.dot(adjNormal) + sqrt(localGRAV*myDepth);
		float lambdaRight2 = adjVel.dot(adjNormal) + sqrt(localGRAV*adjDepth);

		float lambdaAlt[maxCONSERVED] = { 0.0f };
		if (lambdaLeft0 < 0.0f && lambdaRight0 > 0.0f){
			lambdaAlt[0] = lambdaRight0 * (lambda[0] - lambdaLeft0) / (lambdaRight0 - lambdaLeft0);
			lambda[0] = lambdaLeft0 * (lambdaRight0 - lambda[0]) / (lambdaRight0 - lambdaLeft0);
		}

		if (lambdaLeft2 < 0.0f && lambdaRight2 > 0.0f){
			lambdaAlt[2] = lambdaLeft2 * (lambdaRight2 - lambda[2]) / (lambdaRight2 - lambdaLeft2);
			lambda[2] = lambdaRight2 * (lambda[2] - lambdaLeft2) / (lambdaRight2 - lambdaLeft2);
		}

		float eigVec[maxCONSERVED][maxCONSERVED] = { 0.0f };
		
		eigVec[0][0] = 1.0f;
		eigVec[0][1] = approxVel.x - approxCel*adjNormal.x;
		eigVec[0][2] = approxVel.y - approxCel*adjNormal.y;
		
		eigVec[1][0] = 0.0f;
		eigVec[1][1] = -approxCel*adjNormal.y;
		eigVec[1][2] = approxCel*adjNormal.x;

		eigVec[2][0] = 1.0f;
		eigVec[2][1] = approxVel.x + approxCel*adjNormal.x;
		eigVec[2][2] = approxVel.y + approxCel*adjNormal.y;

		vector2D approxVelDelta = (adjVel * adjDepth - myVel * myDepth) - approxVel * (adjDepth - myDepth);

		float approxVelDeltaNorm = approxVelDelta.dot(adjNormal);
		float approxVelDeltaTan = -approxVelDelta.x * adjNormal.y + approxVelDelta.y*adjNormal.x;

		float alpha[maxCONSERVED] = { 0.0f };
		alpha[0] = (adjDepth - myDepth) / 2.0f - approxVelDeltaNorm / (2.0f*approxCel);
		alpha[1] = approxVelDeltaTan / approxCel;
		alpha[2] = (adjDepth - myDepth) / 2.0f + approxVelDeltaNorm / (2.0f*approxCel);

		float deltaZ = 0.0f;
		if (adjZb >= myZb && (myZb + myDepth) < adjZb)
			deltaZ = myDepth;
		else if (adjZb < myZb && (adjZb + adjDepth) < myZb)
			deltaZ = -adjDepth;
		else
			deltaZ = adjZb - myZb;

		float altDepth = 0.0f;
		if (adjZb >= myZb)
			altDepth = myDepth;
		else
			altDepth = adjDepth;

		float bedSourceTerm = -localGRAV * (altDepth - abs(deltaZ) / 2.0f) * deltaZ;
		float bedSourceTermAlt = -localGRAV * ((myDepth + adjDepth) / 2.0f) * (adjZb - myZb);

		if (((adjZb + adjDepth) - (myZb + myDepth)) * (adjZb - myZb) >= 0.0f && approxVelNormal * (adjZb - myZb) > 0.0f){
			if (abs(bedSourceTermAlt) > abs(bedSourceTerm))
				bedSourceTerm = bedSourceTermAlt;
		}

		denoise(bedSourceTerm);

		float beta[maxCONSERVED] = { 0.0f };
		beta[0] = -1.0f / (2.0f*approxCel) * bedSourceTerm;
		beta[2] = -beta[0];

		float adjMedDepth = adjDepth - alpha[2] + beta[2] / lambda[2];
		denoise(adjMedDepth);

		float myMedDepth = myDepth + alpha[0] - beta[0] / lambda[0];
		denoise(myMedDepth);

		if (lambda[0] * lambda[2] < 0.0f){

			float minBeta0 = -(myDepth + alpha[0]) * abs(lambda[0]);
			float minBeta2 = -(adjDepth - alpha[2]) * lambda[2];

			if (myMedDepth < 0.0f){
				float dt1Star = abs(0.5f*myDepth / (myDepth - myMedDepth) * (myArea / adjLength) / lambda[0]);
				if (adjMedDepth > 0.0f && dt1Star < myDt){
					if (-minBeta0 >= minBeta2)
						beta[0] = minBeta0;
					beta[2] = -beta[0];
				}
			}else if (adjMedDepth < 0.0f){
				float dt3Star = abs(0.5f*adjDepth / (adjDepth - adjMedDepth) * (myArea / adjLength) / lambda[0]);
				if (myMedDepth > 0.0f && dt3Star < myDt){
					if (-minBeta2 >= minBeta0)
						beta[2] = minBeta2;
					beta[0] = -beta[2];
				}
			}
		}

		float inFluxes[maxCONSERVED] = { 0.0f };
		float massFluxOut = 0.0f;
		if ((adjMedDepth < 0.0f && !isValid(adjDepth)) || (myMedDepth < 0.0f && !isValid(myDepth)))
			for (unsigned /*short*/ i = 0; i < maxCONSERVED; ++i){
				if (lambda[i] < 0.0f)
					inFluxes[0] += -(lambda[i] * alpha[i] - beta[i]) * eigVec[i][0];
				if (lambdaAlt[i] < 0.0f)
					inFluxes[0] += -(lambdaAlt[i] * alpha[i]) * eigVec[i][0];
				if (lambda[i] > 0.0f)
					massFluxOut += (lambda[i] * alpha[i] - beta[i]) * eigVec[i][0];
				if (lambdaAlt[i] > 0.0f)
					massFluxOut += (lambdaAlt[i] * alpha[i]) * eigVec[i][0];
			}
		else
			for (unsigned /*short*/ i = 0; i < maxCONSERVED; ++i){
				if (lambda[i] < 0.0f)
					for (unsigned /*short*/ j = 0; j < maxCONSERVED; ++j)
						inFluxes[j] += -(lambda[i] * alpha[i] - beta[i]) * eigVec[i][j];
				else if (lambda[i] > 0.0f)
					massFluxOut += (lambda[i] * alpha[i] - beta[i]) * eigVec[i][0];
				if (lambdaAlt[i] < 0.0f)
					for (unsigned /*short*/ j = 0; j < maxCONSERVED; ++j)
						inFluxes[j] += -(lambdaAlt[i] * alpha[i]) * eigVec[i][j];
				else if (lambdaAlt[i] > 0.0f)
					massFluxOut += (lambdaAlt[i] * alpha[i]) * eigVec[i][0];
			}

		for (unsigned /*short*/ i = 0; i < maxCONSERVED; ++i){
			denoise(inFluxes[i]);
			flux->flux[i] += double(inFluxes[i]) * double(adjLength) / double(myArea);
		}

		float adjScalars[maxSCALARS] = { 0.0f };
		if (flux->neighbor[k].scalars != 0x0)
			for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
				adjScalars[s] = flux->neighbor[k].scalars->specie[s];
		else
			for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
				adjScalars[s] = myScalars[s];

		denoise(massFluxOut);
		float massDischarge = myVel.dot(adjNormal)*myDepth + massFluxOut;
		denoise(massDischarge);

		float scalarFluxes[maxSCALARS] = { 0.0f };
		if (isValid(massDischarge)){
			if (massDischarge > 0.0f){
				for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
					scalarFluxes[s] += -massDischarge * myScalars[s];
			}else{
				for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
					scalarFluxes[s] += -massDischarge * adjScalars[s];
			}
		}

		for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s){
			denoise(scalarFluxes[s]);
			flux->flux[maxCONSERVED + s] += double(scalarFluxes[s]) * double(adjLength) / double(myArea);
		}

		*(flux->dt) = min(*(flux->dt), myDt);
	}
}

CPU GPU void elementFlow::applyFluxes_ReducedGudonov_1stOrder(){

#	ifdef __CUDA_ARCH__
	double dt = gpuNumerics.dt;
	double minDepthFactor = double(gpuNumerics.minDepthFactor);
#	else
	double dt = cpuNumerics.dt;
	double minDepthFactor = double(cpuNumerics.minDepthFactor);
	using std::min;
	using std::max;
#	endif

	vector2D myVel = state->vel;
	float myDepth = state->h;
	float myDensity = state->rho;

	float myScalars[maxSCALARS];
	for (unsigned s = 0; s < maxSCALARS; ++s)
		myScalars[s] = scalars->specie[s];

	double conservedVar[maxCONSERVED + maxSCALARS] = { 0.0f };
	conservedVar[0] = double(myDepth);
	conservedVar[1] = double(myDepth) * double(myVel.x);
	conservedVar[2] = double(myDepth) * double(myVel.y);
	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		conservedVar[maxCONSERVED + s] = double(myDepth) * double(myScalars[s]);

	for (unsigned /*short*/ i = 0; i < maxCONSERVED + maxSCALARS; ++i){
		conservedVar[i] += (flux->flux[i])*dt;
		denoise(conservedVar[i]);
	}

	float wetDryCoef = sigmoid(conservedVar[0] / minDepthFactor);
	denoise(wetDryCoef);

	myDepth = float(conservedVar[0]);
	
	if (isValid(myDepth)){
		myVel.x = float(conservedVar[1] / conservedVar[0]);
		myVel.y = float(conservedVar[2] / conservedVar[0]);
		if (myDepth <= minDepthFactor){
			vector2D corrVel;
			corrVel.x = float((sqrt(2.0) * conservedVar[0] * conservedVar[1]) / (sqrt(pow(conservedVar[0], 4.0) + max(pow(conservedVar[0], 4.0), minDepthFactor*max(1.0, sqrt(2.0*double(area)))))));
			corrVel.y = float((sqrt(2.0) * conservedVar[0] * conservedVar[2]) / (sqrt(pow(conservedVar[0], 4.0) + max(pow(conservedVar[0], 4.0), minDepthFactor*max(1.0, sqrt(2.0*double(area)))))));
			if (isValid(wetDryCoef)){
				myVel.x = corrVel.x * (1.0f - wetDryCoef) + myVel.x * wetDryCoef;
				myVel.y = corrVel.y * (1.0f - wetDryCoef) + myVel.y * wetDryCoef;
			}else
				myVel.setNull();
		}
	}else{
		myDepth = 0.0f;
		myVel.setNull();
	}

	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s){
		if (!isValid(1.0f - wetDryCoef))
			myScalars[s] = float(conservedVar[maxCONSERVED + s] / conservedVar[0]);
		else if (isValid(wetDryCoef))
			myScalars[s] = myScalars[s] * (1.0f - wetDryCoef) + float(conservedVar[maxCONSERVED + s] / conservedVar[0]) * wetDryCoef;
		denoise(myScalars[s]);
	}

	denoise(myVel);

	state->vel = myVel;
	state->h = myDepth;
	state->rho = myDensity;
	for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
		scalars->specie[s] = myScalars[s];
}

CPU GPU void elementFlow::applyVelocityCorrections(){

	unsigned short numObstacles = 0;

	if (isValid(state->vel.norm())){
		for (unsigned /*short*/ k = 0; k < 3; ++k){

			bool adjIsWall = false;
			if (flux->neighbor[k].state == 0x0){
				adjIsWall = true;
				++numObstacles;
			}else{
				if (!flux->neighbor[k].state->isWet()){
					adjIsWall = true;
					++numObstacles;
				}else{
					float adjZb = flux->neighbor[k].state->z;
					float adjDepth = flux->neighbor[k].state->h;
					if ((state->z + state->h) < adjZb && !isValid(adjDepth)){
						adjIsWall = true;
						++numObstacles;
					}
				}
			}

			if (adjIsWall){
				if (numObstacles < 2){
					vector2D myVel = state->vel;
					vector2D adjNormal = flux->neighbor[k].normal;
					float velNormalNorm = myVel.dot(flux->neighbor[k].normal);
					myVel = myVel - adjNormal * velNormalNorm;
					state->vel = myVel;
				}else
					state->vel.setNull();
			}
		}
	}
}

CPU GPU void elementFlow::validate(){

#	ifdef __CUDA_ARCH__
	float waterDensity = gpuPhysics.waterDensity;
	float minDepthFactor = gpuNumerics.minDepthFactor;
	float maxConc = gpuBed.maxConc;
#	else
	float waterDensity = cpuPhysics.waterDensity;
	float minDepthFactor = cpuNumerics.minDepthFactor;
	float maxConc = cpuBed.maxConc;
	using std::max;
	using std::min;
#	endif

	denoise(state->h);
	denoise(state->vel);

	if (state->h < minDepthFactor){
		if (state->h <= 0.0f){
			state->vel.setNull();
			state->h = 0.0f;
		}
		state->rho = waterDensity;
		for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s)
			if (s < maxSCALARS)
				scalars->specie[s] = 0.0f;
	}else{
		for (unsigned /*short*/ s = 0; s < maxSCALARS; ++s){
			denoise(scalars->specie[s]);
			if (s < maxFRACTIONS)
				for (unsigned /*short*/ p = 0; p < maxFRACTIONS; ++p)
					scalars->specie[sedCp] = max(0.0f, min(scalars->specie[sedCp], maxConc));
			else if (s == relTemp)
				scalars->specie[relTemp] = max(-1.0f, min(scalars->specie[relTemp], 1.0f));
		}
	}
}
