//region | >> Copyright, doxygen, includes and definitions
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

// STL
#include <algorithm>

// HiSTAV
#include "../headers/numerics.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////
//endregion


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
    for (unsigned short f = 0; f < (maxCONSERVED + maxSCALARS); ++f)
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
    float lowDepth = gpuNumerics.lowDepth;
    float highDepth = gpuNumerics.highDepth;
    float CFL = gpuNumerics.CFL;
#	else
    float localGRAV = cpuPhysics.gravity;
    float lowDepth = cpuNumerics.lowDepth;
    float highDepth = cpuNumerics.highDepth;
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
    float myZb = state->z;

#   pragma unroll
    for (unsigned short k = 0; k < 3; ++k) {

        bool isWall = false;

        vector2D adjNormal = flux->neighbor[k].normal;
        float adjLength = flux->neighbor[k].length;

        vector2D adjVel;
        float adjDepth;
        float adjZb;

        if (flux->neighbor[k].state != 0x0) {
            adjVel = flux->neighbor[k].state->vel;
            adjDepth = flux->neighbor[k].state->h;
            adjZb = flux->neighbor[k].state->z;
        } else {
            isWall = true;
        }

        if (isWall) {
            vector2D adjTangent = vector2D(adjNormal.y, -adjNormal.x);
            float adjVelTangent = myVel.dot(adjTangent);
            if (adjVelTangent < 0.0) {
                adjTangent = -adjTangent;
                adjVelTangent = -adjVelTangent;
            }
            adjVel = adjTangent * (2.0f * adjVelTangent) - myVel;
            adjDepth = myDepth;
            adjZb = myZb;
            /*adjVel.setNull();
            adjDepth = 0.0f;
            adjZb = myZb;*/
        }

        vector2D approxVel = (myVel * sqrt(myDepth) + adjVel * sqrt(adjDepth)) / (sqrt(myDepth) + sqrt(adjDepth));

        float approxVelNormal = approxVel.dot(adjNormal);
        float approxCel = sqrt(localGRAV * (myDepth + adjDepth) / 2.0f);

        float lambda[maxCONSERVED] = {0.0f};
        lambda[0] = approxVelNormal - approxCel;
        lambda[1] = approxVelNormal;
        lambda[2] = approxVelNormal + approxCel;

        double maxDistance = double(CFL) * double(myArea) / double(adjLength);

        // (Compute dt)
#       pragma unroll
        for (unsigned short i = 0; i < maxCONSERVED; ++i)
            if (isValid(lambda[i]) && lambda[i] < 0)
                myDt = min(double(myDt), maxDistance / double(abs(lambda[i])));

        float lambdaLeft0 = myVel.dot(adjNormal) - sqrt(localGRAV * myDepth);
        float lambdaRight0 = adjVel.dot(adjNormal) - sqrt(localGRAV * adjDepth);

        float lambdaLeft2 = myVel.dot(adjNormal) + sqrt(localGRAV * myDepth);
        float lambdaRight2 = adjVel.dot(adjNormal) + sqrt(localGRAV * adjDepth);

        float lambdaAlt[maxCONSERVED] = {0.0f};
        if (lambdaLeft0 < 0.0f && lambdaRight0 > 0.0f) {
            lambdaAlt[0] = lambdaRight0 * (lambda[0] - lambdaLeft0) / (lambdaRight0 - lambdaLeft0);
            lambda[0] = lambdaLeft0 * (lambdaRight0 - lambda[0]) / (lambdaRight0 - lambdaLeft0);
        }

        if (lambdaLeft2 < 0.0f && lambdaRight2 > 0.0f) {
            lambdaAlt[2] = lambdaLeft2 * (lambdaRight2 - lambda[2]) / (lambdaRight2 - lambdaLeft2);
            lambda[2] = lambdaRight2 * (lambda[2] - lambdaLeft2) / (lambdaRight2 - lambdaLeft2);
        }

        float eigVec[maxCONSERVED][maxCONSERVED] = {0.0f};

        eigVec[0][0] = 1.0f;
        eigVec[0][1] = approxVel.x - approxCel * adjNormal.x;
        eigVec[0][2] = approxVel.y - approxCel * adjNormal.y;

        eigVec[1][0] = 0.0f;
        eigVec[1][1] = -approxCel * adjNormal.y;
        eigVec[1][2] = approxCel * adjNormal.x;

        eigVec[2][0] = 1.0f;
        eigVec[2][1] = approxVel.x + approxCel * adjNormal.x;
        eigVec[2][2] = approxVel.y + approxCel * adjNormal.y;

        vector2D approxVelDelta = (adjVel * adjDepth - myVel * myDepth) - approxVel * (adjDepth - myDepth);

        float approxVelDeltaNorm = approxVelDelta.dot(adjNormal);
        float approxVelDeltaTan = -approxVelDelta.x * adjNormal.y + approxVelDelta.y * adjNormal.x;

        float alpha[maxCONSERVED] = {0.0f};
        alpha[0] = (adjDepth - myDepth) / 2.0f - approxVelDeltaNorm / (2.0f * approxCel);
        alpha[1] = approxVelDeltaTan / approxCel;
        alpha[2] = (adjDepth - myDepth) / 2.0f + approxVelDeltaNorm / (2.0f * approxCel);

        float thrustStep = 0.0f;
        if (adjZb >= myZb && (myZb + myDepth) < adjZb)
            thrustStep = myDepth;
        else if (adjZb < myZb && (adjZb + adjDepth) < myZb)
            thrustStep = -adjDepth;
        else
            thrustStep = adjZb - myZb;

        float thrustDepth = 0.0f;
        if (adjZb >= myZb)
            thrustDepth = myDepth;
        else
            thrustDepth = adjDepth;

        float thrustTerm = -localGRAV * (thrustDepth - abs(thrustStep) / 2.0f) * thrustStep;
        float thrustTermAlt = -localGRAV * ((myDepth + adjDepth) / 2.0f) * (adjZb - myZb);

        if (((adjZb + adjDepth) - (myZb + myDepth)) * (adjZb - myZb) >= 0.0f &&
            approxVelNormal * (adjZb - myZb) > 0.0f) {
            if (abs(thrustTermAlt) > abs(thrustTerm))
                thrustTerm = thrustTermAlt;
        }

        float beta[maxCONSERVED] = {0.0f};
        beta[0] = -1.0f / (2.0f * approxCel) * thrustTerm;
        beta[2] = -beta[0];

        if (lambda[0] * lambda[2] < 0.0f) {

            float myMedDepth = myDepth + alpha[0] - beta[0] / lambda[0];
            float adjMedDepth = adjDepth - alpha[2] + beta[2] / lambda[2];

            float minBeta0 = -(myDepth + alpha[0]) * abs(lambda[0]);
            float minBeta2 = -(adjDepth - alpha[2]) * lambda[2];

            if (myMedDepth < 0.0f)
                if (adjMedDepth > 0.0f)
                    if (-minBeta0 >= minBeta2) {
                        beta[0] = minBeta0;
                        beta[2] = -beta[0];
                    }

            if (adjMedDepth < 0.0f)
                if (myMedDepth > 0.0f)
                    if (-minBeta2 >= minBeta0) {
                        beta[2] = minBeta2;
                        beta[0] = -beta[2];
                    }
        }

        float inFluxes[maxCONSERVED] = {0.0f};
#       pragma unroll
        for (unsigned short i = 0; i < maxCONSERVED; ++i) {
            if (lambda[i] < 0.0f)
#               pragma unroll
                for (unsigned short j = 0; j < maxCONSERVED; ++j)
                    inFluxes[j] += -(lambda[i] * alpha[i] - beta[i]) * eigVec[i][j];
            if (lambdaAlt[i] < 0.0f)
#               pragma unroll
                for (unsigned short j = 0; j < maxCONSERVED; ++j)
                    inFluxes[j] += -(lambdaAlt[i] * alpha[i]) * eigVec[i][j];
        }

        float wetDryCoef = getWetCoef(max(myDepth, adjDepth), lowDepth, highDepth);

        if (!isWall)
            flux->flux[0] += double(inFluxes[0] * wetDryCoef) * double(adjLength) / double(myArea);

        float myWSE = myDepth + myZb;
        float adjWSE = adjDepth + adjZb;

        if (isValid(wetDryCoef))
            if (isValid(myVel) || isValid(adjVel) || isWall || isValid(myWSE - adjWSE))
#              pragma unroll
                for (unsigned short i = 1; i < maxCONSERVED; ++i) {
                    flux->flux[i] += double(inFluxes[i] * wetDryCoef) * double(adjLength) / double(myArea);
                }

        *(flux->dt) = min(*(flux->dt), myDt);
    }
}

/* 3.
 CPU GPU void elementFlow::computeFluxes_Alternative(){

#	ifdef __CUDA_ARCH__
    float localGRAV = gpuPhysics.gravity;
    float lowDepth = gpuNumerics.lowDepth;
    float highDepth = gpuNumerics.highDepth;
    float CFL = gpuNumerics.CFL;
#	else
    float localGRAV = cpuPhysics.gravity;
    float lowDepth = cpuNumerics.lowDepth;
    float highDepth = cpuNumerics.highDepth;
    float CFL = cpuNumerics.CFL;
    using std::abs;
    using std::min;
    using std::max;
    using std::sqrt;
#	endif

 ...

*/

CPU GPU void elementFlow::applyFluxes_ReducedGudonov_1stOrder(){

#	ifdef __CUDA_ARCH__
    double dt = gpuNumerics.dt;
    float lowDepth = gpuNumerics.lowDepth;
    float highDepth = gpuNumerics.highDepth;
#	else
    double dt = cpuNumerics.dt;
    float lowDepth = cpuNumerics.lowDepth;
    float highDepth = cpuNumerics.highDepth;
    using std::min;
    using std::max;
#	endif

    vector2D myVel = state->vel;
    float myDepth = state->h;

    double conserved[maxCONSERVED] = {0.0f};
    conserved[0] = double(myDepth);
    conserved[1] = double(myDepth) * double(myVel.x);
    conserved[2] = double(myDepth) * double(myVel.y);

#   pragma unroll
    for (unsigned short i = 0; i < maxCONSERVED; ++i)
        conserved[i] += (flux->flux[i]) * dt;

    myDepth = float(conserved[0]);

    if (isValid(myDepth)){
        myVel.x = float(conserved[1] / conserved[0]);
        myVel.y = float(conserved[2] / conserved[0]);
        if (myDepth < highDepth)
            myVel = myVel * getWetCoef(myDepth, lowDepth, highDepth);
    }else{
        myVel.setNull();
        myDepth = 0.0f;
    }

    state->vel = myVel;
    state->h = myDepth;
}

CPU GPU void elementFlow::applyVelocityCorrections(){

    if (isValid(state->vel.norm())) {

        unsigned short obstacleIdx = 99;
        unsigned short numNoSlip = 0;
        unsigned short numWalls = 0;

#       pragma unroll
        for (unsigned short k = 0; k < 3; ++k) {
            if (flux->neighbor[k].state == 0x0) {
                ++numWalls;
                obstacleIdx = k;
            } else {
                if (!flux->neighbor[k].state->isWet()) {
                    ++numWalls;
                    obstacleIdx = k;
                } else {
                    vector2D adjNormal = flux->neighbor[k].normal;
                    vector2D adjVel = flux->neighbor[k].state->vel;
                    float velNormal = adjVel.dot(adjNormal);
                    float adjZb = flux->neighbor[k].state->z;
                    float myWSE = state->z + state->h;
                    if ((myWSE < adjZb) && velNormal > 0.0f) {
                        /*++numWalls;
                        obstacleIdx = k;*/
                    }
                }
            }
        }

        if (numWalls == 1) {
            vector2D myVel = state->vel;
            vector2D adjNormal = flux->neighbor[obstacleIdx].normal;
            float velNormal = myVel.dot(flux->neighbor[obstacleIdx].normal);
            myVel = myVel - adjNormal * velNormal;
            state->vel = myVel;
        }

        if (numWalls > 1 || numNoSlip > 0)
            state->vel.setNull();
    }
}

CPU GPU void elementFlow::validate(){

#	ifdef __CUDA_ARCH__
    float waterDensity = gpuPhysics.waterDensity;
    float lowDepth = gpuNumerics.lowDepth;
#	else
    float waterDensity = cpuPhysics.waterDensity;
    float lowDepth = cpuNumerics.lowDepth;
    using std::max;
    using std::min;
#	endif

    denoise(state->vel);
    denoise(state->h);

    if (state->h < lowDepth){
        state->vel.setNull();
        state->h = max(state->h, 0.0f);
        state->rho = waterDensity;
    }
}