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

// STL
#include <cmath>
#include <iostream>
#include <fstream>
#include <algorithm>

// STAV
#include "../headers/boundaries.hpp"
#include "../headers/mesh.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU elementGhost::elementGhost(){

	link = 0x0;

	hydroGauge[0] = 0x0;
	hydroGauge[1] = 0x0;
	sediGauge[0] = 0x0;
	sediGauge[1] = 0x0;

	inletRefValue = 0x0;
	inletRefFactor = 0x0;

	gaugeWeight[0] = 0.0f;
	gaugeWeight[1] = 0.0f;
	edgeLength = 0.0f;

	conditionType = 0;
	isAtEquilibrium = false;
}

physicalBoundary::physicalBoundary(){

	elemGhost = 0x0;

	inletRefValue = 0x0;
	inletRefFactor = 0x0;

	for (unsigned l = 0; l < maxDISCHCURLENGTH; ++l){
		inletCurveLevel[l] = 0.0f;
		inletCurveDischarge[l] = 0.0f;
	}

	referenceSlope = 0.0f;
	bufferRadious = 0.0f;

	numElemGhosts = 0;

	numGauges = 0;
	conditionType = 0;
	id = 0;

	isAtEquilibrium = false;
	isUniformInlet = false;
}

domainBoundaries::domainBoundaries(){

	hydroGauge = 0x0;
	sediGauge = 0x0;

	physical = 0x0;
	elemGhost = 0x0;

	numElemGhosts = 0;

	numGauges = 0;
	numBoundaries = 0;

	hasUniformInlets = false;
}

domainBoundaries cpuBoundaries;

std::vector<float> cpuInletRefValue;
std::vector<float> cpuInletRefFactor;

#ifdef __STAV_MPI__
std::vector<float> cpuInletRefValueBuffer;
std::vector<float> cpuInletRefFactorBuffer;
#endif

#ifdef __STAV_CUDA__
physicalBoundary* gpuPhysicalBoundaries;
elementGhost* gpuElemGhost;
timeseries* gpuHydroGauge;
timeseries* gpuSediGauge;
float* gpuInletRefValue;
float* gpuInletRefFactor;
#endif


void physicalBoundary::setUniformInlet(){

	if (conditionType == bndDISCHCUR){

		std::vector<float> level(5000, 0.0f);

		for (unsigned l = 1; l < 1000; ++l)
			level[l] = 0.0001f*float(l);

		for (unsigned l = 1000; l < 2000; ++l)
			level[l] = 0.1f + 0.001f*float(l - 1000);

		for (unsigned l = 2000; l < 3000; ++l)
			level[l] = 1.0f + 0.01f*float(l - 2000);

		for (unsigned l = 3000; l < 4000; ++l)
			level[l] = 10.0f + 0.1f*float(l - 3000);

		for (unsigned l = 4000; l < 5000; ++l)
			level[l] = 100.0f + 1.0f*float(l - 4000);

		float minZb = 9999999999999.0f;
		for (unsigned i = 0; i < numElemGhosts; ++i)
			if (elemGhost[i].state.z <= minZb)
				minZb = elemGhost[i].state.z;

		for (unsigned l = 0; l < maxDISCHCURLENGTH; ++l){

			float totalDischarge = 0.0f;
			float absLevel = minZb + level[l];

			for (unsigned i = 0; i < numElemGhosts; ++i){
				float depth = std::max(0.0f, absLevel - elemGhost[i].state.z);
				if (isValid(depth))
					totalDischarge += elemGhost[i].edgeLength * std::pow(depth, 5.0f / 3.0f) * std::sqrt(referenceSlope) * elemGhost[i].link->bed->fricPar;
			}

			totalDischarge = totalDischarge >= bndTOLERANCE ? totalDischarge : 0.0f;
			inletCurveLevel[l] = absLevel;
			inletCurveDischarge[l] = totalDischarge;
		}
	}
}

CPU GPU void physicalBoundary::setRefValue(){

#	ifndef __CUDA_ARCH__
	using std::max;
	using std::pow;
#	endif

	float totalFactor = 0.0f;

	if (conditionType == bndDISCHRIV || conditionType == bndDISCHRHG){

		for (unsigned i = 0; i < numElemGhosts; ++i){
			float depth = elemGhost[i].link->flow->state->h;
			float length = elemGhost[i].edgeLength;
			if (isValid(depth))
				totalFactor += pow(depth*length, bndWGHTPOWER);
		}

		denoise(totalFactor);

		*inletRefValue = 0.0f;
		*inletRefFactor = totalFactor;

	}else if (conditionType == bndDISCHCUR){

		float targetDischarge = elemGhost[0].hydroGauge[0]->getData();
		denoise(targetDischarge);

		unsigned idxIni = 0;
		if (*inletRefValue >= inletCurveLevel[1000] && *inletRefValue < inletCurveLevel[1999])
			idxIni = 900;
		else if (*inletRefValue >= inletCurveLevel[2000] && *inletRefValue < inletCurveLevel[2999])
			idxIni = 1900;
		else if (*inletRefValue >= inletCurveLevel[3000] && *inletRefValue < inletCurveLevel[3999])
			idxIni = 2900;
		else if (*inletRefValue >= inletCurveLevel[4000] && *inletRefValue < inletCurveLevel[4999])
			idxIni = 3900;

		unsigned idx;
		for (idx = idxIni; idx < maxDISCHCURLENGTH; ++idx)
			if (inletCurveDischarge[idx] > targetDischarge)
				break;

		if (idx == maxDISCHCURLENGTH)
			for (idx = 0; idx < maxDISCHCURLENGTH; ++idx)
				if (inletCurveDischarge[idx] > targetDischarge)
					break;

		float previousDischarge = inletCurveDischarge[idx-1];
		float previousLevel = inletCurveLevel[idx-1];
		float nextDischarge = inletCurveDischarge[idx];
		float nextLevel = inletCurveLevel[idx];

		float targetLevel = previousLevel + (nextLevel - previousLevel) / (nextDischarge - previousDischarge) * (targetDischarge - previousDischarge);

		for (unsigned i = 0; i < numElemGhosts; ++i){
			float depth = max(0.0f, targetLevel - elemGhost[i].link->flow->state->z);
			float length = elemGhost[i].edgeLength;
			if (isValid(depth))
				totalFactor += pow(depth*length, bndWGHTPOWER);
		}

		denoise(totalFactor);

		*inletRefValue = targetLevel;
		*inletRefFactor = totalFactor;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU void getRealRoots3rdOrder(double* realRoots3rdOrder, double aCoef3rd, double bCoef3rd, double cCoef3rd, double dCoef3rd){

#	ifndef __CUDA_ARCH__
	using std::abs;
	using std::sin;
#	endif

	double fCoef3rd = ((3.0*cCoef3rd / aCoef3rd) - ((bCoef3rd*bCoef3rd) / (aCoef3rd*aCoef3rd))) / 3.0;
	double gCoef3rd = ((2.0*(bCoef3rd*bCoef3rd*bCoef3rd) / (aCoef3rd*aCoef3rd*aCoef3rd)) - (9.0*(bCoef3rd*cCoef3rd) / (aCoef3rd*aCoef3rd)) + (27.0*dCoef3rd / aCoef3rd)) / 27.0;
	double hCoef3rd = ((gCoef3rd*gCoef3rd) / 4.0) + ((fCoef3rd*fCoef3rd*fCoef3rd) / 27.0);

	if (fCoef3rd == 0.0 && gCoef3rd == 0.0 && hCoef3rd == 0.0){

		realRoots3rdOrder[0] = -cbrt(dCoef3rd / aCoef3rd);
		realRoots3rdOrder[1] = realRoots3rdOrder[0];
		realRoots3rdOrder[2] = realRoots3rdOrder[1];

	}else if (hCoef3rd > 0.0f){

		double rCoef3rd = -(gCoef3rd / 2.0) + sqrt(hCoef3rd);
		double sCoef3rd = cbrt(rCoef3rd);
		double tCoef3rd = -(gCoef3rd / 2.0) - sqrt(hCoef3rd);
		double uCoef3rd = cbrt(tCoef3rd);

		realRoots3rdOrder[0] = (sCoef3rd + uCoef3rd) - (bCoef3rd / (3.0*aCoef3rd));
		realRoots3rdOrder[1] = -1.0f;
		realRoots3rdOrder[2] = -1.0f;

	}else{

		double iCoef3rd = sqrt(gCoef3rd*gCoef3rd / 4.0 - hCoef3rd);
		double jCoef3rd = cbrt(iCoef3rd);
		double kCoef3rd = acos(-gCoef3rd / (2.0*iCoef3rd));
		double lCoef3rd = -jCoef3rd;
		double mCoef3rd = cos((kCoef3rd / 3.0));
		double nCoef3rd = sqrt(3.0)*sin(kCoef3rd / 3.0);
		double pCoef3rd = -(bCoef3rd / (3.0*aCoef3rd));

		realRoots3rdOrder[0] = 2.0*jCoef3rd*cos(kCoef3rd / 3.0) - (bCoef3rd / (3.0*aCoef3rd));
		realRoots3rdOrder[1] = lCoef3rd*(mCoef3rd + nCoef3rd) + pCoef3rd;
		realRoots3rdOrder[2] = lCoef3rd*(mCoef3rd - nCoef3rd) + pCoef3rd;
	}
}

CPU GPU void getRoots3rdOrder(localCOMPLEX* roots3rdOrder, double aCoef3rd, double bCoef3rd, double cCoef3rd, double dCoef3rd){

#	ifndef __CUDA_ARCH__
	using std::abs;
	using std::sin;
#	endif

	double fCoef3rd = ((3.0*cCoef3rd / aCoef3rd) - (bCoef3rd*bCoef3rd / aCoef3rd*aCoef3rd)) / 3.0;
	double gCoef3rd = ((2.0*bCoef3rd*bCoef3rd*bCoef3rd / aCoef3rd*aCoef3rd*aCoef3rd) - (9.0*bCoef3rd*cCoef3rd / aCoef3rd*aCoef3rd) + (27.0*dCoef3rd / aCoef3rd)) / 27.0;
	double hCoef3rd = (gCoef3rd*gCoef3rd / 4.0) + (fCoef3rd*fCoef3rd*fCoef3rd / 27.0);

	if (fCoef3rd == 0.0 && gCoef3rd == 0.0 && hCoef3rd == 0.0){

		fCoef3rd = 0.0f;
		gCoef3rd = 0.0f;
		hCoef3rd = 0.0f;

		roots3rdOrder[0] = -cbrt(dCoef3rd / aCoef3rd);
		roots3rdOrder[1] = roots3rdOrder[0];
		roots3rdOrder[2] = roots3rdOrder[1];

	}else if (hCoef3rd > 0.0f){

		double rCoef3rd = -(gCoef3rd / 2.0) + sqrt(hCoef3rd);
		double sCoef3rd = cbrt(rCoef3rd);
		double tCoef3rd = -(gCoef3rd / 2.0) - sqrt(hCoef3rd);
		double uCoef3rd = cbrt(tCoef3rd);

		roots3rdOrder[0] = localCOMPLEX((sCoef3rd + uCoef3rd) - (bCoef3rd / (3.0*aCoef3rd)), 0.0);
		roots3rdOrder[1] = localCOMPLEX(-(sCoef3rd + uCoef3rd) / 2.0 - (bCoef3rd / (3.0*aCoef3rd)), (sCoef3rd - uCoef3rd)*sqrt(3.0) / 2.0);
		roots3rdOrder[2] = localCOMPLEX(-(sCoef3rd + uCoef3rd) / 2.0 - (bCoef3rd / (3.0*aCoef3rd)), -(sCoef3rd - uCoef3rd)*sqrt(3.0) / 2.0);

	}else{

		double iCoef3rd = sqrt(gCoef3rd*gCoef3rd / 4.0 - hCoef3rd);
		double jCoef3rd = cbrt(iCoef3rd);
		double kCoef3rd = acos(-gCoef3rd / (2.0*iCoef3rd));
		double lCoef3rd = -jCoef3rd;
		double mCoef3rd = cos((kCoef3rd / 3.0));
		double nCoef3rd = sqrt(3.0)*sin(kCoef3rd / 3.0);
		double pCoef3rd = -(bCoef3rd / (3.0*aCoef3rd));

		roots3rdOrder[0] = localCOMPLEX(2.0*jCoef3rd*cos(kCoef3rd / 3.0) - (bCoef3rd / (3.0*aCoef3rd)), 0.0);
		roots3rdOrder[1] = localCOMPLEX(lCoef3rd*(mCoef3rd + nCoef3rd) + pCoef3rd, 0.0);
		roots3rdOrder[2] = localCOMPLEX(lCoef3rd*(mCoef3rd - nCoef3rd) + pCoef3rd, 0.0);
	}
}

CPU GPU void getRoots4thOrder(localCOMPLEX* roots4thOrder, double aCoef4th, double bCoef4th, double cCoef4th, double dCoef4th, double eCoef4th){

#	ifndef __CUDA_ARCH__
	using std::abs;
#	endif

	bCoef4th /= aCoef4th;
	cCoef4th /= aCoef4th;
	dCoef4th /= aCoef4th;
	eCoef4th /= aCoef4th;
	aCoef4th = 1.0;

	double fCoef4th = cCoef4th - (3.0*bCoef4th*bCoef4th) / 8.0;
	double gCoef4th = dCoef4th + (bCoef4th*bCoef4th*bCoef4th / 8.0) - (bCoef4th*cCoef4th / 2.0);
	double hCoef4th = eCoef4th - (3.0*bCoef4th*bCoef4th*bCoef4th*bCoef4th / 256.0) + (bCoef4th*bCoef4th*cCoef4th / 16.0) - (bCoef4th*dCoef4th / 4.0);

	double aCoef3rd = 1.0;
	double bCoef3rd = fCoef4th / 2.0;
	double cCoef3rd = (fCoef4th*fCoef4th - 4.0*hCoef4th) / 16.0;
	double dCoef3rd = -gCoef4th*gCoef4th / 64.0;

	localCOMPLEX* subRoots3rdOder = 0x0;//[3]; wrong, carefull
	getRoots3rdOrder(subRoots3rdOder, aCoef3rd, bCoef3rd, cCoef3rd, dCoef3rd);

	unsigned imagCounter = 0;

	for (unsigned i = 0; i < 3; ++i)
		if (abs(subRoots3rdOder[i].imag()) > bndTOLERANCE)
			imagCounter++;

	localCOMPLEX pCoef4th, qCoef4th;

	if (imagCounter == 2)
		if (subRoots3rdOder[0].imag() > bndTOLERANCE){
			pCoef4th = sqrt(subRoots3rdOder[0]);
			if (subRoots3rdOder[1].imag() > bndTOLERANCE)
				qCoef4th = sqrt(subRoots3rdOder[1]);
			else
				qCoef4th = sqrt(subRoots3rdOder[2]);
		}else{
			pCoef4th = sqrt(subRoots3rdOder[1]);
			qCoef4th = sqrt(subRoots3rdOder[2]);
		}
	else
		if (abs(subRoots3rdOder[0]) > bndTOLERANCE){
			pCoef4th = sqrt(subRoots3rdOder[0]);
			if (abs(subRoots3rdOder[1]) > bndTOLERANCE)
				qCoef4th = sqrt(subRoots3rdOder[1]);
			else
				qCoef4th = sqrt(subRoots3rdOder[2]);
		}else{
			pCoef4th = sqrt(subRoots3rdOder[1]);
			qCoef4th = sqrt(subRoots3rdOder[2]);
		}

		localCOMPLEX rCoef4th = -gCoef4th / (8.0*pCoef4th*qCoef4th);
		localCOMPLEX sCoef4th = bCoef4th / (4.0*aCoef4th);

		roots4thOrder[0] = pCoef4th + qCoef4th + rCoef4th - sCoef4th;
		roots4thOrder[1] = pCoef4th - qCoef4th - rCoef4th - sCoef4th;
		roots4thOrder[2] = -pCoef4th + qCoef4th - rCoef4th - sCoef4th;
		roots4thOrder[3] = -pCoef4th - qCoef4th + rCoef4th - sCoef4th;
}

void domainBoundaries::readControlFiles(std::string& boundaryDimFile, std::string& boundaryControlFile, std::string& boundaryGaugesFolder){

	std::cout << std::endl;
	std::cout << "  -> Importing boundaries ..." << std::endl;

	std::string inputText;
	unsigned inputInteger;

	std::ifstream meshInformationFile;
	meshInformationFile.open(boundaryDimFile);

	if (!meshInformationFile.is_open() || !meshInformationFile.good()){
		std::cout << std::endl << "   -> *Error* [B-1]: Could not open file " + boundaryDimFile + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + boundaryDimFile << ": ";

	meshInformationFile >> inputInteger;
	std::cout << inputInteger << " boundaries" << std::endl;

	if (inputInteger == 0)
		return;

	numBoundaries = (unsigned short) inputInteger;
	physical = new physicalBoundary[numBoundaries];

	cpuInletRefValue.resize(numBoundaries, 0.0f);
	cpuInletRefFactor.resize(numBoundaries, 0.0f);

#	ifdef __STAV_MPI__
	cpuInletRefValueBuffer.resize(numBoundaries*myProc.worldSize, 0.0f);
	cpuInletRefFactorBuffer.resize(numBoundaries*myProc.worldSize, 0.0f);
#	endif

	std::ifstream boundariesControlFile;
	boundariesControlFile.open(boundaryControlFile);

	if (!boundariesControlFile.is_open() || !boundariesControlFile.good()){
		std::cerr << std::endl << "   -> *Error* [B-1]: Could not open file " + boundaryControlFile + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + boundaryControlFile << std::endl;

	for (unsigned b = 0; b < numBoundaries; ++b){
		boundariesControlFile >> physical[b].id;
		boundariesControlFile >> physical[b].conditionType;
		boundariesControlFile >> physical[b].referenceSlope;
		boundariesControlFile >> physical[b].bufferRadious;
		getline(boundariesControlFile, inputText);
	}

	boundariesControlFile.close();

	getline(meshInformationFile, inputText);
	getline(meshInformationFile, inputText);

	std::vector<timeseries> tempHydroGauge;
	std::vector<timeseries> tempSediGauge;

	unsigned counterTotalGauges = 0;
	unsigned counterTotalElemBnd = 0;

	for (unsigned b = 0; b < numBoundaries; ++b){

		meshInformationFile >> physical[b].id;
		meshInformationFile >> physical[b].numGauges;
		meshInformationFile >> physical[b].numElemGhosts;
		getline(meshInformationFile, inputText);

		physical[b].inletRefValue = &cpuInletRefValue[b];
		physical[b].inletRefFactor = &cpuInletRefFactor[b];

		std::vector< std::string > gaugeFiles;
		bool allTheSameFile = true;

		for (unsigned k = 0; k < physical[b].numGauges; ++k){
			tempHydroGauge.push_back(timeseries());
			meshInformationFile >> inputInteger;
			meshInformationFile >> inputText;
			inputText = boundaryGaugesFolder + inputText;
			tempHydroGauge[counterTotalGauges + k].readData(inputText);
			gaugeFiles.push_back(inputText);
		}

		for (unsigned k = 0; k < physical[b].numGauges; ++k)
			if (gaugeFiles[k] == gaugeFiles[0])
				allTheSameFile = allTheSameFile && true;
			else
				allTheSameFile = allTheSameFile && false;

		if (allTheSameFile && (physical[b].conditionType == bndDISCHCUR || physical[b].conditionType == bndDISCHRHG || physical[b].conditionType == bndDISCHRIV)){
			physical[b].isUniformInlet = true;
			hasUniformInlets = true;
		}

		counterTotalGauges += physical[b].numGauges;
		counterTotalElemBnd += physical[b].numElemGhosts;
		getline(meshInformationFile, inputText);
	}

	numGauges = (unsigned short) counterTotalGauges;
	hydroGauge = new timeseries[numGauges];
	sediGauge = new timeseries[numGauges];

	for (unsigned k = 0; k < tempHydroGauge.size(); ++k)
		hydroGauge[k] = tempHydroGauge[k];

	for (unsigned k = 0; k < tempSediGauge.size(); ++k)
		sediGauge[k] = tempSediGauge[k];

	tempHydroGauge.clear();
	tempSediGauge.clear();

	meshInformationFile.close();

	numElemGhosts = counterTotalElemBnd;
	elemGhost = new elementGhost[numElemGhosts];
}

void domainBoundaries::readMeshFiles(std::string& boundaryIdxFile){

	std::string inputText;

	std::cout << std::endl;
	std::cout << "  -> Setting boundary buffers  " << std::endl;

	unsigned counterTotalGauges = 0;
	unsigned counterTotalGhosts = 0;

	for (unsigned b = 0; b < numBoundaries; ++b){

		std::ostringstream bnd_filename_ss;
		bnd_filename_ss << boundaryIdxFile << "-" << physical[b].id << ".bnd";
		std::string boundaryFileName = bnd_filename_ss.str();
		std::ifstream boundaryConnectivityFile;
		boundaryConnectivityFile.open(boundaryFileName);

		if (!boundaryConnectivityFile.is_open() || !boundaryConnectivityFile.good()){
			std::cerr << std::endl << "   -> *Error* [B-3]: Could not open file " << boundaryFileName << " ... aborting!" << std::endl;
			exitOnKeypress(1);
		}

		for (unsigned j = 0; j < b; ++j){
			counterTotalGauges += physical[j].numGauges;
			counterTotalGhosts += physical[j].numElemGhosts;
		}

		physical[b].elemGhost = &elemGhost[counterTotalGhosts];

		for (unsigned j = 0; j < physical[b].numElemGhosts; ++j){

			unsigned boundaryElemIdx;
			unsigned boundaryElemSide;

			boundaryConnectivityFile >> boundaryElemIdx;
			boundaryConnectivityFile >> boundaryElemSide;

			physical[b].elemGhost[j].link = &cpuMesh.elem[boundaryElemIdx];
			physical[b].elemGhost[j].state.z = cpuMesh.elem[boundaryElemIdx].flow->state->z;

			physical[b].elemGhost[j].normal = -cpuMesh.elem[boundaryElemIdx].flow->flux->neighbor[boundaryElemSide].normal;
			physical[b].elemGhost[j].edgeLength = cpuMesh.elem[boundaryElemIdx].flow->flux->neighbor[boundaryElemSide].length;

			physical[b].elemGhost[j].inletRefValue = physical[b].inletRefValue;
			physical[b].elemGhost[j].inletRefFactor = physical[b].inletRefFactor;

			cpuMesh.elem[boundaryElemIdx].flow->flux->neighbor[boundaryElemSide].state = &physical[b].elemGhost[j].state;
			cpuMesh.elem[boundaryElemIdx].flow->flux->neighbor[boundaryElemSide].scalars = &physical[b].elemGhost[j].scalars;

			unsigned inputInteger;
			unsigned idxTimeSeries;

			boundaryConnectivityFile >> inputInteger;
			idxTimeSeries = counterTotalGauges + inputInteger;
			physical[b].elemGhost[j].hydroGauge[0] = &hydroGauge[idxTimeSeries];

			boundaryConnectivityFile >> inputInteger;
			idxTimeSeries = counterTotalGauges + inputInteger;
			physical[b].elemGhost[j].hydroGauge[1] = &hydroGauge[idxTimeSeries];

			boundaryConnectivityFile >> physical[b].elemGhost[j].gaugeWeight[0];
			boundaryConnectivityFile >> physical[b].elemGhost[j].gaugeWeight[1];

			physical[b].elemGhost[j].conditionType = physical[b].conditionType;
			physical[b].elemGhost[j].isAtEquilibrium = physical[b].isAtEquilibrium;

			getline(boundaryConnectivityFile, inputText);
		}

		boundaryConnectivityFile.close();
	}

	unsigned counter = 0;
	for (unsigned b = 0; b < numBoundaries; ++b){
		if (physical[b].isUniformInlet)
			physical[b].setUniformInlet();
		if (physical[b].bufferRadious > 0.001f)
			for (unsigned i = 0; i < physical[b].numElemGhosts; ++i){
				for (unsigned j = 0; j < cpuMesh.numElems; j++){
					point myCenter = physical[b].elemGhost[i].link->meta->center;
					if (myCenter.distXY(cpuMesh.elem[j].meta->center) <= physical[b].bufferRadious)
						cpuMesh.elem[j].bed->flow = 0;
				}
				showProgress(int(counter++), int(cpuBoundaries.numElemGhosts), "    -> Setting", "buffers");
			}
	}
}
