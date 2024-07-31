/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira
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
#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <string>

// STAV
#include "../headers/sediment.hpp"
#include "../headers/mesh.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU elementBedComp::elementBedComp(){

	for (unsigned short p = 0; p < maxFRACTIONS; ++p){
		bedPercentage[p] = (1.0f / ((float) maxFRACTIONS));
		subPercentage[p] = (1.0f / ((float) maxFRACTIONS));
	}
}

CPU GPU elementBed::elementBed(){

	comp = nullptr;
	flow = nullptr;
	fricPar = 60.0f;
	bedrock = -1.0f;
}

CPU GPU bedLandslide::bedLandslide() {

	elemLink = nullptr;
	landslideTimeTrigger = 0.0f;
	landslideDepth = 0.0f;
	numElems = 0;
	isTriggered = false;
}

CPU GPU void elementBed::applyBedFriction(){

#	ifdef __CUDA_ARCH__
	unsigned short frictionOption = gpuBed.frictionOption;
	float gravity = gpuPhysics.gravity;
	float dt = float(gpuNumerics.dt);
	float lowDepth = gpuNumerics.lowDepth;
	float highDepth = gpuNumerics.highDepth;
#	else
	unsigned short frictionOption = cpuBed.frictionOption;
	float gravity = cpuPhysics.gravity;
	float dt = float(cpuNumerics.dt);
	float lowDepth = cpuNumerics.lowDepth;
    float highDepth = cpuNumerics.highDepth;
	using std::abs;
	using std::min;
	using std::max;
	using std::cbrt;
	using std::pow;
	using std::copysign;
#	endif

	vector2D myVel = flow->state->vel;
	float myDensity = flow->state->rho;
	float myDepth = flow->state->h;

	float myFrPar = fricPar;

    /*float balanceFactor = 0.0;
    if(gpuCurrentTime > 15000.0f) {
        balanceFactor = min(1.0f, (gpuCurrentTime-15000.0f)/15000.0f);
        myFrPar = 1.0f * balanceFactor + fricPar * (1.0f - balanceFactor);
    }*/

	float myFrCoef = 0.0f;
	vector2D myTau;

	if (isValid(myVel.norm()) && myDepth > lowDepth){

		if (frictionOption == 1)
			myFrCoef = gravity / (cbrt(myDepth) * myFrPar * myFrPar);
		else if (frictionOption == 2){
			float mySedC[maxFRACTIONS] = { 0.0f };
			float myVisc = flow->getMu();
			for (unsigned short p = 0; p < maxFRACTIONS; ++p)
				mySedC[p] = flow->scalars->specie[sedCp];
			myFrCoef = min(myFrPar, (myVel.norm() * getAvgDiam(mySedC)) / (myDepth * getAvgFallVel(mySedC, myDensity, myVisc)));
		}

		/*if(true){
            float bedSlope = 0.0007;    // Tua      [-]
            float mParameter = 0.07;    // Default  [-]
            float nParameter = 0.2;     // Default  [-]
            float lambdaOverH = 7.3;    // Default  [-]
            float deltaOverH = 0.1;
		    float fricSlope = myVel.norm() * myVel.norm() / (cbrt(myDepth) * myDepth * myFrPar * myFrPar); // [-]
            float froude = myVel.norm() / sqrt(gravity * myDepth); // [-]
		    float deltaOverH = pow(((bedSlope - fricSlope) * pow(lambdaOverH, 1.2f))/(0.47f * froude * froude), 0.73f); // [-]
            float formSlopeA = mParameter * pow(deltaOverH, nParameter) * froude * froude * pow(1.0f/lambdaOverH, 1.0f + nParameter); // [-]
            float formSlopeB = deltaOverH / pow(1.0f - pow(deltaOverH / 2.0f, 2.0f), 2.0f); // [-]
            float formSlope = formSlopeA * formSlopeB; // [-]
            myFrCoef = myFrCoef + max(0.0f, formSlope * gravity / myDepth); // [-]
		}*/

		myTau.x = myFrCoef * myVel.x * myVel.norm() /* * 2.0f*/;
		myTau.y = myFrCoef * myVel.y * myVel.norm() /* * 2.0f*/;

		/*vector2D binghamTau;
        binghamTau.x = 0.2f + 3.0f * 0.2f * myVel.x / min(myDepth, myDepth * myDepth);
        binghamTau.y = 0.2f + 3.0f * 0.2f * myVel.y / min(myDepth, myDepth * myDepth);

        myTau.x += binghamTau.x * balanceFactor * 2.0f;
        myTau.y += binghamTau.y * balanceFactor * 2.0f;*/

		if (isValid(myTau)){
            float wetCoef = getWetCoef(myDepth, lowDepth, highDepth);
			myVel.x = copysign(max(abs(myVel.x) - abs(dt/2.0f * myTau.x * (1.0f/myDepth) * wetCoef), 0.0f), myVel.x);
			myVel.y = copysign(max(abs(myVel.y) - abs(dt/2.0f * myTau.y * (1.0f/myDepth) * wetCoef), 0.0f), myVel.y);
			if (isValid(myVel.norm())){
				flow->state->vel = myVel;
				tau = myTau;
			}else{
                flow->state->vel.setNull();
                tau.setNull();
            }
		}
	}else{
        tau.setNull();
	}
}

CPU GPU void elementBed::applyMobileBed() {

#	ifdef __CUDA_ARCH__
	sedimentType* grain = gpuBed.grain;
	float maxConc = gpuBed.maxConc;
	unsigned depEroOption = gpuBed.depEroOption;
	unsigned adaptLenOption = gpuBed.adaptLenOption;
	float dt = float(gpuNumerics.dt);
#	else
	sedimentType* grain = cpuBed.grain;
	float maxConc = cpuBed.maxConc;
	unsigned depEroOption = cpuBed.depEroOption;
	unsigned adaptLenOption = cpuBed.adaptLenOption;
	float dt = float(cpuNumerics.dt);
	using std::max;
	using std::min;
	using std::sqrt;
#	endif

	float myBedrock = bedrock;
	float myTauNorm = tau.norm();

	float myVelNorm = flow->state->vel.norm();
	float myDepth = flow->state->h;
	float myDensity = flow->state->rho;
	float myZb = flow->state->z;

	float myMu = flow->getMu();

	float mySedC[maxFRACTIONS] = { 0.0f };
	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		mySedC[p] = flow->scalars->specie[sedCp];

	float myBedPercentage[maxFRACTIONS] = { 0.0f };
	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		myBedPercentage[p] = comp->bedPercentage[p];

	/*float mySubPercentage[maxFRACTIONS] = { 0.0f };
	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		mySubPercentage[p] = comp->subPercentage[p];*/

	float HC[maxFRACTIONS] = { 0.0f };
	float depEro[maxFRACTIONS] = { 0.0f };

	float activeLayerDepth = 5.0f*getActiveDepth(myBedPercentage, 90.0f);
	activeLayerDepth = max(0.0f, min(activeLayerDepth, myZb - myBedrock));

	float avgDepEro = 0.0f;

	for (unsigned short p = 0; p < maxFRACTIONS; p++) {

		HC[p] = myDepth * mySedC[p];
		denoise(HC[p]);

		float adaptLength = 1.0f;

		if (adaptLenOption == 0)
			adaptLength = 1.0f;
		else if (adaptLenOption == 1) 
			adaptLength = getAdaptLenCanelas(p, myTauNorm, myDensity);
		else if (adaptLenOption == 2)
			adaptLength = getAdaptLenArmini(p, myDepth, myVelNorm, myTauNorm, myDensity, myMu);

		float equilibriumQs = 0.0f;
		if (depEroOption == 1) 
			equilibriumQs = getEqDischFerreira(p, myDepth, myVelNorm, myTauNorm, myDensity);
		else if (depEroOption == 2)
			equilibriumQs = getEqDischBagnold(p, myDepth, myVelNorm, myTauNorm, myDensity, myMu);
		else if (depEroOption == 3)
			equilibriumQs = getEqDischMPM(p, myDepth, myVelNorm, myTauNorm, myDensity);
		else if (depEroOption == 4)
			equilibriumQs = getEqDischSmart(p, myDepth, myVelNorm, myTauNorm, myDensity);

		float actualQs = myVelNorm * myDepth * mySedC[p];
		depEro[p] = dt * (actualQs - equilibriumQs) / adaptLength;
		depEro[p] = max(-activeLayerDepth * myBedPercentage[p], min(depEro[p], HC[p]));
		denoise(depEro[p]);
		
		HC[p] -= depEro[p];
		denoise(HC[p]);

		avgDepEro += depEro[p] * myBedPercentage[p] / (1.0f - grain[p].poro);
	}

	myZb += avgDepEro;
	myDepth = max(0.0f, myDepth - avgDepEro);

	for (unsigned short p = 0; p < maxFRACTIONS; p++){

		mySedC[p] = max(0.0f, min(maxConc, HC[p] / myDepth));
		if (!isValid(activeLayerDepth - avgDepEro)){
			denoise(myBedPercentage[p]);
			continue;
		}

		float tempBedPercentage = myBedPercentage[p] * (activeLayerDepth - depEro[p] / (1.0f - grain[p].poro)) / (activeLayerDepth - avgDepEro);
		myBedPercentage[p] = max(0.0f, min(1.0f, (1.0f - avgDepEro) * myBedPercentage[p] + avgDepEro * tempBedPercentage));
		denoise(myBedPercentage[p]);
	}

	flow->updateDensity();

	flow->state->h = myDepth;
	flow->state->z = max(myZb, myBedrock);

	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		flow->scalars->specie[sedCp] = mySedC[p];

	for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		comp->bedPercentage[p] = myBedPercentage[p];

	/*for (unsigned short p = 0; p < maxFRACTIONS; ++p)
		comp->subPercentage[p] = mySubPercentage[p];*/
}

CPU GPU void bedLandslide::applyLandslide() {

#	ifdef __CUDA_ARCH__
	float currentTime = float(gpuCurrentTime);
	float maxConc = gpuBed.maxConc;
#	else
	float currentTime = cpuSimulation.currentTime;
	float maxConc = cpuBed.maxConc;
#	endif

	if (currentTime >= landslideTimeTrigger) {
		for (unsigned i = 0; i < numElems; ++i) {
			elemLink[i].flow->state->h += landslideDepth;
			elemLink[i].flow->state->vel.setNull();
			elemLink[i].flow->state->z -= landslideDepth;
			if (elemLink[i].bed->bedrock > elemLink[i].flow->state->z)
				elemLink[i].bed->bedrock = elemLink[i].flow->state->z;
			for (unsigned p = 0; p < maxFRACTIONS; ++p)
				elemLink[p].flow->scalars->specie[sedCp] += maxConc;
		}
		isTriggered = true;
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void sedimentType::readData(std::string& grainFileName){

	std::ifstream sedimentTypeFile;
	sedimentTypeFile.open(grainFileName);

	if (!sedimentTypeFile.is_open() || !sedimentTypeFile.good()){
		std::cerr << "   -> *Error* [S-1]: Could not open file " + grainFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	sedimentTypeFile >> diam;
	sedimentTypeFile >> specGrav;
	sedimentTypeFile >> poro;
	sedimentTypeFile >> rest;
	sedimentTypeFile >> alpha;
	sedimentTypeFile >> beta;
	sedimentTypeFile >> tanPhi;
	sedimentTypeFile >> adaptLenMinMult;
	sedimentTypeFile >> adaptLenMaxMult;
	sedimentTypeFile >> adaptLenRefShields;
	sedimentTypeFile >> adaptLenShapeFactor;
	sedimentTypeFile.close();
}

void bedParameters::readAllFiles(std::string& bedFile, std::string& layersFile, std::string& layersFolder, std::string& grainsFolder, std::string& curvesFolder){

	std::ifstream bedControlFile;
	bedControlFile.open(bedFile);

	if (!bedControlFile.is_open() || !bedControlFile.good()){
		std::cerr << "   -> *Error* [S-2]: Could not open file " + bedFile + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + bedFile << std::endl;

	bedControlFile >> frictionOption;
	bedControlFile >> depEroOption;
	bedControlFile >> adaptLenOption;
	bedControlFile >> mobileBedTimeTrigger;
	bedControlFile >> frictionCoef;
	bedControlFile >> bedrockOffset;

	if (frictionOption != 0){
		useFriction = true;
		for (unsigned i = 0; i < cpuMesh.numElems; ++i)
			cpuMesh.elemBed[i].fricPar = frictionCoef;
	}

	if (depEroOption != 0)
		useMobileBed = true;

	std::string inputText;
	std::getline(bedControlFile, inputText);
	std::getline(bedControlFile, inputText);

	unsigned numSedFractions;
	bedControlFile >> numSedFractions;

	std::ifstream sedimentTypeFile;
	std::string sedimentTypeFileName;
	for (unsigned p = 0; p < numSedFractions; ++p){
		bedControlFile >> inputText;
		sedimentTypeFileName = grainsFolder + inputText;
		if (p < maxFRACTIONS)
			grain[p].readData(sedimentTypeFileName);
	}

	getline(bedControlFile, inputText);

	bedControlFile >> numGradingCurves;
	std::vector<std::string> curveFileNames;

	if (numGradingCurves > 0){
		curveFileNames.resize(numGradingCurves);
		if (numGradingCurves > 1)
			useGradeIDMap = true;
		for (unsigned g = 0; g < numGradingCurves; ++g){
			bedControlFile >> inputText;
			curveFileNames[g] = curvesFolder + inputText;
		}
	}

	bedControlFile.close();

	std::ifstream bedLayersFile;
	bedLayersFile.open(layersFile);

	if (!bedLayersFile.is_open() || !bedLayersFile.good()){
		std::cerr << "   -> *Error* [S-3]: Could not open file " + layersFile + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	while(!bedLayersFile.eof()){
		
		std::string bedlayerFileName;

		getline(bedLayersFile, inputText);
		if (inputText == "FricCfMap"){
			useFrictionMap = true;
			getline(bedLayersFile, bedlayerFileName);
			bedlayerFileName = layersFolder + bedlayerFileName;
			importFrictionCoef(bedlayerFileName);
		}

        getline(bedLayersFile, inputText);
        if (inputText == "InfiltCfMap"){
            useInfiltrationMap = true;
            getline(bedLayersFile, bedlayerFileName);
            bedlayerFileName = layersFolder + bedlayerFileName;
            importInfiltrationCoef(bedlayerFileName);
        }

		getline(bedLayersFile, inputText);
        if (inputText == "BedrockMap"){
            useBedrockMap = true;
            getline(bedLayersFile, bedlayerFileName);
            bedlayerFileName = layersFolder + bedlayerFileName;
            importBedrockOffset(bedlayerFileName);
        }

		getline(bedLayersFile, inputText);
		if (inputText == "GradeIdMap"){
			useGradeIDMap = true;
			getline(bedLayersFile, bedlayerFileName);
			bedlayerFileName = layersFolder + bedlayerFileName;
			importBedComposition(bedlayerFileName, curveFileNames);
		}

		getline(bedLayersFile, inputText);
		if (inputText == "Landslides"){
			/*useLandslides = true*/;
			bedlayerFileName = layersFolder + bedlayerFileName;
			getline(bedLayersFile, bedlayerFileName);
			/*importLandslideDepth(layersFolder ...);*/
		}

		getline(bedLayersFile, inputText);
		if (inputText == "LandslidesID"){
			/*useLandslidesID = true*/;
			bedlayerFileName = layersFolder + bedlayerFileName;
			getline(bedLayersFile, bedlayerFileName);
			/*importLandslideID(layersFolder ...);*/
		}

		getline(bedLayersFile, inputText);
	}

	bedLayersFile.close();
}

void bedParameters::importBedrockOffset(std::string& bedrockOffsetFileName){

	if (useBedrockMap){
		
		std::ifstream bedrockMap;
		bedrockMap.open(bedrockOffsetFileName);
		
		if (!bedrockMap.is_open() || !bedrockMap.good()){
			std::cerr << "   -> *Error* [S-4]: Could not open file " + bedrockOffsetFileName + " ... aborting!" << std::endl;
			exitOnKeypress(1);
		}

		for (unsigned n = 0; n < cpuMesh.numNodes; ++n){
			bedrockMap >> cpuMesh.elemNode[n].bedrockOffset;
			showProgress(int(n), int(cpuMesh.numNodes), "   -> Reading", "BedrockMap");
		}

		bedrockMap.close();

	}else
		for (unsigned n = 0; n < cpuMesh.numNodes; ++n)
			cpuMesh.elemNode[n].bedrockOffset = bedrockOffset;

	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		float elementBedrockOffset = std::max(0.0f, (1.0f / 3.0f)*(cpuMesh.elem[i].meta->node[0]->bedrockOffset
			+ cpuMesh.elem[i].meta->node[1]->bedrockOffset + cpuMesh.elem[i].meta->node[2]->bedrockOffset));
		cpuMesh.elem[i].bed->bedrock = cpuMesh.elem[i].flow->state->z - std::max(0.0f, elementBedrockOffset);

        /*float x = cpuMesh.elem[i].meta->center.x;
        float y = cpuMesh.elem[i].meta->center.y;
        float z = cpuMesh.elem[i].flow->state->z;
        float offset = elementBedrockOffset;

        if(x >= -884684.0 && x <= -881919.0)
            if(y >= 4516548.0 && y <= 4518584.0)
                if(offset == 0.0)
                    cpuMesh.elem[i].flow->state->h = max(262.5f - z, 0.0f);*/

        showProgress(int(i), int(cpuMesh.numElems), "   -> Setting", "bedrock");
	}

	std::cout << std::endl;
}

void bedParameters::importFrictionCoef(std::string& fricCoefFileName){

	if (useFrictionMap){

		std::ifstream frictionCoefMap;
		frictionCoefMap.open(fricCoefFileName);

		if (!frictionCoefMap.is_open() || !frictionCoefMap.good()){
			std::cerr << "   -> *Error* [S-5]: Could not open file " + fricCoefFileName + " ... aborting!" << std::endl;
			exitOnKeypress(1);
		}

		for (unsigned n = 0; n < cpuMesh.numNodes; ++n){
			frictionCoefMap >> cpuMesh.elemNode[n].fricPar;
			showProgress(int(n), int(cpuMesh.numNodes), "   -> Reading", "FricMap");
		}

		frictionCoefMap.close();

	}else
		for (unsigned n = 0; n < cpuMesh.numNodes; ++n)
			cpuMesh.elemNode[n].fricPar = cpuBed.frictionCoef;

	for (unsigned i = 0; i < cpuMesh.numElems; ++i){
		float elementFrPar = std::max(0.0f, (1.0f / 3.0f)*(cpuMesh.elem[i].meta->node[0]->fricPar
			+ cpuMesh.elem[i].meta->node[1]->fricPar + cpuMesh.elem[i].meta->node[2]->fricPar));
		cpuMesh.elem[i].bed->fricPar = elementFrPar;
		showProgress(int(i), int(cpuMesh.numElems), "   -> Setting", "fricCf");
	}

	std::cout << std::endl;
}

void bedParameters::importInfiltrationCoef(std::string& infiltCoefFileName){

    if (useInfiltrationMap){

        std::ifstream infiltrationCoefMap;
        infiltrationCoefMap.open(infiltCoefFileName);

        if (!infiltrationCoefMap.is_open() || !infiltrationCoefMap.good()){
            std::cerr << "   -> *Error* [S-5]: Could not open file " + infiltCoefFileName + " ... aborting!" << std::endl;
            exitOnKeypress(1);
        }

        for (unsigned n = 0; n < cpuMesh.numNodes; ++n){
            infiltrationCoefMap >> cpuMesh.elemNode[n].infilPar;
            showProgress(int(n), int(cpuMesh.numNodes), "   -> Reading", "InfiltMap");
        }

        infiltrationCoefMap.close();

    }else
        for (unsigned n = 0; n < cpuMesh.numNodes; ++n)
            cpuMesh.elemNode[n].fricPar = cpuBed.frictionCoef;

    for (unsigned i = 0; i < cpuMesh.numElems; ++i){
        float elementInfilPar = std::max(0.0f, (1.0f / 3.0f)*(cpuMesh.elem[i].meta->node[0]->infilPar
                                                           + cpuMesh.elem[i].meta->node[1]->infilPar
                                                           + cpuMesh.elem[i].meta->node[2]->infilPar));
        cpuMesh.elem[i].forcing->infilPar = elementInfilPar;
        showProgress(int(i), int(cpuMesh.numElems), "   -> Setting", "InfitfCf");
    }

    std::cout << std::endl;
}

void bedParameters::importBedComposition(std::string& curveIdFileName, std::vector<std::string>& curveFileNames){

	if (maxFRACTIONS <= 1 || !useGradeIDMap)
		return;

	std::ifstream curveFile;
	std::vector< std::vector<float> > gradeCurvePercentage(numGradingCurves, std::vector<float>(maxFRACTIONS, 0.0f));

	for (unsigned g = 0; g < numGradingCurves; ++g){

		curveFile.open(curveFileNames[g]);

		if (!curveFile.is_open() || !curveFile.good()){
			std::cerr << "   -> *Error* [S-6]: Could not open file " + curveFileNames[g] + " ... aborting!" << std::endl;
			exitOnKeypress(1);
		}

		for (unsigned p = 0; p < maxFRACTIONS; ++p)
			curveFile >> gradeCurvePercentage[g][p];

		curveFile.close();
	}

	if (useGradeIDMap){

		std::ifstream curveIdMap;
		curveIdMap.open(curveIdFileName);

		if (!curveIdMap.is_open() || !curveIdMap.good()){
			std::cerr << "   -> *Error* [S-7]: Could not open file " + curveIdFileName + " ... aborting!" << std::endl;
			exitOnKeypress(1);
		}

		for (unsigned n = 0; n < cpuMesh.numNodes; n++){
			curveIdMap >> cpuMesh.elemNode[n].curveID;
			showProgress(int(n), int(cpuMesh.numNodes), "   -> Reading", "curveID");
		}

		curveIdMap.close();

		for (unsigned i = 0; i < cpuMesh.numElems; i++){
			unsigned elementGradeID = unsigned(std::round((1.0f / 3.0f)*(float(cpuMesh.elem[i].meta->node[0]->curveID +
				cpuMesh.elem[i].meta->node[1]->curveID + cpuMesh.elem[i].meta->node[2]->curveID))));
			elementGradeID = std::max(0u, std::min(numGradingCurves - 1u, elementGradeID));
			for (unsigned p = 0; p < maxFRACTIONS; p++)
				cpuMesh.elem[i].bed->comp->bedPercentage[p] = gradeCurvePercentage[elementGradeID][p];

			showProgress(int(i), int(cpuMesh.numElems), "   -> Setting", "curveID");
		}
	}else
		for (unsigned i = 0; i < cpuMesh.numElems; i++)
			for (unsigned p = 0; p < maxFRACTIONS; p++)
				cpuMesh.elem[i].bed->comp->bedPercentage[p] = gradeCurvePercentage[0][p];

	std::cout << std::endl;
}


/* TODO (Landslides)

void readLandslidesTriggerFile(){

}

void importLandslidesMap(){

	if (useLandslides){
		std::ifstream landslidesDepthFile;
		landslidesDepthFile.open("./sediment/LandslidesMap.nod");
		for (unsigned i = 0; i < cpuNode.size(); i++){
			landslidesDepthFile >> cpuNode[i].landslideDepth;
			showProgress(i, (int)cpuNode.size(), "   -> Reading", "LandslidesMap");
		}
		landslidesDepthFile.close();
	}else{
		return;
	}

	std::set<unsigned> uniqueLandslideID;

	if (useLandslidesID){
		std::ifstream landslidesIDFile;
		landslidesIDFile.open("./sediment/LandslidesIDMap.nod");
		for (unsigned i = 0; i < cpuNode.size(); i++){
			landslidesIDFile >> cpuNode[i].landslideID;
			if (cpuNode[i].landslideID > 0){
				uniqueLandslideID.insert(cpuNode[i].landslideID);
			}
			showProgress(i, (int)cpuNode.size(), "   -> Reading", "LandslidesIDM");
		}
		cpuLandslide.resize(uniqueLandslideID.size());
		landslidesIDFile.close();
	}else{
		cpuLandslide.resize(1);
	}

	if (useLandslides)
		readLandslidesTriggerFile();

	uniqueLandslideID.clear();

	std::vector<unsigned> elementLandslideIDs(cpuElem.size(), 0);
	std::vector<float> elementLandslideDepths(cpuElem.size(), 0);
	for (unsigned i = 0; i < cpuElem.size(); i++){
		unsigned elementLandslideID = std::max(0u, std::min((unsigned)cpuLandslide.size() - 1u, (unsigned)std::round((1.0f / 3.0f)*((float)cpuElem[i].meta->no[0]->landslideID + (float)cpuElem[i].meta->no[1]->landslideID + (float)cpuElem[i].meta->no[2]->landslideID))));
		float elementLandslideDepth = std::max(0.0f, (1.0f / 3.0f)*(cpuElem[i].meta->no[0]->landslideDepth + cpuElem[i].meta->no[1]->landslideDepth + cpuElem[i].meta->no[2]->landslideDepth));
		if (elementLandslideID > 0u && cpuLandslide.size() > 1){
			++cpuLandslide[elementLandslideID - 1].size;
			elementLandslideIDs[i] = elementLandslideID;
			elementLandslideDepths[i] = elementLandslideDepth;
		}else if (elementLandslideID == 0 && !useLandslidesID && elementLandslideDepth >= 0.01f){
			++cpuLandslide[0].size;
			elementLandslideIDs[i] = 0;
			elementLandslideDepths[i] = elementLandslideDepth;
		}
	}

	for (unsigned l = 0; l < cpuLandslide.size(); l++){
		cpuLandslide[l].elemBedLink = new elementBed*[cpuLandslide[l].size];
	}

	std::vector<unsigned> adHocIterator(cpuLandslide.size(), 0);
	for (unsigned i = 0; i < cpuElem.size(); i++){
		if (elementLandslideIDs[i] > 0u && cpuLandslide.size() > 1){
			unsigned idx = adHocIterator[elementLandslideIDs[i]]++;
			cpuLandslide[elementLandslideIDs[i] - 1].elemBedLink[idx] = cpuElem[i].bed;
			cpuLandslide[elementLandslideIDs[i] - 1].landslideDepth = elementLandslideDepths[i];
		}
		showProgress(i, (int)cpuElem.size(), "   -> Setting", "landslides");
	}

	elementLandslideIDs.clear();
	elementLandslideDepths.clear();
	adHocIterator.clear();
	std::cout << std::endl;
}*/
