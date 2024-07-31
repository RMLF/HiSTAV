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

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include "../headers/compile.hpp"
#include "../headers/common.hpp"
#include "../headers/control.hpp"
#include "../headers/geometry.hpp"
#include "../headers/numerics.hpp"
#include "../headers/sediment.hpp"
#include "../headers/mesh.hpp"
#include "../headers/simulation.hpp"
#include "../headers/output.hpp"

/*/////////////////////////////////////////////////////////////////////////////////////////////*/
//endregion

void initialParameters::setInitialConditions(){

	std::cout << std::endl;
	std::cout << "  -> Setting initial conditions" << std::endl;

    if(resumeSimulation)
        loadInitialConditions();

    /*
    if(assimilateData)
        loadOpenDAStatesFile();
    */

    for(unsigned i = 0; i < cpuMesh.numElems; ++i){

        float depth = 0.0;

        if (iniValueType == 1)
            depth = std::max(0.0f, iniValue);
        else if (iniValueType == 2)
            depth = std::max(0.0f, iniValue - cpuMesh.elem[i].flow->state->z);
        else{
            std::cerr << std::endl << "   -> Error [I-1]: Invalid initial condition type" << std::endl;
            exitOnKeypress(1);
        }

        /*
        float x = cpuMesh.elem[i].meta->center.x;
        float y = cpuMesh.elem[i].meta->center.y;
        float zb = std::max(0.0f, (0.0f + x)/10.0f); //-5.0f * (1.0f - (x*x + y*y)/1.0f);
        float wse = x < -0.3f ? 0.1f : 0.0f; //1.0f * (1.0f - (x*x + y*y)/1.0f);
        depth = std::max(wse - zb, 0.0f);
        cpuMesh.elem[i].flow->state->h = depth;
        cpuMesh.elem[i].flow->state->z = zb;
        */

        cpuMesh.elem[i].flow->state->h = depth;

        cpuMesh.elemOutput[i].zIni = cpuMesh.elemOutput[i].link->flow->state->z;
        cpuMesh.elemOutput[i].max.zIni = cpuMesh.elemOutput[i].link->flow->state->z;
        showProgress(int(i), int(cpuMesh.numElems), "    -> Setting", "IC's   ");
    }

	std::cout << std::endl;
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void initialParameters::readControlFile(){

	unsigned inputInt;

	std::string inputText;
	std::ifstream initialControlFile;
	initialControlFile.open(resumeFolder + resumeControlFileName);

	if (!initialControlFile.is_open() || !initialControlFile.good()){
		std::cerr << std::endl << "   -> *Error* [I-2]: Could not open file: " + resumeFolder + resumeControlFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}else{
		std::cout << "   -> Reading " + resumeFolder + resumeControlFileName << std::endl;
		initialControlFile >> iniValueType;
		initialControlFile >> iniValue;
		getline(initialControlFile, inputText);
		getline(initialControlFile, inputText);
		initialControlFile >> inputInt;

		if (inputInt > 0){
			resumeSimulation = true;
			initialControlFile >> cpuSimulation.step;
            initialControlFile >> cpuSimulation.initialTime;
            cpuSimulation.currentTime = cpuSimulation.initialTime;
			getline(initialControlFile, inputText);
			if (!initialControlFile.eof()){
                getline(initialControlFile,inputText);
                std::stringstream ss(inputText);
                ss >> hydroResumeFileName;
			}
		}
	}

	initialControlFile.close();
}

void initialParameters::loadInitialConditions(){

    std::string inputText;

    std::ifstream hydroIn;
    hydroIn.open("./initial/" + hydroResumeFileName);

    if (!hydroIn.is_open() || !hydroIn.good()) {
        std::cerr << std::endl << "   -> *Error* [I-3]: Could not open file: ./initial/" + hydroResumeFileName + " ... aborting!" << std::endl;
        exitOnKeypress(1);
    }

    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);

    for (unsigned i = 0; i < cpuMesh.numNodes; ++i)
        getline(hydroIn, inputText);

    getline(hydroIn, inputText);
    getline(hydroIn, inputText);

    for (unsigned i = 0; i < cpuMesh.numElems; ++i)
        getline(hydroIn, inputText);

    getline(hydroIn, inputText);
    getline(hydroIn, inputText);

    for (unsigned i = 0; i < cpuMesh.numElems; ++i)
        getline(hydroIn, inputText);

    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);

    for (unsigned i = 0; i < cpuMesh.numElems; ++i)
        getline(hydroIn, inputText);

    getline(hydroIn, inputText);
    getline(hydroIn, inputText);
    getline(hydroIn, inputText);

    for (unsigned i = 0; i < cpuMesh.numElems; ++i) {
        getline(hydroIn, inputText);
        float wse = std::stof(inputText);
        float z = cpuMesh.elemFlow[i].state->z;
        float depth = wse - z;
        cpuMesh.elemFlow[i].state->h = depth;
    }

    getline(hydroIn, inputText);
    getline(hydroIn, inputText);

    for (unsigned i = 0; i < cpuMesh.numElems; ++i) {
        getline(hydroIn, inputText, '\t');
        cpuMesh.elemFlow[i].state->vel.x = std::stof(inputText);
        getline(hydroIn, inputText,'\t');
        cpuMesh.elemFlow[i].state->vel.y = std::stof(inputText);
        getline(hydroIn, inputText);
    }

    hydroIn.close();
}
