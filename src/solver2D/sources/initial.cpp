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
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <iostream>

// STAV
#include "../headers/compile.hpp"
#include "../headers/common.hpp"
#include "../headers/control.hpp"
#include "../headers/geometry.hpp"
#include "../headers/numerics.hpp"
#include "../headers/sediment.hpp"
#include "../headers/mesh.hpp"
#include "../headers/simulation.hpp"
#include "../headers/output.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


void initialParameters::setInitialConditions(){

	std::cout << std::endl;
	std::cout << "  -> Setting initial conditions" << std::endl;

	if(resumeSimulation)
		loadInitialConditions();
	else{
		for(unsigned i = 0; i < cpuMesh.numElems; ++i){
			if (iniValueType == 1)
				cpuMesh.elem[i].flow->state->h = iniValue;
			else if (iniValueType == 2)
				cpuMesh.elem[i].flow->state->h = std::max(0.0f, iniValue - cpuMesh.elem[i].flow->state->z);
			else{
				std::cerr << std::endl << "   -> Error [I-1]: Invalid initial condition type" << std::endl;
				exitOnKeypress(1);
			}

			if (cpuMesh.elem[i].meta->center.x >= -884824.0f && cpuMesh.elem[i].meta->center.x <= -881911.0f)
				if (cpuMesh.elem[i].meta->center.y >= 4516420.0f && cpuMesh.elem[i].meta->center.y <= 4518682.0f){
					if (cpuMesh.elem[i].bed->bedrock >= -0.1f && cpuMesh.elem[i].bed->bedrock <= 0.1f)
						cpuMesh.elem[i].flow->state->h = std::max(0.0f, 254.0f - cpuMesh.elem[i].flow->state->z);
					else
						cpuMesh.elem[i].flow->state->h = 0.0f;
				}

			/*if (cpuMesh.elem[i].bed->bedrock >= 910.0 && cpuMesh.elem[i].bed->bedrock <= 930.0)
				cpuMesh.elem[i].flow->state->h = std::max(0.0f, 919.5f - cpuMesh.elem[i].flow->state->z);
			else if (cpuMesh.elem[i].bed->bedrock >= 754.9 && cpuMesh.elem[i].bed->bedrock <= 755.1)
				cpuMesh.elem[i].flow->state->h = std::max(0.0f, 754.5f - cpuMesh.elem[i].flow->state->z);
			else
				cpuMesh.elem[i].flow->state->h = 0.0f;*/

			//Lisbon 2018
			/*float xRef = -1038000.0f;
			float dist = 36650.0f;

			float hLeft = 4.98f;
			float hRight = 5.83f;
			float slope = (hRight - hLeft) / dist;

			float surfaceLevel = hLeft + (cpuMesh.elem[i].meta->center.x - xRef)*slope;
			cpuMesh.elem[i].flow->state->h = std::max(0.0f, surfaceLevel - cpuMesh.elem[i].flow->state->z);*/

			//cpuMesh.elem[i].flow->state->h = std::exp( (pow((cpuMesh.elem[i].meta->center.x - 5.0f), 2.0f) + pow((cpuMesh.elem[i].meta->center.y - 5.0f), 2.0f)) / 20.0f );
			//cpuMesh.elem[i].flow->state->z = 0.0f; //std::pow((cpuMesh.elem[i].meta->center.y-1.0f),2.0f)*2.5f;
			//cpuMesh.elem[i].flow->state->h = 0.25f; //std::max(0.0f, 0.5f - cpuMesh.elem[i].meta->center.x/10.0f - cpuMesh.elem[i].flow->state->z);

			// Special cases go here!
			/*for (unsigned k = 0; k < 3; ++k)
				if (cpuMesh.elem[i].meta->elem[k] != 0x0)
					if (cpuMesh.elem[i].meta->elem[k]->meta->center.distXY(point(7.5f, 5.0f, 0.0f)) < 0.5f){
						cpuMesh.elem[i].flow->flux->neighbor[k].state = 0x0;
						cpuMesh.elem[i].flow->flux->neighbor[k].scalars = 0x0;
					}

			if (cpuMesh.elem[i].meta->center.distXY(point(7.5f, 5.0f, 0.0f)) < 0.5f){
				if (cpuMesh.elem[i].flow->state != 0x0){
					cpuMesh.elem[i].flow->state->h = 0.0f;
					cpuMesh.elem[i].flow->state->z = 2.5f;
				}

				for (unsigned k = 0; k < 3; ++k)
					if (cpuMesh.elem[i].meta->elem[k] != 0x0){
						cpuMesh.elem[i].flow->flux->neighbor[k].state = 0x0;
						cpuMesh.elem[i].flow->flux->neighbor[k].scalars = 0x0;
					}
			}else{

				if (cpuMesh.elem[i].meta->center.x < 6.0f)
					cpuMesh.elem[i].flow->state->h = 2.2f - cpuMesh.elem[i].meta->center.x / 5.0f;
				else
					cpuMesh.elem[i].flow->state->h = 0;

				if (cpuMesh.elem[i].meta->center.distXY(point(4.0f, 5.0f, 0.0f)) < 1.0f){
					cpuMesh.elem[i].flow->scalars->specie[sedC] = 0.5f;
					cpuMesh.elem[i].flow->scalars->setTemp(25.0f);
				}else{
					cpuMesh.elem[i].flow->scalars->specie[sedC] = 0.0f;
					cpuMesh.elem[i].flow->scalars->setTemp(20.0f);
				}
			}*/
			
			cpuMesh.elemOutput[i].zIni = cpuMesh.elemOutput[i].link->flow->state->z;
			cpuMesh.elemOutput[i].max.zIni = cpuMesh.elemOutput[i].link->flow->state->z;
			showProgress(int(i), int(cpuMesh.numElems), "    -> Setting", "IC's   ");
		}
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
			getline(initialControlFile, inputText);
			if (!initialControlFile.eof())
				getline(initialControlFile, hydroResumeFileName);
			if (!initialControlFile.eof())
				getline(initialControlFile, bedResumeFileName);
			if (!initialControlFile.eof())
				getline(initialControlFile, scalarsResumeFileName);
			if (!initialControlFile.eof())
				getline(initialControlFile, maximaResumeFileName);
		}
	}

	initialControlFile.close();
}

void initialParameters::loadInitialConditions(){

}
