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
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

// STAV
#include "../headers/control.hpp"
#include "../headers/geometry.hpp"
#include "../headers/forcing.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU elementForcing::elementForcing(){

	state = 0x0;
	center = 0x0;

	for (unsigned short t = 0; t < maxNN; ++t)
		rainGauge[t] = 0x0;

	numRainGauges = 0;
	rainInt = 0.0f;

    infilPar = 0.0;
    infilTotal = 0.0;
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void forcingParameters::readControlFile(std::string& controlFileName, std::string& controlSubFolder){

	std::string inputText;
	std::ifstream controlFile;
	controlFile.open(controlFileName);

	if (!controlFile.is_open() || !controlFile.good()){
		std::cerr << "   -> *Error* [F-1]: Could not open file " + controlFileName + " ... aborting!" << std::endl;
		exitOnKeypress(1);
	}

	std::cout << "   -> Reading " + controlFileName << std::endl;
	controlFile >> useRainfall;
    controlFile >> constantRainfall;
    controlFile >> infiltrationOption;
    controlFile >> infilParameter1;
    controlFile >> infilParameter2;

    if (infiltrationOption != 0)
        useInfiltration = true;

	getline(controlFile, inputText);
	controlFile >> numRainGauges;

	if (useRainfall && numRainGauges > 0){
		rainGauge = new timeseries[numRainGauges];
		for (unsigned f = 0; f < numRainGauges; ++f){
			std::string gaugeFileName;
			controlFile >> rainGauge[f].position.x >> rainGauge[f].position.y >> gaugeFileName;
			rainGauge[f].readData(controlSubFolder + gaugeFileName);
		}
	}

	controlFile.close();
}
