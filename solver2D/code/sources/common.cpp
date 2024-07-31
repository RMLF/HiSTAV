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
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>

// STAV
#include "../headers/common.hpp"

// Definitions
#define timeseriesMAX 65000

/////////////////////////////////////////////////////////////////////////////////////////////////


timeseries::timeseries(){

	length = 0;
	present = 0;
	for (unsigned short t = 0; t < timeseriesMAX; ++t){
		time[t] = 0.0f;
		data[t] = 0.0f;
	}
}

void timeseries::addData(float inputTime, float inputValue){

	if (length < timeseriesMAX){
		if (length == 0){
			time[length] = inputTime;
			data[length] = inputValue;
			++length;
		}else if (inputTime > time[length - 1]){
			time[length] = inputTime;
			data[length] = inputValue;
			++length;
		}
	}
}

void timeseries::readData(std::string filename){

	std::ifstream timeSeriesFile;
	timeSeriesFile.open(filename);

	if (!timeSeriesFile.is_open() || !timeSeriesFile.good()){
		std::cerr << std::endl << "   -> *Error* [T-1]: Could not open file " << filename << " ... continuing!" << std::endl;
		exitOnKeypress(1);
	}

	float inputFloat;
	float inputFloat2;
	std::string inputText;

	while (timeSeriesFile >> inputFloat >> inputFloat2){
		addData(inputFloat, inputFloat2);
        if(timeSeriesFile.eof())
            break;
        timeSeriesFile.ignore();
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void exitOnKeypress(int exit_status){

#	ifdef __STAV_MPI__

	std::cout << std::endl;
	if (exit_status == 0)
		std::cout << "  Simulation finished successfully";
	else if(exit_status == 1)
		std::cout << "  Simulation cannot proceed";

#	else

	std::cout << std::endl;
	if (exit_status == 0)
		std::cout << "  Simulation finished successfully. Close console? ";
	else if(exit_status == 1)
		std::cerr << "  Simulation cannot proceed. Close console? ";
	

	std::cin.ignore();
	std::exit(exit_status);

#	endif
}

void showProgress(int progress, int total, std::string action, std::string entity){

	static int progressCounter = -1;

	if(((progress + 1) * 100 / total) / 1 != progressCounter){
		progressCounter = ((progress + 1) * 100 / total) / 1;
		std::cout << "\r" << "    " << action << " " << entity;
		int intPos = (progress + 1) * 50 / total;
		float pos = float((progress + 1)) * 100 / float(total);
		std::cout << "	[";
		for (unsigned short iterProgress = 0; iterProgress < 50; ++iterProgress){
			if(iterProgress < intPos) std::cout << "=";
			else if(iterProgress == intPos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "]   " << std::fixed << std::setprecision(0) << pos << " %";
		std::cout.flush();
	}
}
