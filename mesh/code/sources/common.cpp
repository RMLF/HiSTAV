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
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <ctime>

// Pre-Processor Headers
#include "../headers/common.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


std::ofstream preProcLog;

void toLog(const std::string& message, const unsigned level){

	if (!preProcLog.is_open())
		preProcLog.open("PreProcessingSTAV.log");

	if (message != ""){
		time_t now = time(0);
		std::string date = std::string(std::ctime(&now));
		date.pop_back();
		preProcLog << date << "		";
		for (unsigned l = 0; l < level; ++l)
			preProcLog << "	";
		preProcLog << message << std::endl;
	}else
		preProcLog << message << std::endl;
}

bool fileExists(const std::string& fileName){
	
	std::ifstream infile(fileName);
	return infile.good();
}

void showProgress(const int progress, const size_t& total, const std::string& action, const std::string& entity){

	static unsigned progressCounter = 0;

	unsigned pos = unsigned(((progress + 1)) * 100 / unsigned(total));
	unsigned intPos = unsigned((progress + 1) * 50 / unsigned(total));

	if (pos > progressCounter){
		progressCounter = pos;
		std::cout << "\r" << "    " << action << " " << entity << "	[";
		for (unsigned i = 0; i < 50; ++i) {
			if (i < intPos) std::cout << "=";
			else if (i == intPos) std::cout << ">";
			else std::cout << " ";
		}
		std::cout << "]   " << std::fixed << std::setprecision(0) << pos << " %" << std::flush;
	}

	if (progress >= (int(total) - 1))
		progressCounter = 0;
}

void exitOnKeypress(const int exitStatus){

	std::cout << std::endl;
	std::cout << "  Press any key to exit...";
	std::cin.ignore();

	exit(exitStatus);
}