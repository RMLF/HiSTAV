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
#include <cmath>
#include <string>
#include <algorithm>

// STAV
#include "compile.hpp"
#include "geometry.hpp"

// Definitions
#define fEPSILON 1.0e-10f
#define dEPSILON 1.0e-12
#define timeseriesMAX 65000

/////////////////////////////////////////////////////////////////////////////////////////////////


class timeseries{
public:

	timeseries();

	void readData(std::string);
	void addData(float, float);
	CPU GPU INLINE float getData();

	float time[timeseriesMAX];
	float data[timeseriesMAX];

	point position;

	unsigned /*short*/ length;
	unsigned /*short*/ present;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance
/////////////////////////////////////////////////////////////////////////////////////////////////


CPU GPU INLINE bool isValid(float inputValue){

#	ifndef __CUDA_ARCH__
	using std::abs;
#	endif
	
	if (abs(inputValue) <= fEPSILON || (inputValue != inputValue))
		return false;

	return true;
}

CPU GPU INLINE bool isValid(double inputValue){

#	ifndef __CUDA_ARCH__
	using std::abs;
#	endif

	if (abs(inputValue) <= dEPSILON || (inputValue != inputValue))
		return false;

	return true;
}

CPU GPU INLINE bool isValid(vector2D& inputVector){

#	ifndef __CUDA_ARCH__
	using std::abs;
#	endif

	if (abs(inputVector.norm()) <= fEPSILON)
		return false;

	if(inputVector.x != inputVector.x)
		return false;

	if(inputVector.y != inputVector.y)
		return false;

	return true;
}

CPU GPU INLINE void denoise(float &inputValue){
	
	if(!isValid(inputValue))
		inputValue = 0.0f;
}

CPU GPU INLINE void denoise(double &inputValue){
	
	if (!isValid(inputValue))
		inputValue = 0.0;
}

CPU GPU INLINE void denoise(vector2D &inputVector){

	if(!isValid(inputVector)){
		inputVector.x = 0.0f;
		inputVector.y = 0.0f;
		return;
	}

	if(!isValid(inputVector.x))
		inputVector.x = 0.0f;

	if(!isValid(inputVector.y))
		inputVector.y = 0.0f;
}

CPU GPU INLINE float sigmoid(float relX){

#	ifndef __CUDA_ARCH__
	using std::max;
	using std::min;
	using std::exp;
#	endif

	if (relX >= 1.0f)
		return 1.0f;
	else if (relX <= 0.0f || !isValid(relX))
		return 0.0f;
	else{
		float weightX = 1.0f / (1.0f + exp(-15.0f*(relX - 0.5f)));
		return max(0.0f, min(weightX, 1.0f));
	}
}

CPU GPU INLINE float sigmoid(double relX){

#	ifndef __CUDA_ARCH__
	using std::max;
	using std::min;
	using std::exp;
#	endif

	if (relX >= 1.0)
		return 1.0f;
	else if (relX <= 0.0 || !isValid(relX))
		return 0.0;
	else{
		double weightX = 1.0 / (1.0 + exp(-12.5*(relX - 0.5)));
		return float(max(0.0, min(weightX, 1.0)));
	}
}



/////////////////////////////////////////////////////////////////////////////////////////////////
// File I/O and auxilliary functions
/////////////////////////////////////////////////////////////////////////////////////////////////


void showProgress(int, int, std::string, std::string);
void exitOnKeypress(int);
