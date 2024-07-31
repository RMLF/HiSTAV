/*///////////////////////////////////////////////////////////////////////////////////////////////

STAV-2D Hydrodynamic Model

Copyright (C) 2018
Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira
Instituto Superior Técnico - Universidade de Lisboa
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

// STAV
#include "compile.hpp"

// Definitions
#define fEPSILON 1.0e-6f

/////////////////////////////////////////////////////////////////////////////////////////////////


class point{
public:

	INLINE CPU GPU point();
	INLINE CPU GPU point(const float, const float, const float);

	INLINE CPU GPU point operator+(const point&);
	INLINE CPU GPU point operator*(const float&);
	INLINE CPU GPU float distXY(const point&);

	float x;
	float y;
	float z;
};

class vector2D{
public:

	INLINE CPU GPU vector2D();
	INLINE CPU GPU vector2D(const float, const float);
	INLINE CPU GPU vector2D(const point&, const point&);

	INLINE CPU GPU vector2D operator+(const vector2D&);
	INLINE CPU GPU vector2D operator-(const vector2D&);
	INLINE CPU GPU vector2D operator-();
	INLINE CPU GPU vector2D operator*(const float&);
	INLINE CPU GPU vector2D operator/(const float&);
	INLINE CPU GPU float dot(const vector2D&);
	
	INLINE CPU GPU float norm();
	INLINE CPU GPU void normalize();
	INLINE CPU GPU void setNull();

	float x;
	float y;
};



/////////////////////////////////////////////////////////////////////////////////////////////////
// FORCED INLINE FUNCTIONS - These functions are defined and inlined here for better performance!
/////////////////////////////////////////////////////////////////////////////////////////////////


INLINE CPU GPU point::point(){
	x = 0.0f;
	y = 0.0f;
	z = 0.0f;
}

INLINE CPU GPU point::point(const float inputX, const float inputY, const float inputZ){
	x = inputX;
	y = inputY;
	z = inputZ;
}

INLINE CPU GPU point point::operator+(const point& other){
	return point(x + other.x, y + other.y, z + other.z);
}

INLINE CPU GPU point point::operator*(const float& scalar){
	return point(x*scalar, y*scalar, z*scalar);
}

INLINE CPU GPU float point::distXY(const point& other){
#	ifndef __CUDA_ARCH__
	using std::sqrt;
#	endif
	return sqrt((x - other.x)*(x - other.x) + (y - other.y)*(y - other.y));
}

INLINE CPU GPU vector2D::vector2D(){
	x = 0.0f;
	y = 0.0f;
}

INLINE CPU GPU vector2D::vector2D(const float inputX, const float inputY){
	x = inputX;
	y = inputY;
}

INLINE CPU GPU vector2D::vector2D(const point& start, const point& end){
	x = end.x - start.x;
	y = end.y - start.y;
}

INLINE CPU GPU vector2D vector2D::operator+(const vector2D& other){
	return vector2D(x + other.x, y + other.y);
}

INLINE CPU GPU vector2D vector2D::operator-(const vector2D& other){
	return vector2D(x - other.x, y - other.y);
}

INLINE CPU GPU vector2D vector2D::operator-(){
	return vector2D(-x, -y);
}

INLINE CPU GPU vector2D vector2D::operator*(const float& factor){
#	ifndef __CUDA_ARCH__
	using std::abs;
#	endif
	if (abs(factor) >= fEPSILON)
		return vector2D(x*factor, y*factor);
	else
		return vector2D(0.0f, 0.0f);
}

INLINE CPU GPU vector2D vector2D::operator/(const float& factor){
	return vector2D(x/factor, y/factor);
}

INLINE CPU GPU void vector2D::setNull(){
	x = 0.0f;
	y = 0.0f;
}

INLINE CPU GPU float vector2D::dot(const vector2D& other){
	return x*other.x + y*other.y;
}

INLINE CPU GPU float vector2D::norm(){
#	ifndef __CUDA_ARCH__
	using std::sqrt;
#	endif
	return sqrt(x*x + y*y);
}

INLINE CPU GPU void vector2D::normalize(){
	float myNorm = norm();
	x /= myNorm;
	y /= myNorm;
}

void dummyFunction(bool&);

#undef fEPSILON