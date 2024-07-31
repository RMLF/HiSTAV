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

// CUDA
#ifdef __STAV_CUDA__
	#include <cuda.h>
	#include <cuda_runtime.h>
	#define CPU __host__
	#define GPU __device__
	#define GLOBAL __global__
	#define INLINE __forceinline__
#else
	#define CPU
	#define GPU
	#define GLOBAL
	#define INLINE inline
#endif

#ifndef __CUDA_ARCH__
	using namespace std;
#endif

// MPI
#ifdef __STAV_MPI__
	#include <mpi.h>
#endif

// Definitions and Dimensions
// Hydrodynamics (2D): DO NOT change!
#define maxCONSERVED 3
#define maxLAYERS 1

// Sediments: change ONLY maxFRACTIONS (At least: one sediment fraction)
#define sedC 0
#define sedCp (sedC+p) // use with indexer 'unsigned short p ='
#define maxFRACTIONS 1

// Environmental: change ONLY maxENVIRON (At least: temperature)
#define relTemp maxFRACTIONS
#define maxENVIRON 1

// Total scalars:  DO NOT change!
#define maxSCALARS (maxFRACTIONS + maxENVIRON)

/////////////////////////////////////////////////////////////////////////////////////////////////
