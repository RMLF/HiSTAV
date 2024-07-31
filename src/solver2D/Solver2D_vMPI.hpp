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
#include <iostream>
#include <iomanip>
#include <algorithm>

// OpenMP
#include <omp.h>

// MPI
#include <mpi.h>

// STAV
#include "headers/compile.hpp"
#include "headers/geometry.hpp"
#include "headers/initial.hpp"
#include "headers/common.hpp"
#include "headers/control.hpp"
#include "headers/numerics.hpp"
#include "headers/forcing.hpp"
#include "headers/sediment.hpp"
#include "headers/output.hpp"
#include "headers/mesh.hpp"
#include "headers/boundaries.hpp"
//#include "headers/lagrangian.hpp"
#include "headers/simulation.hpp"

#ifdef __STAV_MPI__
#include "headers/mpiIO.hpp"
#include "headers/mpiRun.hpp"
#endif

/////////////////////////////////////////////////////////////////////////////////////////////////