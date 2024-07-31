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
#pragma once

// STL
#include <vector>
#include <cstdint>

// Forward Declarations
class node;
class facet;
class edge;

/////////////////////////////////////////////////////////////////////////////////////////////////


class node{
public:

	node();
	node(const double&, const double&, const double&);
	node operator+(const node&);
	node operator-(const node&);
	void operator=(const node&);
	
	double distXY(const node&);
	double distXYZ(const node&);
	void setSpfOrder();

	std::vector<facet*> facets;
	std::vector<edge*> edges;

	uint64_t spfOrder;

	double x;
	double y;
	double z;

	double frCoef;
    double permeability;
	double dzMax;
	double slideDepth;

	unsigned id;
	unsigned gradeId;
	unsigned slideId;
	int physId;
};

class facet{
public:

	facet();
	void operator=(const facet&);

	void setSpfOrder();
	void setCcWise();

	double getArea();
	double getSkewness();
	double getAspectRatio();
	double getMinEdgeLength();
	double getMaxEdgeLength();

	std::vector<node*> nodes;
	std::vector<edge*> edges;
	std::vector<facet*> neighs;
	
	node center;
	uint64_t spfOrder;

	unsigned id;
	int physId;
};

class edge{
public:

	edge();
	void operator=(const edge&);

	double getLength();

	std::vector<node*> nodes;
	std::vector<facet*> facets;
	
	unsigned id;
	int physId;
};