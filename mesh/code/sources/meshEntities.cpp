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
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
#include <algorithm>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshEntities.hpp"
#include "../headers/meshGlobals.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


node::node(){

	spfOrder = 0;

	x = 0.0;
	y = 0.0;
	z = 0.0;

	frCoef = 0.0;
    permeability = 0.0;
	dzMax = 0.0;
	slideDepth = 0.0;

	id = 0;
	gradeId = 0;
	slideId = 0;
	physId = -1;
}

node::node(const double& coordX, const double& coordY, const double& coordZ){

	spfOrder = 0;

	x = coordX;
	y = coordY;
	z = coordZ;

	frCoef = 0.0;
	dzMax = 0.0;
	slideDepth = 0.0;

	id = 0;
	gradeId = 0;
	slideId = 0;
	physId = -1;
}

node node::operator+(const node& other){
	return node(x + other.x, y + other.y, z + other.z);
}

node node::operator-(const node& other){
	return node(x - other.x, y - other.y, z - other.z);
}

void node::operator=(const node& other){

	if (this != &other) {

		facets = other.facets;
		edges = other.edges;

		spfOrder = other.spfOrder;

		x = other.x;
		y = other.y;
		z = other.z;

		frCoef = other.frCoef;
		dzMax = other.dzMax;
		slideDepth = other.slideDepth;

		id = other.id;
		gradeId = other.gradeId;
		slideId = other.slideId;
		physId = other.physId;
	}
}

double node::distXY(const node& other){
	return sqrt(pow(x - other.x, 2.0) + pow(y - other.y, 2.0));
}

double node::distXYZ(const node& other){
	return sqrt(pow(x - other.x, 2.0) + pow(y - other.y, 2.0) + pow(z - other.z, 2.0));
}

facet::facet(){

	nodes.resize(3, NULL);
	neighs.resize(3, NULL);

	spfOrder = 0;
	id = 0;
	physId = -1;
}

void facet::operator=(const facet& other){

	if (this != &other){

		nodes = other.nodes;
		edges = other.edges;
		neighs = other.neighs;

		center = other.center;
		spfOrder = other.spfOrder;

		id = other.id;
		physId = other.physId;
	}
}

void facet::setCcWise(){

	double pi = 3.14159265358979323846;
	std::vector<double> bisectorTan(nodes.size(), 0.0);

	for (unsigned k = 0; k < nodes.size(); ++k){
		bisectorTan[k] = atan2((*nodes[k] - center).y, (*nodes[k] - center).x);
		bisectorTan[k] = (bisectorTan[k] >= 0.0 ? bisectorTan[k] : (2.0 * pi + bisectorTan[k])) * 360.0 / (2.0 * pi);
	}

	if (bisectorTan[0] > bisectorTan[1])
		std::swap(nodes[0], nodes[1]);
	if (bisectorTan[0] > bisectorTan[2])
		std::swap(nodes[0], nodes[2]);
	if (bisectorTan[1] > bisectorTan[2])
		std::swap(nodes[1], nodes[2]);

	std::vector<edge*> edgesCopy = edges;

	for (unsigned n = 0; n < nodes.size(); ++n){
		unsigned nNext = (n + 1) % nodes.size();
		std::vector<node*> firstPair{ nodes[n], nodes[nNext] };
		for (unsigned k = 0; k < edges.size(); ++k){
			std::vector<node*> secondPair{ edgesCopy[k]->nodes[0], edgesCopy[k]->nodes[1] };
			if (std::is_permutation(firstPair.begin(), firstPair.end(), secondPair.begin()))
				edges[n] = edgesCopy[k];
		}
	}

	for (unsigned k = 0; k < edges.size(); ++k){
		if (edges[k]->facets[0] != this){
			if (edges[k]->facets[1] == this)
				neighs[k] = edges[k]->facets[0];
		}else if (edges[k]->facets[1] != this){
			if (edges[k]->facets[0] == this)
				neighs[k] = edges[k]->facets[1];
		}
	}

	edgesCopy.clear();
	bisectorTan.clear();
}

double facet::getArea(){

	double x[3] = { 0.0 };
	double y[3] = { 0.0 };
	double z[3] = { 0.0 };

	for (unsigned short n = 0; n < 3; ++n){
		x[n] = nodes[n]->x;
		y[n] = nodes[n]->y;
		z[n] = nodes[n]->z;
	}

	return std::abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1])) / 2.0;
}

double facet::getSkewness(){

	double pi = 3.14159265358979323846;
	double thetaMax = -2.0*pi;
	double thetaMin = 2.0*pi;

	for (unsigned k = 0; k < nodes.size(); ++k){
		
		unsigned kNext = k%nodes.size();

		double xA = edges[k]->nodes[1]->x - edges[k]->nodes[0]->x;
		double yA = edges[k]->nodes[1]->x - edges[k]->nodes[0]->x;
		double xB = edges[kNext]->nodes[1]->x - edges[kNext]->nodes[0]->x;;
		double yB = edges[kNext]->nodes[1]->x - edges[kNext]->nodes[0]->x;;

		double theta = acos(abs((xA*xB + yA*yB) / (edges[k]->getLength()*edges[kNext]->getLength())));
		theta = theta < 0.0 ? theta + 2.0*pi : theta;

		if (theta > thetaMax)
			thetaMax = theta;
		if (theta < thetaMin)
			thetaMin = theta;
	}

	return std::max((thetaMax - pi/3.0) / (pi*2.0/3.0), (pi/3.0 - thetaMin) / (pi/3.0));
}

double facet::getAspectRatio(){
	
	return 2.0*getArea() / getMaxEdgeLength();
}

double facet::getMinEdgeLength(){

	double minLength = 9999999999.0;
	for (unsigned k = 0; k < edges.size(); ++k){
		if (edges[k]->getLength() < minLength)
			minLength = edges[k]->getLength();
	}

	return minLength;
}

double facet::getMaxEdgeLength(){

	double maxLength = 0.0;
	for (unsigned k = 0; k < edges.size(); ++k){
		if (edges[k]->getLength() > maxLength)
			maxLength = edges[k]->getLength();
	}

	return maxLength;
}

edge::edge(){

	nodes.resize(2, NULL);
	facets.resize(2, NULL);

	id = 0;
	physId = -1;
}

void edge::operator=(const edge& other){

	if (this != &other) {

		nodes = other.nodes;
		facets = other.facets;

		id = other.id;
		physId = other.physId;
	}
}

double edge::getLength(){

	return nodes[0]->distXY(*nodes[1]);
}