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
#include <cstdint>
#include <utility>
#include <vector>
#include <algorithm>
#include <cmath>

// Boost
#include <boost/config.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/cuthill_mckee_ordering.hpp>
#include <boost/graph/properties.hpp>
#include <boost/graph/bandwidth.hpp>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshEntities.hpp"
#include "../headers/meshGlobals.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


uint64_t getZOrder(const double& coordX, const double& coordY){

	uint64_t zOrder = 0;
	uint32_t x = (uint32_t)(std::round(coordX));
	uint32_t y = (uint32_t)(std::round(coordY));

	for (unsigned i = 0; i < sizeof(uint32_t) * 8; ++i)
		zOrder |= (x & (uint64_t)1 << i) << i | (y & (uint64_t)1 << i) << (i + 1);

	return zOrder;
}

void node::setSpfOrder(){

	if (true)
		spfOrder = getZOrder((x - mesh.lowerLeft.x) / mesh.avgLength, (y - mesh.lowerLeft.y) / mesh.avgLength);
}

void facet::setSpfOrder(){

	if (true)
		spfOrder = getZOrder((center.x - mesh.lowerLeft.x) / mesh.avgLength, (center.y - mesh.lowerLeft.y) / mesh.avgLength);
}

void simulationMesh::applySpfOrderingToNodes(){

	for (unsigned i = 0; i < facets.size(); ++i)
		for (unsigned n = 0; n < facets[i].nodes.size(); ++n)
			facets[i].nodes[n]->facets.push_back(&facets[i]);

	for (unsigned n = 0; n < nodes.size(); ++n)
		nodes[n].setSpfOrder();

	std::vector< std::pair<uint64_t, unsigned> > spfOrderPairs(nodes.size());

	for (unsigned n = 0; n < nodes.size(); n++){
		spfOrderPairs[n].first = (unsigned)nodes[n].spfOrder;
		spfOrderPairs[n].second = nodes[n].id;

		showProgress((int)n, nodes.size() * 4, "Ordering", "Vertex ");
	}

	std::vector<node> nodesCopy = nodes;

	for (unsigned n = 0; n < nodes.size(); ++n){

		for (unsigned j = 0; j < nodes[n].facets.size(); ++j)
			for (unsigned k = 0; k < nodes[n].facets[j]->nodes.size(); ++k)
				if (nodes[n].facets[j]->nodes[k] == &nodes[n]){
					nodes[n].facets[j]->nodes[k] = &nodesCopy[n];
					break;
				}

		for (unsigned j = 0; j < nodes[n].edges.size(); ++j)
			for (unsigned k = 0; k < nodes[n].edges[j]->nodes.size(); ++k)
				if (nodes[n].edges[j]->nodes[k] == &nodes[n]){
					nodes[n].edges[j]->nodes[k] = &nodesCopy[n];
					break;
				}

		showProgress((int)nodes.size() + n, nodes.size() * 4, "Copying", "Vertex  ");
	}

	std::sort(spfOrderPairs.begin(), spfOrderPairs.end());

	for (unsigned k = 0; k < physEdges.size(); ++k)
		for (unsigned n = 0; n < physEdges[k].nodes.size(); ++n){
			unsigned origIdx = physEdges[k].nodes[n]->id;
			std::vector< std::pair<uint64_t, unsigned> >::iterator pairIter;
			pairIter = std::find_if(spfOrderPairs.begin(), spfOrderPairs.end(),
				[&origIdx](const std::pair<uint64_t, unsigned>& findPair){ return findPair.second == origIdx; });
			unsigned newIdx = std::distance(spfOrderPairs.begin(), pairIter);
			physEdges[k].nodes[n] = &nodes[newIdx];
		}

	for (unsigned n = 0; n < nodes.size(); ++n){
		unsigned origIdx = spfOrderPairs[n].second;
		nodes[n] = nodesCopy[origIdx];
		nodes[n].id = n;
		nodes[n].edges.clear();
		nodes[n].facets.clear();

		showProgress((int)nodes.size() * 2 + n, nodes.size() * 4, "Sorting", "Vertex  ");
	}

	for (unsigned n = 0; n < nodesCopy.size(); ++n){

		unsigned origIdx = spfOrderPairs[n].second;

		for (unsigned j = 0; j < nodesCopy[origIdx].facets.size(); ++j)
			for (unsigned k = 0; k < nodesCopy[origIdx].facets[j]->nodes.size(); ++k)
				if (nodesCopy[origIdx].facets[j]->nodes[k] == &nodesCopy[origIdx]){
					nodesCopy[origIdx].facets[j]->nodes[k] = &nodes[n];
					break;
				}

		for (unsigned j = 0; j < nodesCopy[origIdx].edges.size(); ++j)
			for (unsigned k = 0; k < nodesCopy[origIdx].edges[j]->nodes.size(); ++k)
				if (nodesCopy[origIdx].edges[j]->nodes[k] == &nodesCopy[origIdx]){
					nodesCopy[origIdx].edges[j]->nodes[k] = &nodes[n];
					break;
				}

		showProgress((int)nodes.size() * 3 + n, nodes.size() * 4, "Restoring", "Vertex");
	}

	spfOrderPairs.clear();
	nodesCopy.clear();

	std::cout << std::endl;
}

void simulationMesh::applySpfOrderingToFacets(){

	for (unsigned i = 0; i < facets.size(); ++i)
		facets[i].setSpfOrder();

	int progressCounter = 0;
	int progressGoal = int(facets.size()*4);

	/*typedef boost::adjacency_list < boost::vecS, boost::vecS, boost::undirectedS, boost::property < boost::vertex_color_t, boost::default_color_type, boost::property<boost::vertex_degree_t, int > > > boostGraph;
	typedef boost::graph_traits <boostGraph>::vertex_descriptor boostVertex;
	typedef boost::graph_traits <boostGraph>::vertices_size_type boostSize;

	boostGraph meshGraph(facets.size());

	for (int k = 0; k < edges.size(); ++k){

		std::size_t leftIdx = 0;
		std::size_t rightIdx = 0;

		if (edges[k].facets[0])
			leftIdx = std::size_t(edges[k].facets[0]->id);
		else
			leftIdx = std::size_t(edges[k].facets[1]->id);

		if (edges[k].facets[1])
			rightIdx = std::size_t(edges[k].facets[1]->id);
		else
			rightIdx = std::size_t(edges[k].facets[0]->id);

		std::pair <std::size_t, std::size_t> edgeToAdd = std::make_pair(leftIdx, rightIdx);
		boost::add_edge(edgeToAdd.first, edgeToAdd.second, meshGraph);

		showProgress(progressCounter++, progressGoal, "Building", "Graph ");
	}

	boost::graph_traits<boostGraph>::vertex_iterator ui, ui_end;
	boost::property_map <boostGraph, boost::vertex_index_t>::type index_map = boost::get(boost::vertex_index, meshGraph);
	boost::property_map <boostGraph, boost::vertex_degree_t>::type deg = boost::get(boost::vertex_degree, meshGraph);
	for (boost::tie(ui, ui_end) = boost::vertices(meshGraph); ui != ui_end; ++ui)
		deg[*ui] = boost::degree(*ui, meshGraph);

	std::vector <boostSize> perm(boost::num_vertices(meshGraph));
	std::vector <boostVertex> invPerm(boost::num_vertices(meshGraph));

	double sumXCoord = 0.0;
	double sumYCoord = 0.0;
	for (int i = 0; i < facets.size(); ++i){
		sumXCoord += facets[i].center.x;
		sumYCoord += facets[i].center.y;
	}

	node originPoint(lowerLeft.x, sumYCoord / double(facets.size()), 0.0);
	originPoint = upperRight;

	int closestIdx = 0;
	double distToOrigin = 999999999.0;
	for (int i = 0; i < facets.size(); ++i)
		if (facets[i].center.distXY(originPoint) < distToOrigin){
			distToOrigin = facets[i].center.distXY(originPoint);
			closestIdx = i;
		}
	
	boostVertex start = boost::vertex(closestIdx, meshGraph);
	boost::cuthill_mckee_ordering(meshGraph, start, invPerm.rbegin(), boost::get(boost::vertex_color, meshGraph), boost::get(boost::vertex_degree, meshGraph));

	for (boostSize i = 0; i != invPerm.size(); ++i)
		perm[index_map[invPerm[i]]] = i;

	for (unsigned i = 0; i < facets.size(); ++i){
		facets[i].spfOrder = unsigned(perm[i]);*/
	
	std::vector <std::pair <uint64_t, unsigned> > spfOrderPairs(facets.size());

	for (unsigned i = 0; i < facets.size(); ++i){
		spfOrderPairs[i].first = facets[i].spfOrder;
		spfOrderPairs[i].second = facets[i].id;
		showProgress(progressCounter++, progressGoal, "Ordering", "Facets ");
	}

	/*meshGraph.clear();
	perm.clear();
	invPerm.clear();*/

	std::vector<facet> facetsCopy = facets;

	for (unsigned i = 0; i < facets.size(); ++i){
		for (unsigned k = 0; k < facets[i].edges.size(); ++k)
			for (unsigned j = 0; j < facets[i].edges[j]->facets.size(); ++j)
				if (facets[i].edges[k]->facets[j] == &facets[i]){
					facets[i].edges[k]->facets[j] = &facetsCopy[i];
					break;
				}
		showProgress(progressCounter++, progressGoal, "Copying", "Facets  ");
	}

	std::sort(spfOrderPairs.begin(), spfOrderPairs.end());

	for (unsigned i = 0; i < facets.size(); ++i){
		unsigned origIdx = spfOrderPairs[i].second;
		facets[i] = facetsCopy[origIdx];
		facets[i].id = i;
		showProgress(progressCounter++, progressGoal, "Sorting", "Facets  ");
	}

	for (unsigned i = 0; i < facetsCopy.size(); ++i){
		unsigned origIdx = spfOrderPairs[i].second;
		for (unsigned k = 0; k < facetsCopy[origIdx].edges.size(); ++k)
			for (unsigned j = 0; j < facetsCopy[origIdx].edges[k]->facets.size(); ++j)
				if (facetsCopy[origIdx].edges[k]->facets[j] == &facetsCopy[origIdx]){
					facetsCopy[origIdx].edges[k]->facets[j] = &facets[i];
					break;
				}
		showProgress(progressCounter++, progressGoal, "Restoring", "Facets");
	}

	spfOrderPairs.clear();
	facetsCopy.clear();

	std::cout << std::endl;
}