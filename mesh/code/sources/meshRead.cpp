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
#include <vector>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshRead.hpp"
#include "../headers/meshGlobals.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


void simulationMesh::readMeshFile(){

	std::string gMeshFileName = geometry.gmshFolder + geometry.gmshMeshFile;

	meshRead<HalfedgeDS> rMesh(gMeshFileName, nodes, facets, physEdges);
	polyCGAL.delegate(rMesh);
	polyCGAL.normalize_border();

	std::cout << std::endl << std::endl;
	std::cout << "  Verifying mesh";

	if (polyCGAL.is_valid())
		std::cout << "  Imported mesh is valid. Continuing to Half-Edge/Winged-Edge conversion" << std::endl << std::endl;
	else{
		std::cout << "  Mesh is not valid" << std::endl << std::endl;
		std::cout << "  Please check that your .mesh file is using the INRIA MEDIT Version 2 format and ensure that only entities with PHYSICAL tags are saved." << std::endl;
		exitOnKeypress(1);
	}

	if (facets.size() != polyCGAL.size_of_facets() || nodes.size() != polyCGAL.size_of_vertices()){
		std::cout << "  Problem while importing mesh from " << gMeshFileName << "." << std::endl << std::endl;
		std::cout << "  Some elements were not imported to CGAL's HDS due to corrupt original connectivity or bad orientation ..." << std::endl;
		std::cout << "  Please check that your .mesh file is using the INRIA MEDIT Version 2 format and ensure that only entities with PHYSICAL tags are saved." << std::endl;
		exitOnKeypress(1);
	}
}

void simulationMesh::setConnectivity(){

	vertex_iter vertIter;
	facet_iter faceIter;
	edge_iter edgeIter;

	unsigned idx = 0;
	for (vertIter = polyCGAL.vertices_begin(); vertIter != polyCGAL.vertices_end(); ++vertIter){
		vertIter->id() = idx;

		showProgress(int(++idx), nodes.size(), "Indexing", "Vertexes");
	}

	if (idx != nodes.size()){
		std::cout << std::endl << std::endl << " Warning: Inconsistent vertex structure sizes ... " << std::endl << std::endl;
		exitOnKeypress(1);
	}

	std::cout << std::endl;
	edges.resize(polyCGAL.size_of_halfedges() / 2);

	idx = 0;
	for (edgeIter = polyCGAL.edges_begin(); edgeIter != polyCGAL.edges_end(); ++edgeIter){
		edges[idx].id = idx;
		edgeIter->id() = idx;
		edgeIter->opposite()->id() = (unsigned) edgeIter->id();

		showProgress(int(++idx), edges.size(), "Indexing", "Edges");
	}

	if (idx != edges.size()){
		std::cout << std::endl << std::endl << " Warning: Inconsistent Half-Edge <> Winged-Edge structure sizes ... " << std::endl << std::endl;
		exitOnKeypress(1);
	}

	std::cout << std::endl;

	idx = 0;
	for (faceIter = polyCGAL.facets_begin(); faceIter != polyCGAL.facets_end(); ++faceIter){
		faceIter->id() = idx;

		showProgress(int(++idx), facets.size(), "Indexing", "Facets");
	}

	if (idx != facets.size()){
		std::cout << std::endl << std::endl << " Warning: Inconsistent facet structure sizes ... " << std::endl << std::endl;
		exitOnKeypress(1);
	}

	std::cout << std::endl;

	idx = 0;
	for (edgeIter = polyCGAL.edges_begin(); edgeIter != polyCGAL.edges_end(); ++edgeIter){

		edges[edgeIter->id()].nodes[0] = &nodes[edgeIter->vertex()->id()];
		edges[edgeIter->id()].nodes[1] = &nodes[edgeIter->opposite()->vertex()->id()];

		nodes[edgeIter->opposite()->vertex()->id()].edges.push_back(&edges[edgeIter->id()]);
		nodes[edgeIter->vertex()->id()].edges.push_back(&edges[edgeIter->id()]);

		if (!edgeIter->is_border() && !edgeIter->opposite()->is_border()){

			edges[edgeIter->id()].facets[0] = &facets[edgeIter->facet()->id()];
			edges[edgeIter->id()].facets[1] = &facets[edgeIter->opposite()->facet()->id()];
			facets[edgeIter->facet()->id()].edges.push_back(&edges[edgeIter->id()]);
			facets[edgeIter->opposite()->facet()->id()].edges.push_back(&edges[edgeIter->id()]);
		
		}else if (!edgeIter->is_border()){

			edges[edgeIter->id()].facets[0] = &facets[edgeIter->facet()->id()];
			facets[edgeIter->facet()->id()].edges.push_back(&edges[edgeIter->id()]);
		
		}else if (!edgeIter->opposite()->is_border()){

			edges[edgeIter->id()].facets[1] = &facets[edgeIter->opposite()->facet()->id()];
			facets[edgeIter->opposite()->facet()->id()].edges.push_back(&edges[edgeIter->id()]);
		}

		showProgress(int(++idx), edges.size(), "Connecting", "Edges");
	}

	std::cout << std::endl;
}