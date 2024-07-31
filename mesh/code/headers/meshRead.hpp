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

// Pre-Processor Headers
#include "meshEntities.hpp"

// Boost (1.55)
#include <boost/algorithm/string/predicate.hpp>

// CGAL (4.8)
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>

/////////////////////////////////////////////////////////////////////////////////////////////////


typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Polyhedron_3 <K, CGAL::Polyhedron_items_with_id_3>  Polyhedron;

typedef Polyhedron::HalfedgeDS HalfedgeDS;
typedef Polyhedron::Vertex_iterator vertex_iter;
typedef Polyhedron::Edge_iterator edge_iter;
typedef Polyhedron::Facet_iterator facet_iter;

template <class HDS>
class meshRead : public CGAL::Modifier_base<HDS> {

	std::string meshFileName;
	std::vector<node> *nodesPtr;
	std::vector<facet> *facetsPtr;
	std::vector<edge> *physEgesPtr;

public:

	meshRead(const std::string &inputMeshFileName, std::vector<node>& meshNodes, std::vector<facet>& meshFacets, std::vector<edge>& meshPhysEdges){

		meshFileName = inputMeshFileName;
		nodesPtr = &meshNodes;
		facetsPtr = &meshFacets;
		physEgesPtr = &meshPhysEdges;

		std::string textBuffer;
		std::ifstream meshFile;

		int intBuffer;
		int numNodes;
		int numPhysEdges;
		int numElements;
		
		std::cout << std::endl << "  Opening file " << meshFileName << " ..." << std::endl << std::endl;
		meshFile.open(meshFileName);
		if (!meshFile.is_open() || !meshFile.good()){
			std::cout << "Problem while accessing " << meshFileName << "." << std::endl;
			std::cout << "Please check that your .mesh file is using the INRIA MEDIT Version 2 format and ensure that only entities with PHYSICAL tags are saved." << std::endl;
			exitOnKeypress(1);
		}

		getline(meshFile, textBuffer);
		getline(meshFile, textBuffer);
		getline(meshFile, textBuffer);

		getline(meshFile, textBuffer);
		if (textBuffer != " Vertices"){
			std::cout << "Problem while reading vertex section in " << meshFileName << "." << std::endl;
			std::cout << "Please check that your .mesh file is using the INRIA MEDIT Version 2 format and ensure that only entities with PHYSICAL tags are saved." << std::endl;
			exitOnKeypress(1);
		}

		meshFile >> numNodes;
		(*nodesPtr).resize(numNodes);

		for (unsigned n = 0; n < (*nodesPtr).size(); ++n){
			meshFile >> (*nodesPtr)[n].x >> (*nodesPtr)[n].y >> (*nodesPtr)[n].z >> textBuffer;
			(*nodesPtr)[n].id = n;
			showProgress(n, numNodes, "Reading", "Vertices");
		}

		std::cout << std::endl;

		getline(meshFile, textBuffer);
		getline(meshFile, textBuffer);

		if (textBuffer == " Edges"){

			meshFile >> numPhysEdges;
			(*physEgesPtr).resize(numPhysEdges);

			for (unsigned k = 0; k < (*physEgesPtr).size(); ++k){
				
				meshFile >> intBuffer;
				(*physEgesPtr)[k].nodes[0] = &(*nodesPtr)[intBuffer - 1];

				meshFile >> intBuffer;
				(*physEgesPtr)[k].nodes[1] = &(*nodesPtr)[intBuffer - 1];

				meshFile >> (*physEgesPtr)[k].physId;
				(*physEgesPtr)[k].nodes[0]->physId = ((*physEgesPtr)[k].nodes[0]->physId > (*physEgesPtr)[k].physId) ? (*physEgesPtr)[k].nodes[0]->physId : (*physEgesPtr)[k].physId;
				(*physEgesPtr)[k].nodes[1]->physId = ((*physEgesPtr)[k].nodes[1]->physId > (*physEgesPtr)[k].physId) ? (*physEgesPtr)[k].nodes[1]->physId : (*physEgesPtr)[k].physId;

				showProgress(k, numPhysEdges, "Reading", "Phys. Edges");
			}

			std::cout << std::endl;

			getline(meshFile, textBuffer);
			getline(meshFile, textBuffer);
		}

		if (textBuffer == " Triangles"){

			meshFile >> numElements;
			(*facetsPtr).resize(numElements);

			for (unsigned i = 0; i < (*facetsPtr).size(); ++i){
				
				for (unsigned n = 0; n < (*facetsPtr)[i].nodes.size(); ++n){
					meshFile >> intBuffer;
					(*facetsPtr)[i].nodes[n] = &(*nodesPtr)[intBuffer - 1];
				}

				meshFile >> intBuffer;
				(*facetsPtr)[i].id = i;

				(*facetsPtr)[i].center.x = (1.0 / 3.0)*((*facetsPtr)[i].nodes[0]->x + (*facetsPtr)[i].nodes[1]->x + (*facetsPtr)[i].nodes[2]->x);
				(*facetsPtr)[i].center.y = (1.0 / 3.0)*((*facetsPtr)[i].nodes[0]->y + (*facetsPtr)[i].nodes[1]->y + (*facetsPtr)[i].nodes[2]->y);

				for (unsigned n = 0; n < (*facetsPtr)[i].nodes.size(); ++n)
					if ((*facetsPtr)[i].nodes[n]->physId != 0)
						(*facetsPtr)[i].physId = (*facetsPtr)[i].nodes[n]->physId;

				showProgress(i, numElements, "Reading", "Faces");
			}

			std::cout << std::endl << std::endl;

			getline(meshFile, textBuffer);
			getline(meshFile, textBuffer);
			if (textBuffer != " End"){
				std::cout << "Problem while reading end of file in " << meshFileName << "." << std::endl;
				std::cout << "Please check that your .mesh file is using the INRIA MEDIT Version 2 format and ensure that only entities with PHYSICAL tags are saved." << std::endl;
				exitOnKeypress(1);
			}
		}
		else{
			std::cout << "Problem while reading headers in  " << meshFileName << "." << std::endl;
			std::cout << "Please check that your .mesh file is using the INRIA MEDIT Version 2 format and ensure that only entities with PHYSICAL tags are saved." << std::endl;
			exitOnKeypress(1);
		}
	}

	void operator()(HDS& hds){
		
		CGAL::Polyhedron_incremental_builder_3<HDS> mesh(hds, true);
		mesh.begin_surface((*nodesPtr).size(), (*facetsPtr).size(), 0);

		typedef typename HDS::Vertex::Point Point;

		for (unsigned i = 0; i < (*nodesPtr).size(); i++){
			mesh.add_vertex(Point((*nodesPtr)[i].x, (*nodesPtr)[i].y, 0.0));
			showProgress(i, (*nodesPtr).size(), "Adding", "Vertexes");
		}

		std::cout << std::endl;
		std::vector<unsigned> facetToAdd(3);
		//std::vector<unsigned> facetsToRemove;

		for (unsigned i = 0; i < (*facetsPtr).size(); i++){

			for (unsigned n = 0; n < 3; ++n)
				facetToAdd[n] = (*facetsPtr)[i].nodes[n]->id;

			if (mesh.test_facet(facetToAdd.begin(), facetToAdd.end())){
				mesh.begin_facet();
				for (unsigned n = 0; n < 3; ++n)
					mesh.add_vertex_to_facet((*facetsPtr)[i].nodes[n]->id);
				mesh.end_facet();
			}else{
				std::cout << "Warning: element index " << i << " is not valid (possibly collapsed)!" << std::endl;
				//facetsToRemove.push_back(i);
			}
			showProgress(i, (*facetsPtr).size(), "Adding", "Facets");
		}

		//for (unsigned i = 0; i < facetsToRemove.size(); i++){
			//(*facetsPtr).erase((*facetsPtr).begin() + i);
		//}

		mesh.end_surface();
	}
};