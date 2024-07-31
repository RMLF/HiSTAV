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
#include <string>
#include <fstream>
#include <algorithm>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshEntities.hpp"
#include "../headers/meshGlobals.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


void simulationMesh::writeMeshInfo() {

	std::string meshDimInfoFileName = control.targetFolder + control.meshFolder + control.meshDimFileName;

	std::ofstream meshDimInfo;
	meshDimInfo.open(meshDimInfoFileName);
	meshDimInfo << nodes.size() << std::endl;
	meshDimInfo << facets.size() << std::endl;
	meshDimInfo.close();

	toLog("Mesh dimensions file written", 1);

	std::string bedDataInfoFileName = control.targetFolder + control.bedDataFolder + control.bedLayersFileName;
	
	std::ofstream bedDataInfo;
	bedDataInfo.open(bedDataInfoFileName);
	
	if (control.hasFrictionCoef)
		bedDataInfo << "FricCfMap" << std::endl << control.frCoefNodesFileName << std::endl;

    if (control.hasInfiltrationCoef)
        bedDataInfo << "InfiltCfMap" << std::endl << control.infiltCoefNodesFileName << std::endl;

	if (control.hasBedrockOffset)
		bedDataInfo << "BedrockMap" << std::endl << control.bedrockOffsetNodesFileName << std::endl;

	if (control.hasGradeId)
		bedDataInfo << "GradeIdMap" << std::endl << control.gradeIdNodesFileName << std::endl;

	if (control.hasLandslides)
		bedDataInfo << "Landslides" << std::endl << control.landslidesNodesFileName << std::endl;

	if (control.hasLandslidesId)
		bedDataInfo << "LandslidesID" << std::endl << control.landslidesIdNodesFileName << std::endl;

	bedDataInfo.close();

	toLog("Active sediment layers file written", 1);

	std::string boundaryInfoFileName = control.targetFolder + control.boundaryDataFolder + control.boundaryDimFileName;

	std::ofstream boundaryInfo;
	boundaryInfo.open(boundaryInfoFileName);
	
	boundaryInfo << boundaries.size() << std::endl;
	for (unsigned b = 0; b < boundaries.size(); b++){
		boundaryInfo << std::endl << b << "	" << boundaries[b].seriesFile.size() << "	" << boundaries[b].meshFacets.size() << std::endl;
		for (unsigned p = 0; p < boundaries[b].seriesFile.size(); p++)
			boundaryInfo << p << "	" << boundaries[b].seriesFile[p] << std::endl;
	}

	boundaryInfo.close();

	toLog("Boundary dimensions and series file written", 1);
}

void simulationMesh::writeMeshNodes() {

	std::ofstream nodesOutputFile;
	nodesOutputFile.open(control.targetFolder + control.meshFolder + control.nodesFileName);
	for (unsigned n = 0; n < nodes.size(); ++n){
		nodesOutputFile << std::fixed << nodes[n].x << "	" << std::fixed << nodes[n].y << "	" << std::fixed << nodes[n].z << std::endl;
		showProgress(int(n), nodes.size(), "Writing", "Nodes");
	}
	std::cout << std::endl;
	nodesOutputFile.close();

	if (control.hasFrictionCoef){
		nodesOutputFile.open(control.targetFolder + control.bedDataFolder + control.frCoefNodesFileName);
		for (unsigned n = 0; n < nodes.size(); ++n){
			nodesOutputFile << std::fixed << nodes[n].frCoef << std::endl;
			showProgress((int)n, nodes.size(), "Writing", "FrictionMap");
		}
		nodesOutputFile.close();
		std::cout << std::endl;
	}

    if (control.hasInfiltrationCoef){
        nodesOutputFile.open(control.targetFolder + control.bedDataFolder + control.infiltCoefNodesFileName);
        for (unsigned n = 0; n < nodes.size(); ++n){
            nodesOutputFile << std::fixed << nodes[n].permeability << std::endl;
            showProgress((int)n, nodes.size(), "Writing", "Permeability");
        }
        nodesOutputFile.close();
        std::cout << std::endl;
    }

	if (control.hasBedrockOffset){
		nodesOutputFile.open(control.targetFolder + control.bedDataFolder + control.bedrockOffsetNodesFileName);
		for (unsigned n = 0; n < nodes.size(); ++n){
			nodesOutputFile << std::fixed << nodes[n].dzMax << std::endl;
			showProgress((int)n, nodes.size(), "Writing", "BedrockMap");
		}
		nodesOutputFile.close();
		std::cout << std::endl;
	}

	if (control.hasGradeId){
		nodesOutputFile.open(control.targetFolder + control.bedDataFolder + control.gradeIdNodesFileName);
		for (unsigned n = 0; n < nodes.size(); ++n){
			nodesOutputFile << std::fixed << nodes[n].gradeId << std::endl;
			showProgress((int)n, nodes.size(), "Writing", "GradeIdMap");
		}
		nodesOutputFile.close();
		std::cout << std::endl;
	}

	if (control.hasLandslides){
		nodesOutputFile.open(control.targetFolder + control.bedDataFolder + control.landslidesNodesFileName);
		for (unsigned n = 0; n < nodes.size(); ++n){
			nodesOutputFile << std::fixed << nodes[n].slideDepth << std::endl;
			showProgress((int)n, nodes.size(), "Writing", "LnSlideMap");
		}
		nodesOutputFile.close();
		std::cout << std::endl;
	}

	if (control.hasLandslidesId){
		nodesOutputFile.open(control.targetFolder + control.bedDataFolder + control.landslidesIdNodesFileName);
		for (unsigned n = 0; n < nodes.size(); ++n){
			nodesOutputFile << nodes[n].slideId << std::endl;
			showProgress((int)n, nodes.size(), "Writing", "LnSliIDMap");
		}
		nodesOutputFile.close();
		std::cout << std::endl;
	}
}

void simulationMesh::writeMeshFacets() {

	std::ofstream elementsOutputFile;
	elementsOutputFile.open(control.targetFolder + control.meshFolder + control.elementsFileName);

	for (unsigned i = 0; i < facets.size(); ++i){

		for (unsigned k = 0; k < facets[i].nodes.size(); ++k)
			elementsOutputFile << facets[i].nodes[k]->id << "	";

		for (unsigned k = 0; k < facets[i].edges.size(); ++k){
			if (facets[i].neighs[k] != NULL)
				elementsOutputFile << facets[i].neighs[k]->id;
			else
				elementsOutputFile << -1;

			if (k != facets[i].edges.size() - 1)
				elementsOutputFile << "	";
		}

		elementsOutputFile << std::endl;
		showProgress(int(i), facets.size(), "Writing", "Facets");
	}

	std::cout << std::endl;
	elementsOutputFile.close();
}

void simulationMesh::writeMeshBoundaries() {

	std::vector<node*> closestNodeA;
	std::vector<node*> closestNodeB;

	std::vector<double> weightNodeA;
	std::vector<double> weightNodeB;

	for (unsigned b = 0; b < boundaries.size(); ++b){

		unsigned assignedCounter = 0;

		closestNodeA.clear();
		closestNodeB.clear();
		weightNodeA.clear();
		weightNodeB.clear();

		closestNodeA.resize(boundaries[b].meshEdges.size());
		closestNodeB.resize(boundaries[b].meshEdges.size());
		weightNodeA.resize(boundaries[b].meshEdges.size(), 0.0);
		weightNodeB.resize(boundaries[b].meshEdges.size(), 0.0);

		for (unsigned k = 0; k < boundaries[b].meshEdges.size(); ++k){

			node* bndEdgePointA = boundaries[b].meshEdges[k]->nodes[0];
			node* bndEdgePointB = boundaries[b].meshEdges[k]->nodes[1];

			node edgeMidpoint((0.5*(bndEdgePointA->x + bndEdgePointB->x)), (0.5*(bndEdgePointA->y + bndEdgePointB->y)), 0.0);

			double tolerance = 1.0e-3;

			for (unsigned p = 0; p < boundaries[b].nodes.size() - 1; ++p){

				node* bndPointA = &boundaries[b].nodes[p];
				node* bndPointB = &boundaries[b].nodes[p + 1];

				if (edgeMidpoint.distXY(*bndPointA) + edgeMidpoint.distXY(*bndPointB) - bndPointA->distXY(*bndPointB) <= tolerance){
					closestNodeA[k] = &boundaries[b].nodes[p];
					closestNodeB[k] = &boundaries[b].nodes[p + 1];
					weightNodeA[k] = bndPointA->distXY(edgeMidpoint) / bndPointA->distXY(*bndPointB);
					weightNodeB[k] = bndPointB->distXY(edgeMidpoint) / bndPointA->distXY(*bndPointB);
					assignedCounter++;
				}
			}
		}

		std::stringstream idTo;	idTo << b;
		std::string boundaryTotalFileName = control.targetFolder + control.boundaryDataFolder + control.boundaryFileName + "-" + idTo.str() + ".bnd";
		std::ofstream boundaryIndexFile;

		boundaryIndexFile.open(boundaryTotalFileName);
		for (unsigned i = 0; i < boundaries[b].meshFacets.size(); ++i){
			for (unsigned k = 0; k < boundaries[b].meshFacets[i]->edges.size(); ++k)
				if (boundaries[b].meshFacets[i]->edges[k]->physId != -1)
					if (!boundaries[b].meshFacets[i]->neighs[k]){
						boundaryIndexFile << boundaries[b].meshFacets[i]->id << "	" << k << "	";
						break;
					}

			boundaryIndexFile << closestNodeA[i]->id << "	" << closestNodeB[i]->id << "	";
			boundaryIndexFile << std::fixed << weightNodeA[i] << "	" << std::fixed << weightNodeB[i] << std::endl;
		}

		boundaryIndexFile.close();
		showProgress(int(b), boundaries.size(), "Writing", "Boundaries");
	}

	std::cout << std::endl;
}

void simulationMesh::writeMeshQualityVTK() {

	int progressCounter = 0;
	
	unsigned multiplier = 6;
	if (control.hasBedrockOffset)
		multiplier++;
	if (control.hasFrictionCoef)
		multiplier++;
    if (control.hasInfiltrationCoef)
        multiplier++;
	if (control.hasGradeId)
		multiplier++;
	if (control.hasLandslidesId)
		multiplier++;
	if (control.hasLandslides)
		multiplier++;

	int progressGoal = int(facets.size()*multiplier + nodes.size()*2);

	std::ofstream vtkOutputFile;
	std::string vtkOutputFileName = control.targetFolder + control.meshFolder + control.outputVTKFileName;

	vtkOutputFile.open(vtkOutputFileName.c_str());

	vtkOutputFile << "# vtk DataFile Version 2.0" << std::endl;
	vtkOutputFile << "STAV-2D Mesh Pre-Processing" << std::endl;
	vtkOutputFile << "ASCII" << std::endl;
	vtkOutputFile << "DATASET UNSTRUCTURED_GRID" << std::endl;

	vtkOutputFile << std::endl;

	vtkOutputFile << "POINTS " << nodes.size() << " FLOAT" << std::endl;

	for (unsigned i = 0; i < nodes.size(); i++){
		vtkOutputFile << std::fixed << nodes[i].x << "	" << std::fixed << nodes[i].y << "	" << std::fixed << 0 << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	vtkOutputFile << "CELLS " << facets.size() << "	" << facets.size() * 4 << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkOutputFile << 3 << "	" << facets[i].nodes[0]->id << "	" << facets[i].nodes[1]->id << "	" << facets[i].nodes[2]->id << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	vtkOutputFile << "CELL_TYPES " << facets.size() << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkOutputFile << 5 << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	vtkOutputFile << "POINT_DATA " << nodes.size() << std::endl;
	vtkOutputFile << "SCALARS Address INT" << std::endl;
	vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkOutputFile << nodes[n].id << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	vtkOutputFile << "CELL_DATA " << facets.size() << std::endl;
	vtkOutputFile << "SCALARS Zbed FLOAT" << std::endl;
	vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkOutputFile << std::fixed << (1.0 / 3.0)*(facets[i].nodes[0]->z + facets[i].nodes[1]->z + facets[i].nodes[2]->z) << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	vtkOutputFile << "SCALARS Address INT" << std::endl;
	vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkOutputFile << facets[i].id << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	if (control.hasBedrockOffset){
		vtkOutputFile << "SCALARS dZ%20Max FLOAT" << std::endl;
		vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < facets.size(); ++i){
			vtkOutputFile << std::fixed << (1.0 / 3.0)*(facets[i].nodes[0]->dzMax + facets[i].nodes[1]->dzMax + facets[i].nodes[2]->dzMax) << std::endl;
			showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
		}
		vtkOutputFile << std::endl;
	}

	if (control.hasFrictionCoef){
		vtkOutputFile << "SCALARS Friction%20Coef FLOAT" << std::endl;
		vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < facets.size(); ++i){
			vtkOutputFile << std::fixed << (1.0 / 3.0)*(facets[i].nodes[0]->frCoef + facets[i].nodes[1]->frCoef + facets[i].nodes[2]->frCoef) << std::endl;
			showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
		}
		vtkOutputFile << std::endl;
	}

    if (control.hasInfiltrationCoef){
        vtkOutputFile << "SCALARS Infiltration%20Coef FLOAT" << std::endl;
        vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
        for (unsigned i = 0; i < facets.size(); ++i){
            vtkOutputFile << std::fixed << (1.0 / 3.0)*(facets[i].nodes[0]->permeability + facets[i].nodes[1]->permeability + facets[i].nodes[2]->permeability) << std::endl;
            showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
        }
        vtkOutputFile << std::endl;
    }

	if (control.hasGradeId){
		vtkOutputFile << "SCALARS Grade%20ID INT" << std::endl;
		vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < facets.size(); ++i){
			vtkOutputFile << (unsigned)round((1.0 / 3.0)*(facets[i].nodes[0]->gradeId + facets[i].nodes[1]->gradeId + facets[i].nodes[2]->gradeId)) << std::endl;
			showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
		}
		vtkOutputFile << std::endl;
	}

	vtkOutputFile << std::endl;

	vtkOutputFile << "SCALARS Max%20Length FLOAT" << std::endl;
	vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkOutputFile << std::fixed << facets[i].getMaxEdgeLength() << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	if (control.hasLandslidesId){
		vtkOutputFile << "SCALARS Landslide%20ID INT" << std::endl;
		vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < facets.size(); ++i){
			vtkOutputFile << (unsigned)round((1.0 / 3.0)*(facets[i].nodes[0]->slideId + facets[i].nodes[1]->slideId + facets[i].nodes[2]->slideId)) << std::endl;
			showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
		}
		vtkOutputFile << std::endl;
	}

	if (control.hasLandslides){
		vtkOutputFile << "SCALARS Landslide%20Depth FLOAT" << std::endl;
		vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
		for (unsigned i = 0; i < facets.size(); ++i){
			vtkOutputFile << std::fixed << (1.0 / 3.0)*(facets[i].nodes[0]->slideDepth + facets[i].nodes[1]->slideDepth + facets[i].nodes[2]->slideDepth) << std::endl;
			showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
		}
		vtkOutputFile << std::endl;
	}

	vtkOutputFile << "SCALARS Physical%20Tag INT" << std::endl;
	vtkOutputFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkOutputFile << facets[i].physId << std::endl;
		showProgress(progressCounter++, progressGoal, "Writing", "VTK File");
	}

	vtkOutputFile << std::endl;

	vtkOutputFile.close();
	std::cout << std::endl;
}

void simulationMesh::writeMeshDebugVTK() {

	std::string vtkDebugFileName = control.targetFolder + control.meshFolder + control.facetsDebugVTKFileName;

	std::ofstream vtkDebugFile;
	vtkDebugFile.open(vtkDebugFileName);

	vtkDebugFile << "# vtk DataFile Version 2.0" << std::endl;
	vtkDebugFile << "STAV-2D Mesh Pre-Processing" << std::endl;
	vtkDebugFile << "ASCII" << std::endl;
	vtkDebugFile << "DATASET UNSTRUCTURED_GRID" << std::endl;
	vtkDebugFile << std::endl;

	vtkDebugFile << "POINTS " << nodes.size() << " FLOAT" << std::endl;

	unsigned writeCounter = 0;
	unsigned writeTotal = unsigned(nodes.size()*3) + unsigned(facets.size()*5);

	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkDebugFile << std::fixed << nodes[n].x << "	" << std::fixed << nodes[n].y << "	" << std::fixed << 0.0 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "CELLS " << facets.size() << "	" << facets.size() * 4 << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << 3 << "	" << facets[i].nodes[0]->id << "	" << facets[i].nodes[1]->id << "	" << facets[i].nodes[2]->id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "VTK File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "CELL_TYPES " << facets.size() << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << 5 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "VTK File");
	}

	vtkDebugFile << "POINT_DATA " << nodes.size() << std::endl;
	vtkDebugFile << "SCALARS Address INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkDebugFile << std::fixed << nodes[n].id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "SCALARS Z%20Order INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkDebugFile << std::fixed << nodes[n].spfOrder << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "CELL_DATA " << facets.size() << std::endl;
	vtkDebugFile << "SCALARS Address INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << std::fixed << facets[i].id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "SCALARS Z%20Order INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << std::fixed << facets[i].spfOrder << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "SCALARS Verified INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;
	for (unsigned i = 0; i < facets.size(); ++i){

		bool verified = true;
		bool selfMatched = false;

		std::vector<bool> edgeNodesMatched(3, false);
		std::vector<bool> neighbourMatched(3, false);
		std::vector<bool> facetNodesMatched(3, false);

		for (unsigned k = 0; k < facets[i].edges.size(); ++k){

			std::vector<node*> firstPair{ facets[i].edges[k]->nodes[0], facets[i].edges[k]->nodes[1] };
			for (unsigned n = 0; n < facets[i].nodes.size(); ++n){
				unsigned nNext = (n + 1) % facets[i].nodes.size();
				std::vector<node*> secondPair{ facets[i].nodes[n], facets[i].nodes[nNext] };
				if (std::is_permutation(firstPair.begin(), firstPair.end(), secondPair.begin()))
					edgeNodesMatched[k] = true;
			}

			for (unsigned j = 0; j < facets[i].edges[k]->facets.size(); ++j)
				if (facets[i].edges[k]->facets[j] != NULL){
					if (facets[i].edges[k]->facets[j] != &facets[i])
						if (facets[i].neighs[k] == facets[i].edges[k]->facets[j])
							neighbourMatched[k] = true;
				}
				else
					neighbourMatched[k] = true;

			for (unsigned j = 0; j < facets[i].edges[k]->facets.size(); ++j)
				if (facets[i].edges[k]->facets[j] == &facets[i])
					selfMatched = true;
		}

		for (unsigned j = 0; j < facets[i].neighs.size(); ++j)
			if (facets[i].neighs[j] != NULL){
				if (facets[i].neighs[j] != &facets[i])
					for (unsigned n = 0; n < facets[i].nodes.size(); ++n){
						unsigned nNext = (n + 1) % facets[i].nodes.size();
						std::vector<node*> firstPair{ facets[i].nodes[n], facets[i].nodes[nNext] };
						for (unsigned nj = 0; nj < facets[i].neighs[j]->nodes.size(); ++nj){
							unsigned njNext = (nj + 1) % facets[i].neighs[j]->nodes.size();
							std::vector<node*> secondPair{ facets[i].neighs[j]->nodes[nj], facets[i].neighs[j]->nodes[njNext] };
							if (std::is_permutation(firstPair.begin(), firstPair.end(), secondPair.begin()))
								facetNodesMatched[j] = true;
						}
					}
			}else
				facetNodesMatched[j] = true;

		verified = verified && selfMatched;

		for (unsigned bl = 0; bl < 3; ++bl)
			verified = verified && edgeNodesMatched[bl] && facetNodesMatched[bl] && neighbourMatched[bl];

		vtkDebugFile << std::fixed << verified << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	std::cout << std::endl;

	vtkDebugFile.close();

	writeCounter = 0;
	writeTotal = unsigned(nodes.size()*3) + unsigned(facets.size()*3) + unsigned(edges.size()*8);

	vtkDebugFileName = control.targetFolder + control.meshFolder + control.edgesDebugVTKFileName;
	vtkDebugFile.open(vtkDebugFileName);

	vtkDebugFile << "# vtk DataFile Version 2.0" << std::endl;
	vtkDebugFile << "STAV-2D Mesh Pre-Processing" << std::endl;
	vtkDebugFile << "ASCII" << std::endl;
	vtkDebugFile << "DATASET UNSTRUCTURED_GRID" << std::endl << std::endl;

	vtkDebugFile << "POINTS " << nodes.size() + facets.size() << " FLOAT" << std::endl;

	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkDebugFile << std::fixed << nodes[n].x << "	" << std::fixed << nodes[n].y << "	" << std::fixed << 0.0 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << std::fixed << facets[i].center.x << "	" << std::fixed << facets[i].center.y << "	" << std::fixed << 0.0 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "CELLS " << edges.size() * 2 << "	" << edges.size() * 2 * 3 << std::endl;
	
	for (unsigned k = 0; k < edges.size(); ++k){
		vtkDebugFile << 2 << "	" << edges[k].nodes[0]->id << "	" << edges[k].nodes[1]->id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	for (unsigned k = 0; k < edges.size(); ++k){

		unsigned idA;
		if (edges[k].facets[0])
			idA = nodes.size() + edges[k].facets[0]->id;
		else
			idA = edges[k].nodes[0]->id;

		unsigned idB;
		if (edges[k].facets[1])
			idB = nodes.size() + edges[k].facets[1]->id;
		else
			idB = edges[k].nodes[1]->id;

		vtkDebugFile << 2 << "	" << idA << "	" << idB << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "CELL_TYPES " << edges.size()*2 << std::endl;
	
	for (unsigned k = 0; k < edges.size()*2; ++k){
		vtkDebugFile << 3 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "POINT_DATA " << nodes.size() + facets.size() << std::endl;
	vtkDebugFile << "SCALARS Address INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;

	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkDebugFile << nodes[n].id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << facets[i].id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "SCALARS Z%20Order INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;

	for (unsigned n = 0; n < nodes.size(); ++n){
		vtkDebugFile << nodes[n].spfOrder << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	for (unsigned i = 0; i < facets.size(); ++i){
		vtkDebugFile << facets[i].spfOrder << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "CELL_DATA " << edges.size()*2 << std::endl;
	vtkDebugFile << "SCALARS Address INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;
	
	for (unsigned k = 0; k < edges.size(); ++k){
		vtkDebugFile << edges[k].id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	for (unsigned k = 0; k < edges.size(); ++k){
		vtkDebugFile << edges[k].id << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile << std::endl;

	vtkDebugFile << "SCALARS Type INT" << std::endl;
	vtkDebugFile << "LOOKUP_TABLE default" << std::endl;

	for (unsigned k = 0; k < edges.size(); ++k){
		vtkDebugFile << 0 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	for (unsigned k = 0; k < edges.size(); ++k){
		vtkDebugFile << 1 << std::endl;
		showProgress((int)writeCounter++, writeTotal, "Writing", "Debug File");
	}

	vtkDebugFile.close();

	std::cout << std::endl;
}