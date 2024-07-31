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
#include <iostream>
#include <vector>
#include <string>
#include <utility>
#include <algorithm> 

// GDAL/OGR (2.1)
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogrsf_frmts.h>

// Boost (1.55)
#include <boost/algorithm/string/predicate.hpp>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshControl.hpp"
#include "../headers/meshGen.hpp"
#include "../headers/meshEntities.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

meshGeometryType::meshGeometryType(){

	charLength = 0.0;
	auxField = 0.0;
}

meshGen::meshGen(){

	shapesFolder = "./shapefiles/";

	domainFileName = "domain.shp";
	refinementsFileName = "refinements.shp";
	alignmentsFileName = "alignments.shp";
	voidsFileName = "voids.shp";
	boundariesFileName = "boundaries.shp";
	boundaryPointsFileName = "boundary-points.shp";

	gmshFolder = "./gmsh/";

	gmshExe = "gmsh";
	gmshGeoFile = "generatedSTAVMesh.geo";
	gmshMeshFile = "generatedSTAVMesh.mesh";
}

void meshGen::importShapeFile(const std::string& shapeFileName, std::vector<meshGeometryType>& geometryType){

    GDALAllRegister();
    GDALDataset *pDataset;
	OGRLayer *pLayer;
	OGRGeometry *pGeometry;
	OGRFeature *pFeature;
	OGRFeatureDefn *pFDefn;
	OGRFieldDefn *pFieldDefn;
	OGRPoint *pPoint;
	OGRLineString *pLineString;
	OGRLinearRing *pExteriorRing;
	OGRPolygon *pPolygon;

	OGRPoint tmpPoint;
	OGRwkbGeometryType layerType;

    pDataset = (GDALDataset*) GDALOpenEx(shapeFileName.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
	pLayer = pDataset->GetLayer(0);
	pLayer->ResetReading();
	layerType = pLayer->GetGeomType();
	pFDefn = pLayer->GetLayerDefn();

	geometryType.resize(pLayer->GetFeatureCount());

	if (wkbFlatten(layerType) == wkbPolygon){
		
		for (unsigned i = 0; i < geometryType.size(); ++i){
			pFeature = pLayer->GetNextFeature();
			pGeometry = pFeature->GetGeometryRef();

			if (pGeometry != NULL && wkbFlatten(pGeometry->getGeometryType()) == wkbPolygon){

				pPolygon = (OGRPolygon*)pGeometry;
				pExteriorRing = pPolygon->getExteriorRing();
				geometryType[i].nodes.resize(pExteriorRing->getNumPoints() - 1);

				for (int f = 0; f < pFDefn->GetFieldCount(); ++f){
					pFieldDefn = pFDefn->GetFieldDefn(f);
					if (boost::iequals(pFieldDefn->GetNameRef(), "cl")){
						geometryType[i].charLength = pFeature->GetFieldAsDouble(f);
					}else if (boost::iequals(pFieldDefn->GetNameRef(), "aux")){
						geometryType[i].auxField = pFeature->GetFieldAsDouble(f);
					}
				}

				for (unsigned n = 0; n < geometryType[i].nodes.size(); ++n){
					pExteriorRing->getPoint(n, &tmpPoint);
					geometryType[i].nodes[n].x = tmpPoint.getX();
					geometryType[i].nodes[n].y = tmpPoint.getY();
					geometryType[i].nodes[n].id = n;
				}
			}
		}
	}else if (wkbFlatten(layerType) == wkbLineString){
		
		for (int i = 0; i < pLayer->GetFeatureCount(); ++i){
			pFeature = pLayer->GetNextFeature();
			pGeometry = pFeature->GetGeometryRef();

			if (pGeometry != NULL && wkbFlatten(pGeometry->getGeometryType()) == wkbLineString){

				pLineString = (OGRLineString*)pGeometry;
				geometryType[i].nodes.resize(pLineString->getNumPoints());

				for (int f = 0; f < pFDefn->GetFieldCount(); ++f){
					pFieldDefn = pFDefn->GetFieldDefn(f);
					if (boost::iequals(pFieldDefn->GetNameRef(), "cl")){
						geometryType[i].charLength = pFeature->GetFieldAsDouble(f);
					}else if (boost::iequals(pFieldDefn->GetNameRef(), "aux")){
						geometryType[i].auxField = pFeature->GetFieldAsDouble(f);
					}
				}

				for (int n = 0; n < pLineString->getNumPoints(); ++n){
					pLineString->getPoint(n, &tmpPoint);
					geometryType[i].nodes[n].x = tmpPoint.getX();
					geometryType[i].nodes[n].y = tmpPoint.getY();
					geometryType[i].nodes[n].id = n;
				}
			}
		}
	}else if (wkbFlatten(layerType) == wkbPoint){

		geometryType.resize(1);
		geometryType[0].nodes.resize(pLayer->GetFeatureCount());

		for (unsigned i = 0; i < geometryType.size(); ++i){
			
			pFeature = pLayer->GetNextFeature();
			pGeometry = pFeature->GetGeometryRef();

			if (pGeometry != NULL && wkbFlatten(pGeometry->getGeometryType()) == wkbPoint){
				pPoint = (OGRPoint*)pGeometry;
				geometryType[0].nodes[i].x = pPoint->getX();
				geometryType[0].nodes[i].y = pPoint->getY();
			}
		}
	}
}

void meshGen::writeGmshFile(){

	unsigned ptCounter = 0;
	unsigned ptSavedCounter = 0;
	unsigned linCounter = 0;
	unsigned linSavedCounter = 0;
	unsigned loopCounter = 0;

	std::ofstream gmshFile;
	gmshFile.open(gmshFolder + gmshGeoFile);
	//gmshFile << std::endl << "Mesh.RandomFactor = 1e-6;" << std::endl << std::endl;
	gmshFile << "// Domain definition ------------------------------------------------------------------------------" << std::endl << std::endl;

	for (unsigned i = 0; i < domain[0].nodes.size(); ++i){
		gmshFile << "Point(" << i << ") = {" << std::fixed << domain[0].nodes[i].x << std::fixed << ", " << domain[0].nodes[i].y << ", 0.0, " << domain[0].charLength << "};" << std::endl;
		++ptCounter;
	}

	gmshFile << std::endl;

	for (unsigned i = 0; i < domain[0].nodes.size(); ++i){
		gmshFile << "Line(" << i << ") = {" << i << ", " << (i + 1) % domain[0].nodes.size() << "};" << std::endl;
		++linCounter;
	}

	gmshFile << std::endl;

	for (int b = 0; b < (int) boundaries.size(); ++b){

		gmshFile << "Physical Line(" << b << ") = {";
		int increment = 1;
		unsigned writtenLinesCounter = 0;
		
		for (unsigned p = 0; p < boundaries[b].nodes.size() - 1; ++p){

			unsigned targetPtId = boundaries[b].domainNodes[p]->id;
			unsigned nextTargetPtId = (targetPtId + increment)%domain[0].nodes.size();
			
			if (domain[0].nodes[targetPtId].physId != domain[0].nodes[nextTargetPtId].physId)
				increment = -1;
			
			nextTargetPtId = (targetPtId + increment) % domain[0].nodes.size();
			
			if (domain[0].nodes[targetPtId].physId == b && domain[0].nodes[nextTargetPtId].physId == b){

				int writeThisLine = std::min(targetPtId, nextTargetPtId);
				if ((targetPtId == 0 && nextTargetPtId == domain[0].nodes.size() - 1) || (nextTargetPtId == 0 && targetPtId == domain[0].nodes.size() - 1))
					writeThisLine = std::max(targetPtId, nextTargetPtId);

				gmshFile << writeThisLine;
				writtenLinesCounter++;

                if (writtenLinesCounter < (boundaries[b].nodes.size() - 1))
                    gmshFile << ", ";
                else if (writtenLinesCounter == (boundaries[b].nodes.size() - 1))
                    gmshFile << "};" << std::endl;
			}
		}
	}

	if (boundaries.size() > 0)
		gmshFile << std::endl;

	gmshFile << "Line Loop(" << loopCounter++ << ") = {0:" << domain[0].nodes.size() - 1 << "};" << std::endl << std::endl;

	if (!voids.empty()){
		gmshFile << std::endl << "// Void areas definition --------------------------------------------------------------------------" << std::endl << std::endl;
		for (unsigned i = 0; i < voids.size(); ++i){
			if (!voids[i].nodes.empty()){
				ptSavedCounter = ptCounter;
				for (unsigned k = 0; k < voids[i].nodes.size(); k++){
					gmshFile << "Point(" << k + ptSavedCounter << ") = {" << std::fixed << voids[i].nodes[k].x << std::fixed
						<< ", " << voids[i].nodes[k].y << ", 0.0, " << voids[i].charLength << "};" << std::endl;
					++ptCounter;
				}
				gmshFile << std::endl << "Line(" << linCounter++ << ") = {" << ptSavedCounter << ":" << (ptCounter - 1) << "};" << std::endl;
				gmshFile << "Line(" << linCounter++ << ") = {" << (ptCounter - 1) << "," << ptSavedCounter << "};" << std::endl << std::endl;
				gmshFile << "Line Loop(" << loopCounter++ << ") = {" << (linCounter - 2) << ":" << (linCounter - 1) << "};" << std::endl << std::endl;
			}
		}
	}

	if (loopCounter > 1)
		gmshFile << std::endl << "Plane Surface(0) = {0:" << (loopCounter - 1) << "};" << std::endl;
	else
		gmshFile << std::endl << "Plane Surface(0) = {0};" << std::endl;

	if (!refinements.empty()){
		gmshFile << std::endl << "// Refinement areas definition --------------------------------------------------------------------" << std::endl << std::endl;
		for (unsigned i = 0; i < refinements.size(); ++i){
			if (!refinements[i].nodes.empty()){
				ptSavedCounter = ptCounter;
				linSavedCounter = linCounter;
				for (unsigned k = 0; k < refinements[i].nodes.size(); k++){
					gmshFile << "Point(" << k + ptSavedCounter << ") = {" << std::fixed << refinements[i].nodes[k].x << std::fixed
						<< ", " << refinements[i].nodes[k].y << ", 0.0, " << refinements[i].charLength << "};" << std::endl;
					ptCounter++;
				}
				gmshFile << std::endl << "Line(" << linCounter++ << ") = {" << ptSavedCounter << ":" << (ptCounter - 1) << "};" << std::endl;
				gmshFile << "Line(" << linCounter++ << ") = {" << (ptCounter - 1) << "," << ptSavedCounter << "};" << std::endl << std::endl;
				gmshFile << "Line{" << linSavedCounter << ":" << (linCounter - 1) << "} In Surface{0};" << std::endl << std::endl;
			}
		}
	}

	if (!alignments.empty()){
		gmshFile << std::endl << "// Alignment areas definition --------------------------------------------------------------------" << std::endl << std::endl;
		for (unsigned i = 0; i < alignments.size(); i++){
			if (!alignments[i].nodes.empty()){
				linSavedCounter = linCounter;
				ptSavedCounter = ptCounter;
				for (unsigned k = 0; k < alignments[i].nodes.size(); k++){
					gmshFile << "Point(" << k + ptSavedCounter << ") = {" << std::fixed << alignments[i].nodes[k].x << std::fixed
						<< ", " << alignments[i].nodes[k].y << ", 0.0, " << alignments[i].charLength << "};" << std::endl;
					ptCounter++;
				}
				gmshFile << std::endl << "Line(" << linCounter++ << ") = {" << ptSavedCounter << ":" << (ptCounter - 1) << "};" << std::endl << std::endl;
				gmshFile << "Line{" << linSavedCounter << ":" << (linCounter - 1) << "} In Surface{0};" << std::endl << std::endl;
			}
		}
	}

	gmshFile << std::endl << "Physical Surface(0) = {0};" << std::endl;
	gmshFile << "Mesh 2;" << std::endl;
	gmshFile << "Coherence;" << std::endl;
	gmshFile << "Mesh.SaveElementTagType=2;" << std::endl;
	gmshFile.close();
}

void meshGen::callGmsh(){

	std::string fullCommand = "./gmsh/" + gmshExe + " ./gmsh/" + gmshGeoFile + " -0  -format mesh -smooth 5 -o ./gmsh/" + gmshMeshFile;
	toLog("Gmsh called with command: " + fullCommand, 0);

	std::cout << std::endl << std::endl << "  Transfering control to Gmsh ..." << std::endl << std::endl;
	std::cout << "GMSH OUTPUT ------------------------------------------------------------------------" << std::endl;

	system(fullCommand.c_str());
	toLog("Gmsh started running", 1);

	std::cout << "GMSH OUTPUT ------------------------------------------------------------------------" << std::endl;
	std::cout << std::endl << "  Transfering control back to STAV-PreProcessing ..." << std::endl << std::endl;
	toLog("Gmsh returned from execution", 1);
}