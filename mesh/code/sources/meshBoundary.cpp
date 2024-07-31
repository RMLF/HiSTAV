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
#include <iostream>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshBoundary.hpp"
#include "../headers/meshEntities.hpp"
#include "../headers/meshGen.hpp"
#include "../headers/meshGlobals.hpp"

// GDAL/OGR (2.1)
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogrsf_frmts.h>

// Boost (1.55)
#include <boost/algorithm/string/predicate.hpp>

/////////////////////////////////////////////////////////////////////////////////////////////////


meshBoundary::meshBoundary(){

	id = 0;
}

void meshGen::importBoundaryShapeFiles(){

    GDALAllRegister();
    GDALDataset *pDataset;
    OGRLayer *pLayer;
    OGRGeometry *pGeometry;
    OGRFeature *pFeature;
    OGRFeatureDefn *pFDefn;
    OGRFieldDefn *pFieldDefn;
    OGRPoint *pPoint;
    OGRLineString *pLineString;

	OGRPoint tmpPoint;
	OGRwkbGeometryType LayerType;

    pDataset = (GDALDataset*) GDALOpenEx((shapesFolder + boundariesFileName).c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
	pLayer = pDataset->GetLayer(0);
	pLayer->ResetReading();
	LayerType = pLayer->GetGeomType();
	pFDefn = pLayer->GetLayerDefn();

	boundaries.resize(pLayer->GetFeatureCount());

	if (wkbFlatten(LayerType) == wkbLineString)
		for (unsigned b = 0; b < boundaries.size(); ++b){

			pFeature = pLayer->GetNextFeature();
			pGeometry = pFeature->GetGeometryRef();

			if (pGeometry != NULL && wkbFlatten(pGeometry->getGeometryType()) == wkbLineString){

				pLineString = (OGRLineString*)pGeometry;
				for (int f = 0; f < pFDefn->GetFieldCount(); ++f){
					pFieldDefn = pFDefn->GetFieldDefn(f);
					if (boost::iequals(pFieldDefn->GetNameRef(), "type"))
						boundaries[b].type = pFeature->GetFieldAsString(f);
				}

				boundaries[b].nodes.resize(pLineString->getNumPoints());
				boundaries[b].seriesFile.resize(boundaries[b].nodes.size());
				boundaries[b].domainNodes.resize(boundaries[b].nodes.size(), NULL);

				for (unsigned n = 0; n < boundaries[b].nodes.size(); ++n){
					pLineString->getPoint(n, &tmpPoint);
					boundaries[b].nodes[n].x = tmpPoint.getX();
					boundaries[b].nodes[n].y = tmpPoint.getY();
					boundaries[b].nodes[n].id = n;
				}
			}
		}

    pDataset = (GDALDataset*) GDALOpenEx((shapesFolder + boundaryPointsFileName).c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
	pLayer = pDataset->GetLayer(0);
	pLayer->ResetReading();
	LayerType = pLayer->GetGeomType();
	pFDefn = pLayer->GetLayerDefn();

	double tol = 1.0e-3;

	if (wkbFlatten(LayerType) == wkbPoint)
		for (int i = 0; i < pLayer->GetFeatureCount(); ++i){
			pFeature = pLayer->GetNextFeature();
			pGeometry = pFeature->GetGeometryRef();

			if (pGeometry != NULL && wkbFlatten(pGeometry->getGeometryType()) == wkbPoint){
				pPoint = (OGRPoint*)pGeometry;

				for (unsigned b = 0; b < boundaries.size(); ++b)
					for (unsigned p = 0; p < boundaries[b].nodes.size(); ++p)
						if (sqrt(pow(boundaries[b].nodes[p].x - pPoint->getX(), 2.0) + pow(boundaries[b].nodes[p].y - pPoint->getY(), 2.0)) <= tol)
							for (int f = 0; f < pFDefn->GetFieldCount(); f++){
								pFieldDefn = pFDefn->GetFieldDefn(f);
								if (boost::iequals(pFieldDefn->GetNameRef(), "series"))
									boundaries[b].seriesFile[p] = (pFeature->GetFieldAsString(f));
							}
			}
		}

	for (unsigned b = 0; b < boundaries.size(); b++)
		for (unsigned p = 0; p < boundaries[b].nodes.size(); p++)
			for (unsigned k = 0; k < domain[0].nodes.size(); k++)
				if (sqrt(pow(boundaries[b].nodes[p].x - domain[0].nodes[k].x, 2.0) + pow(boundaries[b].nodes[p].y - domain[0].nodes[k].y, 2.0)) <= tol){
					boundaries[b].domainNodes[p] = &domain[0].nodes[k];
					domain[0].nodes[k].physId = b;
				}
}

void simulationMesh::assignPhysicalEdges(){

	boundaries = geometry.boundaries;
	std::vector<edge*> boundaryEdges;

	for (unsigned k = 0; k < edges.size(); k++){
		if (edges[k].facets[0] == NULL || edges[k].facets[1] == NULL)
			boundaryEdges.push_back(&edges[k]);
		showProgress(k, edges.size(), "Scanning", "Boundaries");
	}

	std::cout << std::endl;

	for (unsigned k = 0; k < boundaryEdges.size(); ++k)
		for (unsigned kk = 0; kk < physEdges.size(); ++kk){
			if (boundaryEdges[k]->nodes[0] == physEdges[kk].nodes[0] || boundaryEdges[k]->nodes[0] == physEdges[kk].nodes[1])
				if (boundaryEdges[k]->nodes[1] == physEdges[kk].nodes[0] || boundaryEdges[k]->nodes[1] == physEdges[kk].nodes[1]){
					
					int bndIdx = physEdges[kk].physId;
					boundaryEdges[k]->physId = bndIdx;
					boundaries[bndIdx].meshEdges.push_back(boundaryEdges[k]);

					if (boundaryEdges[k]->facets[0]){
						boundaryEdges[k]->facets[0]->physId = bndIdx;
						boundaries[bndIdx].meshFacets.push_back(boundaryEdges[k]->facets[0]);
					}else if (boundaryEdges[k]->facets[1]){
						boundaryEdges[k]->facets[1]->physId = bndIdx;
						boundaries[bndIdx].meshFacets.push_back(boundaryEdges[k]->facets[1]);
					}
				}

			showProgress((int) k, boundaryEdges.size(), "Setting", "Boundaries");
		}

	std::cout << std::endl;
}