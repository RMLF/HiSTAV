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
#include <string>
#include <vector>
#include <algorithm>

// GDAL/OGR (2.1)
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <ogrsf_frmts.h>

// Pre-Processor Headers
#include "../headers/common.hpp"
#include "../headers/meshInterpolation.hpp"
#include "../headers/meshEntities.hpp"
#include "../headers/meshGlobals.hpp"

// Forward Declarations
class simulationMesh;

/////////////////////////////////////////////////////////////////////////////////////////////////


regularGrid::regularGrid(){

	dx = 0.0;
	dy = 0.0;
	rootX = 0.0;
	rootY = 0.0;
}

void regularGrid::clear(){

	dx = 0.0;
	dy = 0.0;
	rootX = 0.0;
	rootY = 0.0;

	rasterVal.clear();
}

void regularGrid::allocateRaster(){

	rasterVal.resize(nLin, std::vector<double>(nCol, 0.0));
}

void regularGrid::importRaster(const std::string& rasterFileName, const std::string& rasterFileType){

	GDALAllRegister();
	GDALDataset* pDataset;
	GDALRasterBand* pRasData;

	double metaData[6];

	pDataset = (GDALDataset*)GDALOpen(rasterFileName.c_str(), GA_ReadOnly);
	pDataset->GetGeoTransform(metaData);
	pRasData = pDataset->GetRasterBand(1);

	dx = metaData[1];
	dy = metaData[5];

	rootX = metaData[0];
	rootY = metaData[3];

	nLin = (unsigned) pRasData->GetYSize();
	nCol = (unsigned) pRasData->GetXSize();
	allocateRaster();

	float* pScanRaster = (float*)CPLMalloc(sizeof(float)*nCol*nLin);
    pRasData->RasterIO(GF_Read, 0, 0, (int) nCol, (int) nLin, pScanRaster, nCol, nLin, GDT_Float32, 0, 0);

	for (unsigned i = 0; i < nLin; ++i){
		for (unsigned j = 0; j < nCol; ++j) {
            rasterVal[i][j] = (double) pScanRaster[nCol * i + j];
        }
		showProgress(i, nLin, "Importing", rasterFileType);
	}

	std::cout << std::endl;
	CPLFree(pScanRaster);
}

void simulationMesh::setInterpolatedData(const regularGrid& genGrid, const std::string& rasterFileType){

	double dx = genGrid.dx;
	double dy = genGrid.dy;
	double rootX = genGrid.rootX + dx / 2;
	double rootY = genGrid.rootY + dy / 2;

	unsigned nLin = genGrid.nLin;
	unsigned nCol = genGrid.nCol;

	double interpolatedValue;

	for (unsigned i = 0; i < nodes.size(); i++){

		unsigned m2 = (unsigned)std::min(std::max(0, (int)floor((nodes[i].y - rootY) / dy)), (int)nLin - 1);
		unsigned m1 = (unsigned)std::min(std::max(0, (int)ceil((nodes[i].y - rootY) / dy)), (int)nLin - 1);
		unsigned n1 = (unsigned)std::min(std::max(0, (int)floor((nodes[i].x - rootX) / dx)), (int)nCol - 1);
		unsigned n2 = (unsigned)std::min(std::max(0, (int)ceil((nodes[i].x - rootX) / dx)), (int)nCol - 1);

		double x1 = ((double) n1)*dx + rootX;
		double x2 = ((double) n2)*dx + rootX;
		double y1 = ((double) m1)*dy + rootY;
		double y2 = ((double) m2)*dy + rootY;

		if ((n1 == n2) && (m1 == m2))
			interpolatedValue = genGrid.rasterVal[m1][n1];
		else if (n1 == n2)
			interpolatedValue = (y2 - nodes[i].y) / (y2 - y1)*genGrid.rasterVal[m1][n1] + (nodes[i].y - y1) / (y2 - y1)*genGrid.rasterVal[m2][n1];
		else if (m1 == m2)
			interpolatedValue = (x2 - nodes[i].x) / (x2 - x1)*genGrid.rasterVal[m1][n1] + (nodes[i].x - x1) / (x2 - x1)*genGrid.rasterVal[m1][n2];
		else{
			double valXY1 = (x2 - nodes[i].x) / (x2 - x1)*genGrid.rasterVal[m1][n1] + (nodes[i].x - x1) / (x2 - x1)*genGrid.rasterVal[m1][n2];
			double valXY2 = (x2 - nodes[i].x) / (x2 - x1)*genGrid.rasterVal[m2][n2] + (nodes[i].x - x1) / (x2 - x1)*genGrid.rasterVal[m2][n2];
			interpolatedValue = (y2 - nodes[i].y) / (y2 - y1)*valXY1 + (nodes[i].y - y1) / (y2 - y1)*valXY2;
		}

		if (interpolatedValue != interpolatedValue){

			double validWeight = 0.0;
			double validSum = 0.0;

			double weightX1 = (x2 - nodes[i].x) / (x2 - x1);
			double weightX2 = (nodes[i].x - x1) / (x2 - x1);
			double weightY1 = (y2 - nodes[i].y) / (y2 - y1);
			double weightY2 = (nodes[i].y - y1) / (y2 - y1);

			if (genGrid.rasterVal[m1][n1] == genGrid.rasterVal[m1][n1]){
				validWeight += weightX1*weightY1;
				validSum += genGrid.rasterVal[m1][n1] * weightX1*weightY1;
			}

			if (genGrid.rasterVal[m1][n2] == genGrid.rasterVal[m1][n2]){
				validWeight += weightX2*weightY1;
				validSum += genGrid.rasterVal[m1][n2] * weightX2*weightY1;
			}

			if (genGrid.rasterVal[m2][n2] == genGrid.rasterVal[m2][n2]){
				validWeight += weightX1*weightY2;
				validSum += genGrid.rasterVal[m2][n2] * weightX1*weightY2;
			}

			if (genGrid.rasterVal[m2][n1] == genGrid.rasterVal[m2][n1]){
				validWeight += weightX2*weightY2;
				validSum += genGrid.rasterVal[m2][n1] * weightX2*weightY2;
			}

			interpolatedValue = validSum / validWeight;
		}

		if (rasterFileType == "DTM")
			nodes[i].z = interpolatedValue;
		else if (rasterFileType == "frCoef")
			nodes[i].frCoef = interpolatedValue;
        else if (rasterFileType == "infiltCoef")
            nodes[i].permeability = interpolatedValue;
		else if (rasterFileType == "dZMax")
			nodes[i].dzMax = interpolatedValue;
		else if (rasterFileType == "gradeId")
			nodes[i].gradeId = (unsigned) round(interpolatedValue);
		else if (rasterFileType == "slideDepth")
			nodes[i].slideDepth = interpolatedValue;
		else if (rasterFileType == "slideId")
			nodes[i].slideId = (unsigned) round(interpolatedValue);

		showProgress(i, nodes.size(), "Interpolating", rasterFileType);
	}

	std::cout << std::endl;
}
