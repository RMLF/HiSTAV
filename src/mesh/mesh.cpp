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

// STL
#include <iostream>
#include <chrono>

// Pre-Processor Header's
#include "./headers/common.hpp"
#include "./headers/meshControl.hpp"
#include "./headers/meshEntities.hpp"
#include "./headers/meshOrder.hpp"
#include "./headers/meshRead.hpp"
#include "./headers/meshBoundary.hpp"
#include "./headers/meshGen.hpp"
#include "./headers/meshGlobals.hpp"
#include "./headers/meshInterpolation.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


int main(){

	std::cout << std::endl << "******************************************************************************************************" << std::endl;
	std::cout << "STAV-2D Pre-Processor v1.0 2018(C) by Daniel A. S. Conde, Ricardo B. Canelas & Rui M. L. Ferreira" << std::endl;
	std::cout << "******************************************************************************************************" << std::endl << std::endl;
	std::cout << std::endl;
	std::cout << "            _____ _______  __      __    ___  _____" << std::endl;
	std::cout << "           / ____|__   __|/\\ \\    / /   |__ \\|  __ \\ " << std::endl;
	std::cout << "          | (___    | |  /  \\ \\  / /       ) | |  | |" << std::endl;
	std::cout << "           \\___ \\   | | / /\\ \\ \\/ /       / /| |  | |" << std::endl;
	std::cout << "           ____) |  | |/ ____ \\  /       / /_| |__| |" << std::endl;
	std::cout << "          |_____/   |_/_/    \\_\\/       |____|_____/" << std::endl;
	std::cout << std::endl;
	std::cout << std::endl << "  Importing GIS shapefiles:" << std::endl << std::endl;

	std::chrono::time_point<std::chrono::system_clock> timer = std::chrono::system_clock::now();

	toLog("", 0);
	toLog("Started running STAV-2D MeshPreProcessor v1.0", 0);

	std::string inputFileName = mesh.geometry.shapesFolder + mesh.geometry.domainFileName;
	if (fileExists(inputFileName)){
		mesh.geometry.importShapeFile(inputFileName, mesh.geometry.domain);
		toLog("Domain file parsed", 1);
		std::cout << "    -> Domain		[OK]" << std::endl;
	}else{
		std::cout << "    -> No domain file?! LOL" << std::endl;
		exitOnKeypress(1);
	}

	inputFileName = mesh.geometry.shapesFolder + mesh.geometry.boundariesFileName;
	std::string inputOtherFileName = mesh.geometry.shapesFolder + mesh.geometry.boundaryPointsFileName;
	if (fileExists(inputFileName) && fileExists(inputOtherFileName)){
		mesh.geometry.importBoundaryShapeFiles();
		toLog("Boundaries file parsed", 1);
		std::cout << "    -> Boundaries	[OK]" << std::endl;
	}else
		std::cout << "    -> Boundaries	[NA]" << std::endl;

	inputFileName = mesh.geometry.shapesFolder + mesh.geometry.refinementsFileName;
	if (fileExists(inputFileName)){
		mesh.geometry.importShapeFile(inputFileName, mesh.geometry.refinements);
		toLog("Refinements file parsed", 1);
		std::cout << "    -> Refinements	[OK]" << std::endl;
	}else
		std::cout << "    -> Refinements	[NA]" << std::endl;

	inputFileName = mesh.geometry.shapesFolder + mesh.geometry.alignmentsFileName;
	if (fileExists(inputFileName)){
		mesh.geometry.importShapeFile(inputFileName, mesh.geometry.alignments);
		toLog("Alignments file parsed", 1);
		std::cout << "    -> Alignments	[OK]" << std::endl;
	}else
		std::cout << "    -> Alignments	[NA]" << std::endl;

	inputFileName = mesh.geometry.shapesFolder + mesh.geometry.voidsFileName;
	if (fileExists(inputFileName)){
		mesh.geometry.importShapeFile(inputFileName, mesh.geometry.voids);
		toLog("Obstacles file parsed", 1);
		std::cout << "    -> Obstacles	[OK]" << std::endl;
	}else
		std::cout << "    -> Obstacles	[NA]" << std::endl;

	mesh.geometry.writeGmshFile();
	mesh.geometry.callGmsh();

	mesh.readMeshFile();
	mesh.setConnectivity();

	if (true){
		std::cout << std::endl << "  Enhancing memory locality with continuous space-fill ordering" << std::endl << std::endl;
		mesh.setBox();
		mesh.applySpfOrderingToNodes();
		mesh.applySpfOrderingToFacets();
		toLog("Mesh re-ordering completed", 1);
	}else
		std::cout << std::endl << "  No space-fill ordering: cache performance will not be optimal"<< std::endl << std::endl;

	if (mesh.polyCGAL.size_of_halfedges() / 2 != mesh.polyCGAL.size_of_facets() + mesh.polyCGAL.size_of_vertices() - 1){
		std::cout << std::endl << "  Mesh has voids" << std::endl << std::endl;
		toLog("Mesh appears to have voids", 1);
	}

	mesh.setFacetsCcWise();
	mesh.assignPhysicalEdges();
	mesh.polyCGAL.clear();

	toLog("CGAL structures cleared", 1);
	std::cout << std::endl;

	toLog("Scanning for interpolation data", 0);
	inputFileName = mesh.control.rasterFolder + mesh.control.dtmFileName;
	if (fileExists(inputFileName)){
		regularGrid dtmRaster;
		std::cout << "  Opening DTM file: " << mesh.control.dtmFileName << std::endl << std::endl;
		dtmRaster.importRaster(inputFileName, "DTM");
		mesh.setInterpolatedData(dtmRaster, "DTM");
		toLog("DTM raster parsed", 1);
		std::cout << std::endl;
		dtmRaster.clear();
		mesh.control.hasDtm = true;
	}else
		std::cout << "  No DTM file" << std::endl;

	inputFileName = mesh.control.rasterFolder + mesh.control.frCoefFileName;
	if (fileExists(inputFileName)){
		regularGrid frCoefRaster;
		std::cout << "  Opening friction coefficient file: " << mesh.control.frCoefFileName << std::endl << std::endl;
		frCoefRaster.importRaster(inputFileName, "frCoef");
		mesh.setInterpolatedData(frCoefRaster, "frCf");
		toLog("Friction coefficient raster parsed", 1);
		std::cout << std::endl;
		frCoefRaster.clear();
		mesh.control.hasFrictionCoef = true;
	}else
		std::cout << "  No friction coefficient file" << std::endl;

	inputFileName = mesh.control.rasterFolder + mesh.control.bedrockOffsetFileName;
	if (fileExists(inputFileName)){
		regularGrid dZMaxRaster;
		std::cout << "  Opening bedrock offset file " << mesh.control.bedrockOffsetFileName << std::endl << std::endl;
		dZMaxRaster.importRaster(inputFileName, "dZMax");
		mesh.setInterpolatedData(dZMaxRaster, "dZMax");
		toLog("Bedrock offset raster parsed", 1);
		std::cout << std::endl;
		dZMaxRaster.clear();
		mesh.control.hasBedrockOffset = true;
	}else
		std::cout << "  No bedrock offset file" << std::endl;

	inputFileName = mesh.control.rasterFolder + mesh.control.gradeIdFileName;
	if (fileExists(inputFileName)){
		regularGrid gradeIdRaster;
		std::cout << "  Opening grading curve file: " << mesh.control.gradeIdFileName << std::endl << std::endl;
		gradeIdRaster.importRaster(inputFileName, "gradeId");
		mesh.setInterpolatedData(gradeIdRaster, "grId");
		toLog("Grading curve raster parsed", 1);
		std::cout << std::endl;
		gradeIdRaster.clear();
		mesh.control.hasGradeId = true;
	}else
		std::cout << "  No grading curve file" << std::endl;

	inputFileName = mesh.control.rasterFolder + mesh.control.landslidesFileName;
	if (fileExists(inputFileName)){
		regularGrid slideDepthRaster;
		std::cout << "  Opening landslide depths file: " << mesh.control.landslidesFileName << std::endl << std::endl;
		slideDepthRaster.importRaster(inputFileName, "sldDepth");
		mesh.setInterpolatedData(slideDepthRaster, "Depth");
		toLog("Landslide depths raster parsed", 1);
		std::cout << std::endl;
		slideDepthRaster.clear();
		mesh.control.hasLandslides = true;
	}else
		std::cout << "  No landslide depths file" << std::endl;

	inputFileName = mesh.control.rasterFolder + mesh.control.landslidesIdFileName;
	if (fileExists(inputFileName)){
		regularGrid slideIdRaster;
		std::cout << "  Opening landslide IDs file " << mesh.control.landslidesIdFileName << std::endl << std::endl;
		slideIdRaster.importRaster(inputFileName, "slideId");
		mesh.setInterpolatedData(slideIdRaster, "sldId");
		toLog("Landslide IDs raster parsed", 1);
		std::cout << std::endl;
		slideIdRaster.clear();
		mesh.control.hasLandslidesId = true;
	}else
		std::cout << "  No landslide IDs file" << std::endl;

	int elapsedTime = (int) std::chrono::duration_cast<std::chrono::seconds> (std::chrono::system_clock::now() - timer).count();
	std::cout << std::endl << "  Pre-processing completed in " << elapsedTime << " seconds" << std::endl << std::endl;
	toLog("Mesh generation and pre-processing completed", 0);
	timer = std::chrono::system_clock::now();

	mesh.writeMeshInfo();
	mesh.writeMeshNodes();
	mesh.writeMeshFacets();
	mesh.writeMeshBoundaries();
	mesh.writeMeshQualityVTK();
	//mesh.writeMeshDebugVTK();

	elapsedTime = (int) std::chrono::duration_cast<std::chrono::seconds> (std::chrono::system_clock::now() - timer).count();
	std::cout << std::endl << "  All files written in " << elapsedTime << " seconds" << std::endl;

	preProcLog.close();
	exitOnKeypress(0);
}
