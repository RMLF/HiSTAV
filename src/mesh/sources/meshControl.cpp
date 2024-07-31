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
#include <string>

// STAV's Headers
#include "../headers/meshControl.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////


meshControl::meshControl(){

	hasDtm = false;
	hasFrictionCoef = false;
	hasBedrockOffset = false;
	hasGradeId = false;
	hasLandslides = false;
	hasLandslidesId = false;

	rasterFolder = "./rasters/";
	dtmFileName = "DTM.tif";
	bedrockOffsetFileName = "bedrockOffset.tif";
	frCoefFileName = "frictionCoef.tif";
	gradeIdFileName = "gradeId.tif";
	landslidesFileName = "landslideDepth.tif";
	landslidesIdFileName = "landslideId.tif";

	meshFolder = "mesh/";
	meshDimFileName = "info.mesh";
	nodesFileName = "nodes.mesh";
	elementsFileName = "elements.mesh";

	boundaryDataFolder = "boundary/meshData/";
	boundaryFileName = "boundaryIdx";
	boundaryDimFileName = "boundaryDim.bnd";

	bedDataFolder = "bed/activeLayers/";
	bedLayersFileName = "bedLayers.sed";
	bedrockOffsetNodesFileName = "nodesBedrockOffset.sed";
	frCoefNodesFileName = "nodesFrictionCoef.sed";
	gradeIdNodesFileName = "nodesGradeId.sed";
	landslidesNodesFileName = "nodesLandslides.sed";
	landslidesIdNodesFileName = "nodesLandslideId.sed";

	outputVTKFileName = "vtk/meshQuality.vtk";
	facetsDebugVTKFileName = "vtk/facetsDebug.vtk";
	edgesDebugVTKFileName = "vtk/edgesDebug.vtk";

	targetFolder = "../";
}