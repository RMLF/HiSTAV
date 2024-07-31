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
#include <string>

/////////////////////////////////////////////////////////////////////////////////////////////////


class meshControl{
public:

	meshControl();

	bool hasDtm;
	bool hasFrictionCoef;
	bool hasBedrockOffset;
    bool hasInfiltrationCoef;
	bool hasGradeId;
	bool hasLandslides;
	bool hasLandslidesId;

	std::string rasterFolder;
	std::string dtmFileName;
	std::string bedrockOffsetFileName;
	std::string frCoefFileName;
    std::string infiltCoefFileName;
	std::string gradeIdFileName;
	std::string landslidesFileName;
	std::string landslidesIdFileName;
	
	std::string meshFolder;
	std::string meshDimFileName;
	std::string nodesFileName;
	std::string elementsFileName;

	std::string boundaryDataFolder;
	std::string boundaryDimFileName;
	std::string boundaryFileName;

	std::string bedDataFolder;
	std::string bedLayersFileName;
	std::string bedrockOffsetNodesFileName;
    std::string infiltCoefNodesFileName;
	std::string frCoefNodesFileName;
	std::string gradeIdNodesFileName;
	std::string landslidesNodesFileName;
	std::string landslidesIdNodesFileName;

	std::string outputVTKFileName;
	std::string facetsDebugVTKFileName;
	std::string edgesDebugVTKFileName;

	std::string targetFolder;
};