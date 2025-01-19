//
// Created by chkwon on 3/22/22.
//

#include <fstream>
#include <cmath>
#include "InstanceCVRPLIB.h"

InstanceCVRPLIB::InstanceCVRPLIB(std::string pathToInstance, bool isRoundingInteger = true)
{
	std::string content, content2, content3;
	double serviceTimeData = 0.;

	// Read INPUT dataset
	std::ifstream inputFile(pathToInstance);
	if (inputFile.is_open())
	{
		getline(inputFile, content);
		getline(inputFile, content);
		getline(inputFile, content);
		for (inputFile >> content ; content != "NODE_COORD_SECTION" ; inputFile >> content)
		{
			if (content == "DIMENSION") { inputFile >> content2 >> nbClients; nbClients--; } // Need to substract the depot from the number of nodes
			else if (content == "EDGE_WEIGHT_TYPE")	inputFile >> content2 >> content3;
			else if (content == "CAPACITY")	inputFile >> content2 >> vehicleCapacity;
			else if (content == "DISTANCE") { inputFile >> content2 >> durationLimit; isDurationConstraint = true; }
			else if (content == "SERVICE_TIME")	inputFile >> content2 >> serviceTimeData;
			else throw std::string("Unexpected data in input file: " + content);
		}
		if (nbClients <= 0) throw std::string("Number of nodes is undefined");
		if (vehicleCapacity == 1.e30) throw std::string("Vehicle capacity is undefined");

		x_coords = std::vector<double>(nbClients + 1);
		y_coords = std::vector<double>(nbClients + 1);
		demands = std::vector<double>(nbClients + 1);
		service_time = std::vector<double>(nbClients + 1);

		// Reading node coordinates
		// depot must be the first element
		// 		- i = 0 in the for-loop below, or
		// 		- node_number = 1 in the .vrp file
		// customers are
		// 		- i = 1, 2, ..., nbClients in the for-loop below, or
		// 		- node_number = 2, 3, ..., nb_Clients in the .vrp file
		int node_number;
		for (int i = 0; i <= nbClients; i++)
		{
			inputFile >> node_number >> x_coords[i] >> y_coords[i];
			if (node_number != i + 1) throw std::string("The node numbering is not in order.");
		}

		// Reading demand information
		inputFile >> content;
		if (content != "DEMAND_SECTION") throw std::string("Unexpected data in input file: " + content);
		for (int i = 0; i <= nbClients; i++)
		{
			inputFile >> content >> demands[i];
			service_time[i] = (i == 0) ? 0. : serviceTimeData ;
		}

		// Calculating 2D Euclidean Distance
		dist_mtx = std::vector < std::vector< double > >(nbClients + 1, std::vector <double>(nbClients + 1));
		for (int i = 0; i <= nbClients; i++)
		{
			for (int j = 0; j <= nbClients; j++)
			{
				dist_mtx[i][j] = std::sqrt(
					(x_coords[i] - x_coords[j]) * (x_coords[i] - x_coords[j])
					+ (y_coords[i] - y_coords[j]) * (y_coords[i] - y_coords[j])
				);

				if (isRoundingInteger) dist_mtx[i][j] = round(dist_mtx[i][j]);
			}
		}

		// Reading depot information (in all current instances the depot is represented as node 1, the program will return an error otherwise)
		inputFile >> content >> content2 >> content3 >> content3;
		if (content != "DEPOT_SECTION") throw std::string("Unexpected data in input file: " + content);
		if (content2 != "1") throw std::string("Expected depot index 1 instead of " + content2);
		if (content3 != "EOF") throw std::string("Unexpected data in input file: " + content3);
	}
	else
		throw std::string("Impossible to open instance file: " + pathToInstance);
}
