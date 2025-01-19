/*MIT License

Copyright(c) 2020 Thibaut Vidal

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#ifndef PARAMS_H
#define PARAMS_H

#include "CircleSector.h"
#include "AlgorithmParameters.h"
#include <string>
#include <vector>
#include <list>
#include <set>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>
#include <time.h>
#include <climits>
#include <algorithm>
#include <unordered_set>
#include <random>
#define MY_EPSILON 0.00001 // Precision parameter, used to avoid numerical instabilities
#define PI 3.14159265359

struct Client
{
	double coordX;			// Coordinate X
	double coordY;			// Coordinate Y
	double serviceDuration; // Service duration
	double demand;			// Demand
	int polarAngle;			// Polar angle of the client around the depot, measured in degrees and truncated for convenience
};

class Params
{
public:

	/* PARAMETERS OF THE GENETIC ALGORITHM */
	bool verbose;                       // Controls verbose level through the iterations
	AlgorithmParameters ap;	            // Main parameters of the HGS algorithm

	/* ADAPTIVE PENALTY COEFFICIENTS */
	double penaltyCapacity;				// Penalty for one unit of capacity excess (adapted through the search)
	double penaltyDuration;				// Penalty for one unit of duration excess (adapted through the search)

	/* START TIME OF THE ALGORITHM */
	clock_t startTime;                  // Start time of the optimization (set when Params is constructed)

	/* RANDOM NUMBER GENERATOR */       
	std::minstd_rand ran;               // Using the fastest and simplest LCG. The quality of random numbers is not critical for the LS, but speed is

	/* DATA OF THE PROBLEM INSTANCE */
	bool isDurationConstraint ;								// Indicates if the problem includes duration constraints
	int nbClients ;											// Number of clients (excluding the depot)
	int nbVehicles ;										// Number of vehicles
	double durationLimit;									// Route duration limit
	double vehicleCapacity;									// Capacity limit
	double totalDemand ;									// Total demand required by the clients
	double maxDemand;										// Maximum demand of a client
	double maxDist;											// Maximum distance between two clients
	std::vector< Client > cli ;								// Vector containing information on each client
	const std::vector< std::vector< double > >& timeCost;	// Distance matrix
	std::vector< std::vector< int > > correlatedVertices;	// Neighborhood restrictions: For each client, list of nearby customers
	bool areCoordinatesProvided;                            // Check if valid coordinates are provided

	// Initialization from a given data set
	Params(const std::vector<double>& x_coords,
		const std::vector<double>& y_coords,
		const std::vector<std::vector<double>>& dist_mtx,
		const std::vector<double>& service_time,
		const std::vector<double>& demands,
		double vehicleCapacity,
		double durationLimit,
		int nbVeh,
		bool isDurationConstraint,
		bool verbose,
		const AlgorithmParameters& ap);
};
#endif

