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

#ifndef INDIVIDUAL_H
#define INDIVIDUAL_H

#include "Params.h"

struct EvalIndiv
{
	double penalizedCost = 0.;		// Penalized cost of the solution
	int nbRoutes = 0;				// Number of routes
	double distance = 0.;			// Total distance
	double capacityExcess = 0.;		// Sum of excess load in all routes
	double durationExcess = 0.;		// Sum of excess duration in all routes
	bool isFeasible = false;		// Feasibility status of the individual
};

class Individual
{
public:

  EvalIndiv eval;															// Solution cost parameters
  std::vector < int > chromT ;												// Giant tour representing the individual
  std::vector < std::vector <int> > chromR ;								// For each vehicle, the associated sequence of deliveries (complete solution)
  std::vector < int > successors ;											// For each node, the successor in the solution (can be the depot 0)
  std::vector < int > predecessors ;										// For each node, the predecessor in the solution (can be the depot 0)
  std::multiset < std::pair < double, Individual* > > indivsPerProximity ;	// The other individuals in the population, ordered by increasing proximity (the set container follows a natural ordering based on the first value of the pair)
  double biasedFitness;														// Biased fitness of the solution

  // Measuring cost and feasibility of an Individual from the information of chromR (needs chromR filled and access to Params)
  void evaluateCompleteCost(const Params & params);

  void exportCVRPLibFormat(std::string fileName);

  // Constructor of a random individual containing only a giant tour with a shuffled visit order
  Individual(Params & params);

  // Constructor of an individual from a file in CVRPLib solution format as produced by the algorithm (useful if a user wishes to input an initial solution)
  Individual(Params & params, std::string fileName);
};
#endif
