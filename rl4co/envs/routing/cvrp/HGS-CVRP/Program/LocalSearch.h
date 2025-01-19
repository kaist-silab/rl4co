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

#ifndef LOCALSEARCH_H
#define LOCALSEARCH_H

#include "Individual.h"

struct Node ;

// Structure containing a route
struct Route
{
	int cour;							// Route index
	int nbCustomers;					// Number of customers visited in the route
	int whenLastModified;				// "When" this route has been last modified
	int whenLastTestedSWAPStar;			// "When" the SWAP* moves for this route have been last tested
	Node * depot;						// Pointer to the associated depot
	double duration;					// Total time on the route
	double load;						// Total load on the route
	double reversalDistance;			// Difference of cost if the route is reversed
	double penalty;						// Current sum of load and duration penalties
	double polarAngleBarycenter;		// Polar angle of the barycenter of the route
	CircleSector sector;				// Circle sector associated to the set of customers
};

struct Node
{
	bool isDepot;						// Tells whether this node represents a depot or not
	int cour;							// Node index
	int position;						// Position in the route
	int whenLastTestedRI;				// "When" the RI moves for this node have been last tested
	Node * next;						// Next node in the route order
	Node * prev;						// Previous node in the route order
	Route * route;						// Pointer towards the associated route
	double cumulatedLoad;				// Cumulated load on this route until the customer (including itself)
	double cumulatedTime;				// Cumulated time on this route until the customer (including itself)
	double cumulatedReversalDistance;	// Difference of cost if the segment of route (0...cour) is reversed (useful for 2-opt moves with asymmetric problems)
	double deltaRemoval;				// Difference of cost in the current route if the node is removed (used in SWAP*)
};

// Structure used in SWAP* to remember the three best insertion positions of a customer in a given route
struct ThreeBestInsert
{
	int whenLastCalculated;
	double bestCost[3];
	Node * bestLocation[3];

	void compareAndAdd(double costInsert, Node * placeInsert)
	{
		if (costInsert >= bestCost[2]) return;
		else if (costInsert >= bestCost[1])
		{
			bestCost[2] = costInsert; bestLocation[2] = placeInsert;
		}
		else if (costInsert >= bestCost[0])
		{
			bestCost[2] = bestCost[1]; bestLocation[2] = bestLocation[1];
			bestCost[1] = costInsert; bestLocation[1] = placeInsert;
		}
		else
		{
			bestCost[2] = bestCost[1]; bestLocation[2] = bestLocation[1];
			bestCost[1] = bestCost[0]; bestLocation[1] = bestLocation[0];
			bestCost[0] = costInsert; bestLocation[0] = placeInsert;
		}
	}

	// Resets the structure (no insertion calculated)
	void reset()
	{
		bestCost[0] = 1.e30; bestLocation[0] = NULL;
		bestCost[1] = 1.e30; bestLocation[1] = NULL;
		bestCost[2] = 1.e30; bestLocation[2] = NULL;
	}

	ThreeBestInsert() { reset(); };
};

// Structured used to keep track of the best SWAP* move
struct SwapStarElement
{
	double moveCost = 1.e30 ;
	Node * U = NULL ;
	Node * bestPositionU = NULL;
	Node * V = NULL;
	Node * bestPositionV = NULL;
};

// Main local learch structure
class LocalSearch
{

private:
	
	Params & params ;							// Problem parameters
	bool searchCompleted;						// Tells whether all moves have been evaluated without success
	int nbMoves;								// Total number of moves (RI and SWAP*) applied during the local search. Attention: this is not only a simple counter, it is also used to avoid repeating move evaluations
	std::vector < int > orderNodes;				// Randomized order for checking the nodes in the RI local search
	std::vector < int > orderRoutes;			// Randomized order for checking the routes in the SWAP* local search
	std::set < int > emptyRoutes;				// indices of all empty routes
	int loopID;									// Current loop index

	/* THE SOLUTION IS REPRESENTED AS A LINKED LIST OF ELEMENTS */
	std::vector < Node > clients;				// Elements representing clients (clients[0] is a sentinel and should not be accessed)
	std::vector < Node > depots;				// Elements representing depots
	std::vector < Node > depotsEnd;				// Duplicate of the depots to mark the end of the routes
	std::vector < Route > routes;				// Elements representing routes
	std::vector < std::vector < ThreeBestInsert > > bestInsertClient;   // (SWAP*) For each route and node, storing the cheapest insertion cost 

	/* TEMPORARY VARIABLES USED IN THE LOCAL SEARCH LOOPS */
	// nodeUPrev -> nodeU -> nodeX -> nodeXNext
	// nodeVPrev -> nodeV -> nodeY -> nodeYNext
	Node * nodeU ;
	Node * nodeX ;
    Node * nodeV ;
	Node * nodeY ;
	Route * routeU ;
	Route * routeV ;
	int nodeUPrevIndex, nodeUIndex, nodeXIndex, nodeXNextIndex ;	
	int nodeVPrevIndex, nodeVIndex, nodeYIndex, nodeYNextIndex ;	
	double loadU, loadX, loadV, loadY;
	double serviceU, serviceX, serviceV, serviceY;
	double penaltyCapacityLS, penaltyDurationLS ;
	bool intraRouteMove ;

	void setLocalVariablesRouteU(); // Initializes some local variables and distances associated to routeU to avoid always querying the same values in the distance matrix
	void setLocalVariablesRouteV(); // Initializes some local variables and distances associated to routeV to avoid always querying the same values in the distance matrix

	inline double penaltyExcessDuration(double myDuration) {return std::max<double>(0., myDuration - params.durationLimit)*penaltyDurationLS;}
	inline double penaltyExcessLoad(double myLoad) {return std::max<double>(0., myLoad - params.vehicleCapacity)*penaltyCapacityLS;}

	/* RELOCATE MOVES */
	// (Legacy notations: move1...move9 from Prins 2004)
	bool move1(); // If U is a client node, remove U and insert it after V
	bool move2(); // If U and X are client nodes, remove them and insert (U,X) after V
	bool move3(); // If U and X are client nodes, remove them and insert (X,U) after V

	/* SWAP MOVES */
	bool move4(); // If U and V are client nodes, swap U and V
	bool move5(); // If U, X and V are client nodes, swap (U,X) and V
	bool move6(); // If (U,X) and (V,Y) are client nodes, swap (U,X) and (V,Y) 
	 
	/* 2-OPT and 2-OPT* MOVES */
	bool move7(); // If route(U) == route(V), replace (U,X) and (V,Y) by (U,V) and (X,Y)
	bool move8(); // If route(U) != route(V), replace (U,X) and (V,Y) by (U,V) and (X,Y)
	bool move9(); // If route(U) != route(V), replace (U,X) and (V,Y) by (U,Y) and (V,X)

	/* SUB-ROUTINES FOR EFFICIENT SWAP* EVALUATIONS */
	bool swapStar(); // Calculates all SWAP* between routeU and routeV and apply the best improving move
	double getCheapestInsertSimultRemoval(Node * U, Node * V, Node *& bestPosition); // Calculates the insertion cost and position in the route of V, where V is omitted
	void preprocessInsertions(Route * R1, Route * R2); // Preprocess all insertion costs of nodes of route R1 in route R2

	/* ROUTINES TO UPDATE THE SOLUTIONS */
	static void insertNode(Node * U, Node * V);		// Solution update: Insert U after V
	static void swapNode(Node * U, Node * V) ;		// Solution update: Swap U and V							   
	void updateRouteData(Route * myRoute);			// Updates the preprocessed data of a route

	public:

	// Run the local search with the specified penalty values
	void run(Individual & indiv, double penaltyCapacityLS, double penaltyDurationLS, int count=INT_MAX);

	// Loading an initial solution into the local search
	void loadIndividual(const Individual & indiv);

	// Exporting the LS solution into an individual and calculating the penalized cost according to the original penalty weights from Params
	void exportIndividual(Individual & indiv);

	// Constructor
	LocalSearch(Params & params);
};

#endif
