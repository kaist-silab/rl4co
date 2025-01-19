#include "Split.h" 

void Split::generalSplit(Individual & indiv, int nbMaxVehicles)
{
	// Do not apply Split with fewer vehicles than the trivial (LP) bin packing bound
	maxVehicles = std::max<int>(nbMaxVehicles, std::ceil(params.totalDemand/params.vehicleCapacity));

	// Initialization of the data structures for the linear split algorithms
	// Direct application of the code located at https://github.com/vidalt/Split-Library
	for (int i = 1; i <= params.nbClients; i++)
	{
		cliSplit[i].demand = params.cli[indiv.chromT[i - 1]].demand;
		cliSplit[i].serviceTime = params.cli[indiv.chromT[i - 1]].serviceDuration;
		cliSplit[i].d0_x = params.timeCost[0][indiv.chromT[i - 1]];
		cliSplit[i].dx_0 = params.timeCost[indiv.chromT[i - 1]][0];
		if (i < params.nbClients) cliSplit[i].dnext = params.timeCost[indiv.chromT[i - 1]][indiv.chromT[i]];
		else cliSplit[i].dnext = -1.e30;
		sumLoad[i] = sumLoad[i - 1] + cliSplit[i].demand;
		sumService[i] = sumService[i - 1] + cliSplit[i].serviceTime;
		sumDistance[i] = sumDistance[i - 1] + cliSplit[i - 1].dnext;
	}

	// We first try the simple split, and then the Split with limited fleet if this is not successful
	if (splitSimple(indiv) == 0)
		splitLF(indiv);

	// Build up the rest of the Individual structure
	indiv.evaluateCompleteCost(params);
}

int Split::splitSimple(Individual & indiv)
{
	// Reinitialize the potential structures
	potential[0][0] = 0;
	for (int i = 1; i <= params.nbClients; i++)
		potential[0][i] = 1.e30;

	// MAIN ALGORITHM -- Simple Split using Bellman's algorithm in topological order
	// This code has been maintained as it is very simple and can be easily adapted to a variety of constraints, whereas the O(n) Split has a more restricted application scope
	if (params.isDurationConstraint)
	{
		for (int i = 0; i < params.nbClients; i++)
		{
			double load = 0.;
			double distance = 0.;
			double serviceDuration = 0.;
			for (int j = i + 1; j <= params.nbClients && load <= 1.5 * params.vehicleCapacity ; j++)
			{
				load += cliSplit[j].demand;
				serviceDuration += cliSplit[j].serviceTime;
				if (j == i + 1) distance += cliSplit[j].d0_x;
				else distance += cliSplit[j - 1].dnext;
				double cost = distance + cliSplit[j].dx_0
					+ params.penaltyCapacity * std::max<double>(load - params.vehicleCapacity, 0.)
					+ params.penaltyDuration * std::max<double>(distance + cliSplit[j].dx_0 + serviceDuration - params.durationLimit, 0.);
				if (potential[0][i] + cost < potential[0][j])
				{
					potential[0][j] = potential[0][i] + cost;
					pred[0][j] = i;
				}
			}
		}
	}
	else
	{
		Trivial_Deque queue = Trivial_Deque(params.nbClients + 1, 0);
		for (int i = 1; i <= params.nbClients; i++)
		{
			// The front is the best predecessor for i
			potential[0][i] = propagate(queue.get_front(), i, 0);
			pred[0][i] = queue.get_front();

			if (i < params.nbClients)
			{
				// If i is not dominated by the last of the pile
				if (!dominates(queue.get_back(), i, 0))
				{
					// then i will be inserted, need to remove whoever is dominated by i.
					while (queue.size() > 0 && dominatesRight(queue.get_back(), i, 0))
						queue.pop_back();
					queue.push_back(i);
				}
				// Check iteratively if front is dominated by the next front
				while (queue.size() > 1 && propagate(queue.get_front(), i + 1, 0) > propagate(queue.get_next_front(), i + 1, 0) - MY_EPSILON)
					queue.pop_front();
			}
		}
	}

	if (potential[0][params.nbClients] > 1.e29)
		throw std::string("ERROR : no Split solution has been propagated until the last node");

	// Filling the chromR structure
	for (int k = params.nbVehicles - 1; k >= maxVehicles; k--)
		indiv.chromR[k].clear();

	int end = params.nbClients;
	for (int k = maxVehicles - 1; k >= 0; k--)
	{
		indiv.chromR[k].clear();
		int begin = pred[0][end];
		for (int ii = begin; ii < end; ii++)
			indiv.chromR[k].push_back(indiv.chromT[ii]);
		end = begin;
	}

	// Return OK in case the Split algorithm reached the beginning of the routes
	return (end == 0);
}

// Split for problems with limited fleet
int Split::splitLF(Individual & indiv)
{
	// Initialize the potential structures
	potential[0][0] = 0;
	for (int k = 0; k <= maxVehicles; k++)
		for (int i = 1; i <= params.nbClients; i++)
			potential[k][i] = 1.e30;

	// MAIN ALGORITHM -- Simple Split using Bellman's algorithm in topological order
	// This code has been maintained as it is very simple and can be easily adapted to a variety of constraints, whereas the O(n) Split has a more restricted application scope
	if (params.isDurationConstraint) 
	{
		for (int k = 0; k < maxVehicles; k++)
		{
			for (int i = k; i < params.nbClients && potential[k][i] < 1.e29 ; i++)
			{
				double load = 0.;
				double serviceDuration = 0.;
				double distance = 0.;
				for (int j = i + 1; j <= params.nbClients && load <= 1.5 * params.vehicleCapacity ; j++) // Setting a maximum limit on load infeasibility to accelerate the algorithm
				{
					load += cliSplit[j].demand;
					serviceDuration += cliSplit[j].serviceTime;
					if (j == i + 1) distance += cliSplit[j].d0_x;
					else distance += cliSplit[j - 1].dnext;
					double cost = distance + cliSplit[j].dx_0
								+ params.penaltyCapacity * std::max<double>(load - params.vehicleCapacity, 0.)
								+ params.penaltyDuration * std::max<double>(distance + cliSplit[j].dx_0 + serviceDuration - params.durationLimit, 0.);
					if (potential[k][i] + cost < potential[k + 1][j])
					{
						potential[k + 1][j] = potential[k][i] + cost;
						pred[k + 1][j] = i;
					}
				}
			}
		}
	}
	else // MAIN ALGORITHM -- Without duration constraints in O(n), from "Vidal, T. (2016). Split algorithm in O(n) for the capacitated vehicle routing problem. C&OR"
	{
		Trivial_Deque queue = Trivial_Deque(params.nbClients + 1, 0);
		for (int k = 0; k < maxVehicles; k++)
		{
			// in the Split problem there is always one feasible solution with k routes that reaches the index k in the tour.
			queue.reset(k);

			// The range of potentials < 1.29 is always an interval.
			// The size of the queue will stay >= 1 until we reach the end of this interval.
			for (int i = k + 1; i <= params.nbClients && queue.size() > 0; i++)
			{
				// The front is the best predecessor for i
				potential[k + 1][i] = propagate(queue.get_front(), i, k);
				pred[k + 1][i] = queue.get_front();

				if (i < params.nbClients)
				{
					// If i is not dominated by the last of the pile 
					if (!dominates(queue.get_back(), i, k))
					{
						// then i will be inserted, need to remove whoever he dominates
						while (queue.size() > 0 && dominatesRight(queue.get_back(), i, k))
							queue.pop_back();
						queue.push_back(i);
					}

					// Check iteratively if front is dominated by the next front
					while (queue.size() > 1 && propagate(queue.get_front(), i + 1, k) > propagate(queue.get_next_front(), i + 1, k) - MY_EPSILON)
						queue.pop_front();
				}
			}
		}
	}

	if (potential[maxVehicles][params.nbClients] > 1.e29)
		throw std::string("ERROR : no Split solution has been propagated until the last node");

	// It could be cheaper to use a smaller number of vehicles
	double minCost = potential[maxVehicles][params.nbClients];
	int nbRoutes = maxVehicles;
	for (int k = 1; k < maxVehicles; k++)
		if (potential[k][params.nbClients] < minCost)
			{minCost = potential[k][params.nbClients]; nbRoutes = k;}

	// Filling the chromR structure
	for (int k = params.nbVehicles-1; k >= nbRoutes ; k--)
		indiv.chromR[k].clear();

	int end = params.nbClients;
	for (int k = nbRoutes - 1; k >= 0; k--)
	{
		indiv.chromR[k].clear();
		int begin = pred[k+1][end];
		for (int ii = begin; ii < end; ii++)
			indiv.chromR[k].push_back(indiv.chromT[ii]);
		end = begin;
	}

	// Return OK in case the Split algorithm reached the beginning of the routes
	return (end == 0);
}

Split::Split(const Params & params): params(params)
{
	// Structures of the linear Split
	cliSplit = std::vector <ClientSplit>(params.nbClients + 1);
	sumDistance = std::vector <double>(params.nbClients + 1,0.);
	sumLoad = std::vector <double>(params.nbClients + 1,0.);
	sumService = std::vector <double>(params.nbClients + 1, 0.);
	potential = std::vector < std::vector <double> >(params.nbVehicles + 1, std::vector <double>(params.nbClients + 1,1.e30));
	pred = std::vector < std::vector <int> >(params.nbVehicles + 1, std::vector <int>(params.nbClients + 1,0));
}
