//
// Created by chkwon on 3/23/22.
//

#ifndef C_INTERFACE_H
#define C_INTERFACE_H
#include "AlgorithmParameters.h"

struct SolutionRoute
{
	int length;
	int * path;
};

struct Solution
{
	double cost;
	double time;
	int n_routes;
	struct SolutionRoute * routes;
};

#ifdef __cplusplus
extern "C"
#endif
struct Solution * solve_cvrp(
	int n, double* x, double* y, double* serv_time, double* dem,
	double vehicleCapacity, double durationLimit, char isRoundingInteger, char isDurationConstraint,
	int max_nbVeh, const struct AlgorithmParameters* ap, char verbose);

#ifdef __cplusplus
extern "C"
#endif
struct Solution *solve_cvrp_dist_mtx(
	int n, double* x, double* y, double *dist_mtx, double *serv_time, double *dem,
	double vehicleCapacity, double durationLimit, char isDurationConstraint,
	int max_nbVeh, const struct AlgorithmParameters *ap, char verbose);

#ifdef __cplusplus
extern "C"
#endif
int local_search(
	int n, double* x, double* y, double *dist_mtx, double *serv_time, double *dem,
	double vehicleCapacity, double durationLimit, char isDurationConstraint,
	int max_nbVeh, const struct AlgorithmParameters *ap, char verbose, int callid, int count);

#ifdef __cplusplus
extern "C"
#endif
void delete_solution(struct Solution * sol);


#endif //C_INTERFACE_H
