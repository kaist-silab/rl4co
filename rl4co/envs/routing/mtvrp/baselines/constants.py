LKH_SCALING_FACTOR = 100_000
ORTOOLS_SCALING_FACTOR = 100_000
PYVRP_SCALING_FACTOR = 1_000


ROUTEFINDER2LKH = {
    "CVRP": "CVRP",
    "OVRP": "OVRP",
    "OVRPB": None,  # Issue: don't know
    "OVRPBL": None,  # Issue: distance limits
    "OVRPBLTW": None,  # Issue: distance limits
    "OVRPBTW": None,  # Issue: service times don't work in VRPBTW
    "OVRPL": "OVRP",
    "OVRPLTW": "CVRPTW",
    "OVRPMB": "VRPMPD",
    "OVRPMBL": "VRPMPD",
    "OVRPMBTW": "VRPMPDTW",
    "OVRPMBLTW": "VRPMPDTW",  # Issue: distance limits
    "OVRPTW": "CVRPTW",
    "VRPB": None,  # Issue: don't know: linehaul after backhaul
    "VRPBL": None,
    "VRPBLTW": None,  # Issue: service times don't work in VRPBTW
    "VRPBTW": None,  # Issue: service times don't work in VRPBTW
    "VRPL": "DCVRP",
    "VRPLTW": "CVRPTW",  # I don't think that limits get respected
    "VRPMB": "VRPMPD",
    "VRPMBL": "VRPMPD",  # I don't think that limits get respected
    "VRPMBTW": "VRPMPDTW",
    "VRPMBLTW": None,  # Issue: don't know
    "VRPTW": "CVRPTW",
}

LKH_VARIANTS = [
    "CVRP",
    "OVRP",
    "CVRPTW",
    "DCVRP",
    "VRPB",
    "VRPBTW",
    "VRPMPD",
    "VRPMPDTW",
]
