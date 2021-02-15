import sys
from typing import Optional, Dict, Union
import dataclasses

import ecole
import ecole.typing
import pyscipopt.scip as scip

import ecole_vs_gasse.observation


def get_instance_features(model: ecole.scip.Model) -> Dict[str, Union[int, float, str]]:
    """Extract some static information about the instance."""
    # Ensure we do not modify input model
    model = model.copy_orig()
    # Get model to the root note to extract root node info
    dyn = ecole.dynamics.BranchingDynamics()
    dyn.reset_dynamics(model)
    pyscipopt_model = model.as_pyscipopt()
    is_solving = pyscipopt_model.getStage() == scip.PY_SCIP_STAGE.SOLVING
    return {
        "n_vars": pyscipopt_model.getNVars(),
        "n_cons": pyscipopt_model.getNConss(),
        "root_nnz": pyscipopt_model.getNNZs() if is_solving else 0,
        "root_n_cols": pyscipopt_model.getNLPCols() if is_solving else 0,
        "root_n_rows": pyscipopt_model.getNLPRows() if is_solving else 0,
        "name": pyscipopt_model.getProbName(),
    }


def make_information_functions() -> Dict[str, ecole.typing.InformationFunction]:
    """Create the information function used in benchmarking the observation.

    This is a combination of sloving features such as number of nodes, and the timing of
    observation functions.
    """
    information_funcs = {
        "n_nodes": ecole.reward.NNodes().cumsum(),
        "n_lp_iterations": ecole.reward.LpIterations().cumsum(),
    }
    for module in (ecole, ecole_vs_gasse):
        for obs_func_name in ("NodeBipartite", "Khalil2016"):
            for wall in (True, False):
                clock_name = "wall_time_s" if wall else "cpu_time_s"
                name = f"{module.__name__}:{obs_func_name}:{clock_name}"
                obs_func = getattr(module.observation, obs_func_name)()
                # TimedFunction.cumsum is not available so we create it manually
                information_funcs[name] = ecole.reward.Cumulative(
                    ecole.data.TimedFunction(obs_func, wall=wall), lambda x, y: x + y, 0.0, "CumSum"
                )

    return information_funcs


def benchmark_observations(model: ecole.scip.Model) -> Dict[str, Union[int, float, str]]:
    """Benchmark the observation function in a branching environment.

    The observation function are benchmark on a single episode of the branching enviromnent
    on the given model.
    """
    env = ecole.environment.Branching(
        observation_function=None,
        information_function=make_information_functions(),
    )

    _, action_set, _, done, info = env.reset(model)
    while not done:
        _, action_set, _, done, info = env.step(action_set[0])

    return {**get_instance_features(model), **info}


def make_generators():
    """Create the instance generators used for benchmarking."""
    return (
        ecole.instance.SetCoverGenerator(n_rows=500, n_cols=1000),
        ecole.instance.SetCoverGenerator(n_rows=1000, n_cols=1000),
        ecole.instance.SetCoverGenerator(n_rows=2000, n_cols=1000),
        ecole.instance.CombinatorialAuctionGenerator(n_items=100, n_bids=500),
        ecole.instance.CombinatorialAuctionGenerator(n_items=200, n_bids=1000),
        ecole.instance.CombinatorialAuctionGenerator(n_items=300, n_bids=1500),
        ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=100, n_facilities=100),
        ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=200, n_facilities=100),
        ecole.instance.CapacitatedFacilityLocationGenerator(n_customers=400, n_facilities=100),
        ecole.instance.IndependentSetGenerator(n_nodes=500, graph_type="erdos_renyi"),
        ecole.instance.IndependentSetGenerator(n_nodes=1000, graph_type="erdos_renyi"),
        ecole.instance.IndependentSetGenerator(n_nodes=1500, graph_type="erdos_renyi"),
    )


@dataclasses.dataclass
class CsvPrinter:
    """A class to print a dictionnary as csv.

    Automatically print the title on the first call.
    """

    first: bool = True

    def print(self, d: dict) -> None:
        """Print the dictionnary as csv."""
        # Print the title on first line
        if self.first:
            print(",".join(f'"{key}"' for key in d.keys()))
            self.first = False
        # Print the values
        print(",".join(f'"{val}"' for val in d.values()))


def main(instances_per_generator: int, node_limit: int, seed: Optional[int] = None) -> None:
    """Repeatedly benchmark the observation functions on generated instances."""
    if seed is not None:
        ecole.seed(seed)
    generators = make_generators()
    printer = CsvPrinter()

    for i in range(instances_per_generator):
        for gen in generators:
            try:
                model = next(gen)
                model.disable_presolve()
                model.disable_cuts()
                model.set_param("limits/totalnodes", node_limit)
                printer.print(benchmark_observations(model))
            except Exception as e:
                print(e, file=sys.stderr)
                raise e


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--instances-per-generator",
        "--ipg",
        type=int,
        default=100,
        help="Number of instances generated by each instance generator",
    )
    parser.add_argument("--node-limit", "--nl", type=int, default=100, help="Limit the number of nodes in each run")
    parser.add_argument("--seed", "-s", type=int, help="Global Ecole random seed")
    args = parser.parse_args()
    main(**vars(args))
