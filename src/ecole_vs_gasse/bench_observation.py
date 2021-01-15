import google_benchmark as benchmark

import ecole

import ecole_vs_gasse.utils
import ecole_vs_gasse.observation


def benchmark_observation(func_to_bench):
    @benchmark.register(name=func_to_bench.__class__.__name__)
    @benchmark.option.measure_process_cpu_time()
    @benchmark.option.use_real_time()
    @benchmark.option.unit(benchmark.kMillisecond)
    def _benchmark(state):
        n_nodes = 0
        while state:
            state.pause_timing()
            model = ecole_vs_gasse.utils.get_model()

            state.resume_timing()
            func_to_bench.before_reset(model)

            state.pause_timing()
            model.solve_iter()

            while not model.solve_iter_is_done():
                state.resume_timing()
                func_to_bench.extract(model, False)

                state.pause_timing()
                model.solve_iter_branch(model.lp_branch_cands[0])

            n_nodes += model.as_pyscipopt().getNNodes()  # FIXME different from C++
            state.resume_timing()

        state.counters["Nodes"] = benchmark.Counter(n_nodes, benchmark.Counter.kAvgIterations)

    return _benchmark


if __name__ == "__main__":
    # Google benchmark quirk: needs to exists for the whole program
    extend_lifetime = [
        benchmark_observation(ecole.observation.NodeBipartite()),
        benchmark_observation(ecole_vs_gasse.observation.NodeBipartite_L2B()),
        benchmark_observation(ecole.observation.Khalil2016()),
        benchmark_observation(ecole_vs_gasse.observation.Khalil2016_L2B()),
    ]
    benchmark.main()
