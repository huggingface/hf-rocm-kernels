import torch
from typing import Callable, Dict
from triton.testing import do_bench, do_bench_cudagraph
import tabulate
import warnings
import itertools


def randomize_tensors(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().float().normal_().to(x.dtype)
    return x

def benchmark_cuda_graph_no_cache(
    fn, args, kwargs, 
    graph_size: int = 8,
    warmups: int = 32,
    iterations: int = 128,
    device: str = "cuda",
) -> float:
    # Create inputs
    list_of_args = [
        tuple(map(randomize_tensors, args)) 
        for _ in range(graph_size)
    ]
    list_of_kwargs = [
        {k: randomize_tensors(v) for k, v in kwargs.items()}
        for _ in range(graph_size)
    ]

    # Create a side-stream to benchmark in
    stream = torch.cuda.Stream(device)
    with torch.cuda.stream(stream):

        # Create graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for i in range(graph_size):
                fn(*list_of_args[i], **list_of_kwargs[i])
        torch.cuda.synchronize()

        # Benchmark its replays
        t = 0
        for i in range(warmups + iterations):

            # Prepare events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # Clear cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            # Run
            start_event.record()
            graph.replay()
            end_event.record()

            # Accumulate
            torch.cuda.synchronize()
            if i > warmups:
                t += start_event.elapsed_time(end_event)

    # Post-process time
    t *= 1000 / (iterations * graph_size)
    return t


class Bench: 
    """An object to benchmark different version of the same operator."""

    def __init__(self) -> None:
        self.enter_benchmark_mode()
        self.check_device_is_free(warn_if_not=True)
        self._measures = {}

    def enter_benchmark_mode(self) -> None:
        """Turns off gradient and activates TunableOps to get maximum performances."""
        torch.set_grad_enabled(False)
        torch.cuda.tunable.enable(val=True)

    def check_device_is_free(self, warn_if_not: bool = False) -> Dict[str, int]:
        """Checks whether or not the GPU is free and returns the relevant metrics."""
        metrics = {
            "utilization": torch.cuda.utilization(),
            "memory_usage": torch.cuda.memory_usage(),
        }
        over_the_limit = any([
            metrics["utilization"] > 0,
            metrics["memory_usage"] > 784,
        ])
        if warn_if_not and over_the_limit:
            warnings.warn(f"Device {torch.cuda.current_device()} is not fully free: {metrics = }")
        return metrics

    def benchmark_fn(self, fn: Callable[[], None], rep: int = 100, cg: bool = True) -> float:
        """Benchmarks a function using triton's cuda graphs's benchmark."""
        if cg:
            with torch.cuda.stream(torch.cuda.Stream()):
                t = do_bench_cudagraph(fn, rep=rep)
        else:
            t = do_bench(fn, warmup=rep, rep=5*rep)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return t * 1e3
    
    def add_measure(self, header: str, label: int, fn: Callable[[], None], rep: int = 100) -> float:
        t = self.benchmark_fn(fn, rep)
        self.add_raw_measure(header, label, t)

    def add_raw_measure(self, header: str, label: int, measure: float) -> float:
        """Add a measure to the benchmark table in the (header) column at the given (row)."""
        if header not in self._measures:
            self._measures[header] = {}
        self._measures[header][label] = measure

    def add_speedup_column(self, ref_header: str, our_header: str) -> None:
        """Add a speedup column to the benchmark table."""
        if ref_header not in self._measures or our_header not in self._measures:
            raise ValueError(f"Headers {ref_header} and {our_header} must be present in the benchmark table.")
        # Add the speedup column
        for label in self._measures[ref_header].keys():
            self.add_raw_measure(f"Speedup", label, self._measures[ref_header][label] / self._measures[our_header][label])

    def display_table(self, row_header: str) -> None:
        """Display the current benchmark table with a (row_header) for the row's column."""
        # Gather all rows
        rows = list(set(itertools.chain(*[m.keys() for m in self._measures.values()])))
        rows.sort()
        # Create the data to tabulate
        data = {row_header: rows}
        for header, measures in self._measures.items():
            data[header] = [measures.get(row, "-") for row in rows]
        # Print it out
        txt = tabulate.tabulate(data, list(data.keys()))
        print(txt)
