# SPDX-License-Identifier: Apache-2.0
"""Benchmark the latency of processing a single batch of requests."""

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import PromptType
from vllm.sampling_params import BeamSearchParams
from vllm.utils import FlexibleArgumentParser

# NOTE(woosuk): If the request cannot be processed in a single batch, the engine will automatically process the request 
# in multiple batches.

CORRECTNESS_PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
    "The causes of the revolution were a combination of social, political, and economic factors which the ancien régime"
    " ('old regime') proved unable to manage. A financial crisis and widespread social distress led to the convocation "
    "of the Estates General in May 1789, its first meeting since 1614. The representatives of the Third Estate broke "
    "away and re-constituted themselves as a National Assembly in June. The Storming of the Bastille in Paris on 14 "
    "July was followed by radical measures by the Assembly, among them the abolition of feudalism, state control over "
    "the Catholic Church, and a declaration of rights. The next three years were dominated by a struggle for political "
    "control.",
    "The greatest river in Europe is",
    "The Persian Empire was the largest empire ",
    "The founder of Babylon is",
    "The first record of a human being in the Americas was found in",
    "The average size of a dog is",
    "Knock knock. Who's there?",
]

class LLMBenchmarker:

    def __init__(self, args: argparse.Namespace):
        """Initialize the LLMBenchmarker with the parsed (args)."""
        # Create the LLM instance
        self.args = args
        self.llm = LLM(**dataclasses.asdict(EngineArgs.from_cli_args(args)))
        # Create the sampling and beam search parameters
        if args.use_beam_search:
            self.sampling_params = None
            self.beam_search_params = BeamSearchParams(beam_width=args.n, max_tokens=args.output_len, ignore_eos=True)
        else:
            self.sampling_params = SamplingParams(
                n=args.n, temperature=1.0, top_p=1.0, ignore_eos=True, max_tokens=args.output_len
            )
            self.beam_search_params = None
        print(f"Sampling params: {self.sampling_params}\nBeam search params: {self.beam_search_params}")
        # Create the output directory
        if self.args.results_dir is None:
            self.results_dir = (Path(".") / "vllm_benchmark_result" / f"results_{time.time()}")
        else:
            self.results_dir = Path(self.args.results_dir)
        print(f"Results will be saved to '{self.results_dir}'...")
        os.makedirs(self.results_dir, exist_ok=True)

    def llm_generate(self, batch_size: int, input_len: int, output_len: int) -> None:
        """Generate a batch of dummy prompts and run the model on them."""
        # Create data
        dummy_prompt_token_ids = np.random.randint(10000, size=(batch_size, input_len))
        dummy_prompts: List[PromptType] = [{"prompt_token_ids": batch} for batch in dummy_prompt_token_ids.tolist()]
        # Run model
        if self.beam_search_params is not None:
            self.beam_search_params.max_tokens = output_len
            self.llm.beam_search(dummy_prompts, self.beam_search_params)
        else: 
            self.sampling_params.max_tokens = output_len
            self.llm.generate(dummy_prompts, sampling_params=self.sampling_params, use_tqdm=False)

    def time_one_generate(self, batch_size: int, input_len: int, output_len: int) -> float:
        """Time the generation of a single batch of prompts."""
        start_time = time.perf_counter()
        self.llm_generate(batch_size, input_len, output_len)
        end_time = time.perf_counter()
        return end_time - start_time

    def warmup(self, batch_size: int, input_len: int, output_len: int) -> None:
        """Warm up the model by running a few iterations."""
        print("Warming up...")
        for _ in tqdm(range(self.args.num_iters_warmup), desc="Warmup iterations"):
            self.time_one_generate(batch_size, input_len, output_len)

    def check_correctness(self, output_length: int) -> None:
        print("Checking correctness...")
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=output_length)
        outputs = self.llm.generate(CORRECTNESS_PROMPTS, sampling_params)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    def profile(self, batch_size: int, input_len: int, output_len: int) -> None:
        """Profile the generation process of a single batch."""
        # Prepare profiler
        profile_context = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            # on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir)),
        )
        # Run profiling
        with profile_context as p:
            self.llm_generate(batch_size, input_len, output_len)
        # Display results
        table = p.key_averages().table(sort_by="self_cuda_time_total")
        print(table)
        # Save results
        with open(self.results_dir / f"profile_bs{batch_size}_il{input_len}_ol{output_len}.txt", "w") as f:
            f.write(table)
        p.export_chrome_trace(str(self.results_dir / f"profile_bs{batch_size}_il{input_len}_ol{output_len}.json"))

    def benchmark(self, batch_size: int, input_len: int, output_len: int) -> None:
        """Benchmark the latency of processing a single batch of requests."""
        # Gather latencies
        latencies = []
        for _ in tqdm(range(self.args.num_iters), desc="Profiling iterations"):
            latencies.append(self.time_one_generate(batch_size, input_len, output_len))
        # Compute percentiles
        latencies = np.array(latencies)
        percentages = [10, 25, 50, 75, 90, 99]
        percentiles = np.percentile(latencies, percentages)
        # Display results
        print(f"Settings: {batch_size = }, {input_len = }, {output_len = }")
        print(f"Avg latency: {np.mean(latencies)} seconds")
        for percentage, percentile in zip(percentages, percentiles):
            print(f"{percentage}% percentile latency: {percentile} seconds")
        # Output JSON results
        results_file = self.results_dir / f"results_bs{batch_size}_il{input_len}_ol{output_len}.json"
        results = {
            "avg_latency": np.mean(latencies),
            "latencies": latencies.tolist(),
            "percentiles": dict(zip(percentages, percentiles.tolist())),
        }
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)


if __name__ == "__main__":
    # Prepare parser
    parser = FlexibleArgumentParser(
        description="Benchmark the latency of processing a single batch of "
        "requests till completion.")
    parser.add_argument("--input-len", nargs="+", type=int, default=[32])
    parser.add_argument("--output-len", type=int, default=1)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8])
    parser.add_argument("--n", type=int, default=1, help="Number of generated sequences per prompt.")
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument("--num-iters-warmup", type=int, default=3, help="Number of iterations to run for warmup.")
    parser.add_argument("--num-iters", type=int, default=20, help="Number of iterations to run.")
    parser.add_argument("--profile", action="store_true", help="profile the generation process of a single batch")
    parser.add_argument("--results-dir", type=str, default=None, 
                        help="Path to save the latency results in JSON format or the profiling results.")

    # Parse arguments
    parser = EngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    # Initialize LLMBenchmarker check correctness
    llm_benchmarker = LLMBenchmarker(args)
    llm_benchmarker.check_correctness(output_length=20)  

    # For each batch size, profile or benchmark
    for input_len in args.input_len:
        for batch_size in args.batch_sizes:
            llm_benchmarker.warmup(batch_size=batch_size, input_len=input_len, output_len=args.output_len)
            llm_benchmarker.benchmark(batch_size=batch_size, input_len=input_len, output_len=args.output_len)
            if args.profile:
                llm_benchmarker.profile(batch_size=batch_size, input_len=input_len, output_len=args.output_len)
