import tqdm
import multiprocessing as mp
import torch
import numpy as np
import torch.nn as nn
import cma
import argparse
import pickle
from solutions import LinearModelSolution, PermutationInvariantSolution
from tasks import Task

def save_solutions(folder, iteration, solver, solution_instance, solution_type):
	path = folder+"/"+f"{solution_type}{iteration}.pkl"

	with open(path, 'wb') as f:
		model = (solver, solution_instance)
		pickle.dump(model, f)


def load_checkpoint(args):
	with open(args.checkpoint, 'rb') as f:
		solver, solution_ = pickle.load(f)
	
	return (solver, solution_)


def prepare_solver(solution_instance, args):

	x0 = np.zeros(solution_instance.num_parameters())
	solver = cma.CMAEvolutionStrategy(
		x0=x0,
		sigma0=0.1,
		inopts={
			'popsize': args.population_size,
			'seed': 5,
			'randn': np.random.randn,
		},
	)

	return solver

def get_solution(args):
	if args.solution == "bls":
		config = {
			'num_layers': 3,
			'layer_sizes': [64, 128, 1],
		}

		return LinearModelSolution(
			config=config,
			num_features=4
		)

	elif args.solution == "pis":
		return PermutationInvariantSolution()
	
	else:
		raise ValueError("Invalid solution type, choose one of either 'bls' or 'pis'")

parser = argparse.ArgumentParser(
	"Training the agents on the evolutionary strategy", 
	formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument(
	"solution",
	type=str,
	choices=("bls", "pis")
)

parser.add_argument(
	"--checkpoint",
	type=str,
	help="checkpoint location containing both the solver and the solution"
)

parser.add_argument(
	"--env-seed",
	type=int,
	default=0,
	help="random seed controlling the gym environment"
)	

parser.add_argument(
	"--permutation-noise-seed",
	type=int,
	help="seed for controlling the permutation of the feature indices and noise generation"
)

parser.add_argument(
	"--render-video",
	type=bool,
	default=False,
	help="toggle video rendering during training"
)

parser.add_argument(
	"--max-iterations",
	type=int,
	default=1000,
	help="maximum number of iterations"
)

parser.add_argument(
	"--episodes"
	type=int,
	default=16,
	help="number of episodes to evaluate rewards for"
)

parser.add_argument(
	"--population-size",
	type=int,
	default=256,
	help="controls the popsize parameter in our evolutionary strategy"
)

parser.add_argument(
	"--n-jobs",
	type=int,
	default=-1,
	help="number of parallel jobs to run, by default set to run all jobs"
)

args = parser.parse_args()

solution = get_solution(args)

if args.checkpoint is not None:
	solver = prepare_solver(solution, args)

else:
	(solver, solution) = load_checkpoint(args)

task = Task()
fitness = task.rollout_episodes(solution, args.episodes)

if args.n_jobs  == -1:
	n_jobs = mp.cpu_count()

else:
	n_jobs = args.n_jobs

with mp.Pool(processes=n_jobs) as pool:
	for iteration in tqdm.tqdm(range(args.max_iterations)):
			try: 
				parameter_set = solver.ask()
				iterable = [solution.clone().set_parameters(p) for p in parameter_set]

				rewards = pool.map(fitness, iterable)
				positive_fitness = [np.mean(reward) for reward in rewards]
				negative_fitness = [-x for x in positive_fitness]

				all_parameters = np.concatenate(parameter_set)
				solver.tell(parameter_set, negative_fitness)

			except KeyboardInterrupt:
				save_solutions()
				print('run saved')
				break