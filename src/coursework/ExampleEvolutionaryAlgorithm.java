package coursework;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import model.Fitness;
import model.Individual;
import model.LunarParameters.DataSet;
import model.NeuralNetwork;

/**
 * Implements a basic Evolutionary Algorithm to train a Neural Network
 * 
 * You Can Use This Class to implement your EA or implement your own class that extends {@link NeuralNetwork} 
 * 
 */
public class ExampleEvolutionaryAlgorithm extends NeuralNetwork 
{
	double randomForScaler()
	{	
		long seed = System.currentTimeMillis();
		Random random = new Random(seed);
		return random.nextDouble();
	}

	/**
	 * The Main Evolutionary Loop
	 */
	@Override
	public void run() {
		//Initialise a population of Individuals with random weights
		population = initialise();

		//Record a copy of the best Individual in the population
		best = getBest();
		System.out.println("Best From Initialisation " + best);

		/**
		 * main EA processing loop
		 */
		
		double sum_fitness = 0.0f;
		// Initialize new arraylist to hold the new fitness respective to the index of the individual it comes from
		// ( so index 1 of new_fitnessess is the temporary new fitness we use to calculate the new probability ) 
		
		while (evaluations < Parameters.maxEvaluations) 
		{
			long seed = System.currentTimeMillis();
			Random rand_eng = new Random(seed);
			
			/**
			 * this is a skeleton EA - you need to add the methods.
			 * You can also change the EA if you want 
			 * You must set the best Individual at the end of a run
			 * 
			 */
			// Recalculate sum of fitnesses
			sum_fitness = sumFitnesses(population);
			
			// Select 2 Individuals from the current population. RouletteWheelSelection
			int parent1_index = SelectRouletteWheel(sum_fitness);
			Individual parent1 = population.get(parent1_index);
			int parent2_index = 0;
			while(true)
			{
				parent2_index = SelectRouletteWheel(sum_fitness);
				
				if(parent2_index != parent1_index)
				{
					break;
				}
			}
			Individual parent2 = population.get(parent2_index);
			
			System.out.println("Parent 1 selection is " + parent1);
			System.out.println("Parent 2 selection is " + parent2);

			// Generate a child by crossover. Not Implemented			
			Individual new_child = reproduce(parent1, parent2);			
			
			//mutate the offspring
			mutate(new_child);
			
			// Evaluate the children
			evaluateIndividuals(new_child);			

			// Replace children in population
			replace(new_child);

			// check to see if the best has improved
			best = getBest();
			
			// Implemented in NN class. 
			outputStats();
			
			//Increment number of completed generations			
		}

		//save the trained network to disk
		saveNeuralNetwork();
	}

	

	/**
	 * Sets the fitness of the individuals passed as parameters (whole population)
	 * 
	 */
	private void evaluateIndividuals(Individual individual) {

			individual.fitness = Fitness.evaluate(individual, this);

	}


	/**
	 * Returns a copy of the best individual in the population
	 * 
	 */
	private Individual getBest() {
		best = null;;
		for (Individual individual : population) {
			if (best == null) {
				best = individual.copy();
			} else if (individual.fitness < best.fitness) {
				best = individual.copy();
			}
		}
		return best;
	}

	/**
	 * Generates a randomly initialised population
	 * 
	 */
	private ArrayList<Individual> initialise() {
		population = new ArrayList<>();
		for (int i = 0; i < Parameters.popSize; ++i) {
			//chromosome weights are initialised randomly in the constructor
			Individual individual = new Individual();
			population.add(individual);
		}
		for(Individual indiv : population)
		{
			evaluateIndividuals(indiv);
		}
		return population;
	}

	/**
	 * Selection --
	 * 
	 * NEEDS REPLACED with proper selection this just returns a copy of a random
	 * member of the population
	 */
	private Individual select() 
	{
		Individual parent = population.get(Parameters.random.nextInt(Parameters.popSize));
		return parent.copy();	
	}
	
	private double sumFitnesses(ArrayList<Individual> pop)
	{
		
		// Accumulative Fitness
		double sum_all = 0.0f;
		for(int i = 0; i < pop.size(); i++)
		{
			sum_all += pop.get(i).fitness;
		}
		return sum_all;
	}
	
	private int SelectRouletteWheel(double sum_all) // change return type to Individual
	{
		ArrayList<Individual> new_probs = new ArrayList<Individual>();
		// Roulette Wheel Selection
		// Roulette Wheel doesn't work by default with a minimization problem
		// So we need to find the inverse fitness probability
		for(int i = 0; i < population.size(); i++)
		{
			double new_fit = 1 - population.get(i).fitness;
			
			Individual temp = new Individual();
			temp.fitness = new_fit / sum_all;
			
			new_probs.add(temp);
		}
		double new_sum = sumFitnesses(new_probs);
		
		double selector = Parameters.random.nextDouble() * new_sum;
		
		for(int j = 0; j < new_probs.size(); j++)
		{
			selector -= new_probs.get(j).fitness;
			
			if(selector < 0)
			{
				return j;
			}
		}
		
		new_probs.clear();
		
		// Due to rounding errors its possible to get past the last index, in this case return the last index
		return population.size() - 1;
	}

	/**
	 * Crossover / Reproduction
	 * 
	 * Explained methodology in comments - Single Point Crossover
	 */
	private Individual reproduce(Individual parent1, Individual parent2) 
	{
		// Seed random engine
		long seed = System.currentTimeMillis();
		Random rand = new Random(seed);
		
		// Find a random gene index between 0 and the max num of genes
		int random_index = rand.nextInt(parent1.chromosome.length - 1);
		// Create child container
		Individual child = new Individual();
		// Initialize start index
		int start;
		// Iterate through parent 1 and copy genes to child chromosome up to index
		for(start = 0; start < random_index; start++)
		{
			child.chromosome[start] = parent1.chromosome[start];
		}
		// Iterate through parent 2 and copy genes to child chromosome from index to end
		for(int end = start; end < child.chromosome.length - 1; end++)
		{
			child.chromosome[end] = parent2.chromosome[end];
		}	
		// Return single child offspring
		return child;
	} 
	
	/**
	 * Mutation
	 * 
	 * 
	 */
	private void mutate(Individual individual)
	{
		for (int i = 0; i < individual.chromosome.length; i++) 
		{
			if (Parameters.random.nextDouble() < Parameters.mutateRate) 
			{
				if (Parameters.random.nextBoolean()) 
				{
					individual.chromosome[i] += (Parameters.mutateChange);
				} 
				else 
				{
					individual.chromosome[i] -= (Parameters.mutateChange);
				}
			}
		}
	}

	/**
	 * 
	 * Replaces the worst member of the population 
	 * (regardless of fitness)
	 * 
	 */
	private void replace(Individual individual) 
	{
		// Get index of the current worst fitness in population
		int idx = getWorstIndex();		
		// Replace that index individual with the new individual - irrespective of whether new > old
		population.set(idx, individual);	
	}

	

	/**
	 * Returns the index of the worst member of the population
	 * @return
	 */
	private int getWorstIndex() {
		Individual worst = null;
		int idx = -1;
		for (int i = 0; i < population.size(); i++) {
			Individual individual = population.get(i);
			if (worst == null) {
				worst = individual;
				idx = i;
			} else if (individual.fitness > worst.fitness) {
				worst = individual;
				idx = i; 
			}
		}
		return idx;
	}	

	@Override
	public double activationFunction(double x) {
		if (x < -20.0) {
			return -1.0;
		} else if (x > 20.0) {
			return 1.0;
		}
		return Math.tanh(x);
	}
}
