#include "include.hpp"

#define N_INPUTS 3
#define N_OUTPUTS 1
#define POPULATION_SIZE 30

#define FITNESS_SAMPLES 3

void fitness_func(Genome& genome, uint32_t samples){
    // calculate fitness based on how good it is at calculating (N1 + N2 * N3)
    // N1, N2, N3 are random numbers between -0 and 20

    // this is just to test NEAT

    genome.fitness = 0.0f;

    for (uint32_t i = 0; i < samples; i++){
        float n1 = rng.next_float_range(0.0f, 20.0f);
        float n2 = rng.next_float_range(0.0f, 20.0f);
        float n3 = rng.next_float_range(0.0f, 20.0f);

        float result = n1 + n2 * n3;

        float predicted_result = genome.nn.evaluate({n1, n2, n3}, N_INPUTS, N_OUTPUTS)[0];

        if (samples == 3){
            cout << predicted_result << " " << result << endl;
        }


        genome.fitness += 1.0f / (1.05 + abs(predicted_result - result));
    }

    genome.fitness /= samples;
}

void evaluate_fitnesses(Population& pop){
    for (Genome& genome : pop.genomes){
        //cout << genome.fitness << endl;
        fitness_func(genome, 400);
        //cout << genome.fitness << endl;
    }
}

int main(){
    uint64_t seed;
    _rdrand64_step(&seed);
    rng = FastRNG(seed);


    PopulationConfig config;
    config.elite_factor = 0.1f;
    config.crossover_rate = 0.8f;
    config.mutation_rate = 0.3f;

    config.add_neuron_rate = 0.1f;
    config.remove_neuron_rate = 0.099f;
    config.add_connection_rate = 0.1f;
    config.toggle_connection_rate = 0.101f;

    config.perturb_biases_rate = 0.4f;
    config.perturb_weights_rate = 0.4f;

    config.perturbation_rate = 0.7f;
    config.perturbation_magnitude = 0.1f;
    config.replacement_rate = 0.01f;
    config.min_value = -20.0f;
    config.max_value = 20.0f;
    config.stddev = 5.0f;

    config.crossover_connection_toggling_rate = 0.0f;

    Population pop;



    pop.initialize(POPULATION_SIZE, N_INPUTS, N_OUTPUTS);
    pop.config = config;

    for (size_t i = 0;; i++){
        cout << endl << "GENERATION " << i << endl;

        evaluate_fitnesses(pop);

        pop.generate_offsprings(config);

        // print the average fitness values
        float avg_fitness = 0.0f;
        float best_fitness = 0.0f;
        Genome* best_fitness_genome;
        for (Genome& genome : pop.genomes){
            avg_fitness += genome.fitness;
            if (genome.fitness > best_fitness){
                best_fitness_genome = &genome;
                best_fitness = genome.fitness;
            }
        }

        //print_neurons(pop.genomes[0].nn, N_INPUTS, N_OUTPUTS);
        //print_connections(pop.genomes[0].nn);
        avg_fitness /= POPULATION_SIZE;
        cout << "average fitness: " << avg_fitness << ", best fitness: " << best_fitness << endl;

        cout << best_fitness_genome->fitness << endl;
        fitness_func(*best_fitness_genome, FITNESS_SAMPLES);
        cout << best_fitness_genome->fitness << endl;
        float avg_n_neurons = 0.0f;
        float avg_n_connections = 0.0f;

        for (Genome& genome : pop.genomes){
            avg_n_neurons += genome.nn.neurons.size();
            avg_n_connections += genome.nn.connections.size();
        }

        avg_n_neurons /= POPULATION_SIZE;
        avg_n_connections /= POPULATION_SIZE;

        cout << "average n_neurons " <<  avg_n_neurons << ", average n_connections " << avg_n_connections << endl;

        //Sleep(30);
    }
}

int main2(){
    uint64_t seed;
    _rdrand64_step(&seed);
    rng = FastRNG(seed);

    //NeuralNet nn;
    //nn.initialize(N_INPUTS, N_OUTPUTS);

    //nn.prepare_evaluation(N_INPUTS);

    //print_vector(nn.order);

    //vector<float> out = nn.evaluate({2, 2},N_INPUTS, N_OUTPUTS);

    //print_neurons(nn, N_INPUTS, N_OUTPUTS);
    //print_vector(out);
    //print_connections(nn);

    cout << "finished" << endl;
    return 0;
}