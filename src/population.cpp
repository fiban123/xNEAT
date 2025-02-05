void Population::initialize(uint32_t _population_size, uint32_t _n_inputs, uint32_t _n_outputs){
    n_inputs = _n_inputs;
    n_outputs = _n_outputs;
    population_size = _population_size;

    for (genome_counter = 0; genome_counter < population_size; genome_counter++){
        Genome genome(&iid_counter);
        genome.nn.initialize(n_inputs, n_outputs, config);

        genome.nn.prepare_evaluation(n_inputs);
        genome.fitness = 0.0f;
        genome.id = genome_counter;

        genomes.push_back(genome);
    }
}

Genome& select_genome(vector<Genome>& sorted_genomes, vector<uint32_t>& genome_ranks, size_t rank_sum, uint32_t population_size){
    uint32_t random_rank = rng.next_in_range(0, rank_sum - 1);

    size_t cumulative_sum = 0;
    for (uint32_t i = 0; i < population_size; i++){
        cumulative_sum += genome_ranks[i];
        if (random_rank < cumulative_sum) {

            return sorted_genomes[i];
        }
    }

    return sorted_genomes.back(); // return last genome
}

bool genome_fitness_cmp(const Genome& a, const Genome& b){
    return a.fitness > b.fitness;
}

void Population::generate_offsprings(PopulationConfig& config){
    vector<Genome> new_genomes;
    new_genomes.reserve(population_size);

    // sort genomes by fitness
    sort(genomes.begin(), genomes.end(), genome_fitness_cmp);

    uint32_t n_elite = ceilf( (float)population_size * config.elite_factor);

    // keep n_elite genomes without crossover or mutation
    new_genomes.insert(new_genomes.begin(), genomes.begin(), genomes.begin() + n_elite);


    vector<uint32_t> genome_ranks(population_size);
    // rank genomes based on their fitness
    for (uint32_t i = 0; i < population_size; i++){
        genome_ranks[i] = population_size - i;
    }

    size_t rank_sum = 0;
    // calculate total rank number of genomes
    for (uint32_t i = 0; i < population_size; i++){
        rank_sum += genome_ranks[i];
    }

    // crossover and mutation
    while (new_genomes.size() < population_size){
        // select 2 random parents for crossover
        Genome& parent1 = select_genome(genomes, genome_ranks, rank_sum, population_size);
        Genome& parent2 = select_genome(genomes, genome_ranks, rank_sum, population_size);
        // make sure they arent the same
        if (&parent1 == &parent2){
            continue;
        }

        // find dominant genome & recessive genome
        Genome& dominant_genome = parent1.fitness > parent2.fitness ? parent1 : parent2;
        Genome& recessive_genome = &dominant_genome == &parent1 ? parent2 : parent1;

        bool rng2 = rng.next_with_probability(config.crossover_rate);

        Genome child = rng2 ? 
            crossover(dominant_genome, recessive_genome, config) : // crossover
            dominant_genome; // no crossover, clone dominant parent

        // prepare child for network evaluation
        child.id = genome_counter;
        genome_counter++;

        // mutate
        if (rng.next_with_probability(config.mutation_rate)){
            child.mutate(config, n_inputs, n_outputs);
        }

        child.nn.prepare_evaluation(n_inputs);
        new_genomes.push_back(child);
    }

    genomes = new_genomes;
}
