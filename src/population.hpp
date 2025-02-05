

struct Population{
    // population config
    uint32_t population_size;
    uint32_t n_inputs;
    uint32_t n_outputs;

    PopulationConfig config;

    vector<Genome> genomes;
    
    uint32_t genome_counter = 0;
    uint32_t iid_counter = 0;
    queue<uint32_t> deleted_neuron_ids; // used for neuron ID recycling

    void initialize(uint32_t population_size, uint32_t n_inputs, uint32_t n_outputs);

    void generate_offsprings(PopulationConfig &config);

};