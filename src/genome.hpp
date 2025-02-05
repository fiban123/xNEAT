

struct Genome{
    NeuralNet nn;
    float fitness = 0.0f;
    uint32_t id;

    void mutate(PopulationConfig &config, uint32_t n_inputs, uint32_t n_outputs);

    // structural mutations
    void add_random_neuron(PopulationConfig &config);
    void remove_random_neuron(uint32_t n_inputs, uint32_t n_outputs);
    void add_random_connection();

    // non-structural mutations
    void toggle_random_connection();
    void perturb_weights(PopulationConfig config);
    void perturb_biases(PopulationConfig config);


    Genome(uint32_t* iid_counter) : nn(iid_counter){};

    Genome() = delete;
    Genome(const Genome& other) noexcept = default;
    Genome(Genome&& other) = default;
    Genome& operator=(const Genome& other) = default;
    Genome& operator=(Genome&& other) = default;
};

Neuron crossover_neuron(const Neuron &dominant_neuron, const Neuron &recessive_neuron);

Connection crossover_connection(const Connection &dominant_connection, const Connection &recessive_connection, PopulationConfig &config);

Genome crossover(const Genome &dominant_genome, const Genome &recessive_genome, PopulationConfig &config);

Connection &random_connection(NeuralNet &nn);

Neuron &random_neuron(NeuralNet &nn);
