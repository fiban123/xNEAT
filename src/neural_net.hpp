struct ConnectionId{
    uint32_t from;
    uint32_t to;

    ConnectionId() = delete;

    ConnectionId(uint32_t from, uint32_t to) : from(from), to(to) {};
    
    bool operator==(const ConnectionId& other) const;
};

struct ConnectionIdHash{
    size_t operator()(const ConnectionId& id) const;
};


struct Connection{
    ConnectionId id;

    float weight = 1.0f;
    bool enabled = true;

    Connection(ConnectionId id) : id(id){};
    Connection(ConnectionId id, float _weight) : id(id), weight(_weight){};

    Connection() = delete;
    Connection(const Connection& other) noexcept = default;
    Connection(Connection&& other) = default;
    Connection& operator=(const Connection& other) = default;
    Connection& operator=(Connection&& other) = default;

};


struct Neuron{
    float bias;
    float output = 0.0f;
    uint32_t iid; // innovation id
    vector<Connection> incoming_connections;

    Neuron(uint32_t iid) : iid(iid){};
    Neuron(uint32_t iid, float bias) : iid(iid), bias(bias){};

    Neuron() = delete;
    Neuron(const Neuron& other) noexcept = default;
    Neuron(Neuron&& other) = default;
    Neuron& operator=(const Neuron& other) = default;
    Neuron& operator=(Neuron&& other) = default;
};

struct NeuralNet{
    unordered_map<ConnectionId, Connection, ConnectionIdHash> connections;
    unordered_map<uint32_t, Neuron> neurons;
    vector<uint32_t> order;

    vector<uint32_t> input_neuron_iids;
    vector<uint32_t> output_neuron_iids;
    uint32_t* iid_counter;

    void initialize(uint32_t n_inputs, uint32_t n_outputs, PopulationConfig &config);

    vector<float> evaluate(vector<float> inputs, uint32_t n_inputs, uint32_t n_outputs);

    void evaluate_connections();
    void update_order(uint32_t n_inputs);

    void prepare_evaluation(uint32_t n_inputs);

    NeuralNet(uint32_t* iid_counter) : iid_counter(iid_counter) {};

    NeuralNet() = delete;
    NeuralNet(const NeuralNet& other) noexcept = default;
    NeuralNet(NeuralNet&& other) = default;
    NeuralNet& operator=(const NeuralNet& other) = default;
    NeuralNet& operator=(NeuralNet&& other) noexcept = default;
};