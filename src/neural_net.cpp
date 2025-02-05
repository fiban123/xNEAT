
// initialites the network with input and output neurons. Also adds connectivity
void NeuralNet::initialize(uint32_t n_inputs, uint32_t n_outputs, PopulationConfig& config){

    neurons.clear();
    connections.clear();

    neurons.reserve(n_inputs + n_outputs);

    // the input neurons ALWAYS have id's 0 to n_inputs -1
    // the output neurons ALWAYS have id's n_inputs to n_inputs + n_outputs -1
    for (uint32_t i = 0; i < n_inputs; i++){
        float rand_bias = clamp(rng.next_gaussian_range(0.0f, config.stddev), config.min_value, config.max_value);
        neurons.insert({*iid_counter, Neuron(*iid_counter, rand_bias)});

        input_neuron_iids.push_back(*iid_counter);

        (*iid_counter)++;
    }
    for (uint32_t i = 0; i < n_outputs; i++){
        float rand_bias = clamp(rng.next_gaussian_range(0.0f, config.stddev), config.min_value, config.max_value);
        neurons.insert({*iid_counter, Neuron(*iid_counter, rand_bias)});

        output_neuron_iids.push_back(*iid_counter);

        (*iid_counter)++;
    }


    // connect every input neuron to every output neuron

    connections.reserve(n_inputs * n_outputs);
    for (uint32_t& input_iid : input_neuron_iids){
        for (uint32_t& output_iid : output_neuron_iids){
            ConnectionId id = ConnectionId{input_iid, output_iid};

            float rand_weight = clamp(rng.next_gaussian_range(1.0f, config.stddev), config.min_value, config.max_value);

            Connection connection = Connection(id, rand_weight);
            
            connections.insert({id, connection});
        }
    }
}

/* 
after strucutre updates have been made, prepare_evaluation() must be called before evaluation() !!
returns the outputs of the network given some inputs.
*/
vector<float> NeuralNet::evaluate(vector<float> inputs, uint32_t n_inputs, uint32_t n_outputs){
    assert(inputs.size() == n_inputs);

    for (uint32_t i = 0; i < inputs.size(); i++){
        neurons.at(input_neuron_iids[i]).output = inputs[i];
    }

    // Process neurons in topological order
    for (uint32_t id : order) {
        // make suere id isn't an input neuron
        if (find(input_neuron_iids.begin(), input_neuron_iids.end(), id) != input_neuron_iids.end()){
            continue;
        }
        
        Neuron& neuron = neurons.at(id);
        float sum = neuron.bias;

        for (const Connection connection : neuron.incoming_connections) {
            if (connection.enabled) {
                sum += connection.weight * neurons.at(connection.id.from).output;
            }
        }

        neuron.output = sum;

        // relu

        if (neuron.output < 0){
            neuron.output = 0;
        }

        //neuron.output *= neuron.output;
    }

    vector<float> outputs(n_outputs);

    for (uint32_t i = 0; i < n_outputs; i++){
        outputs[i] = neurons.at(output_neuron_iids[i]).output;
    }

    return outputs;
}

// reevaluates the neuron connections.
void NeuralNet::evaluate_connections(){
    // clear neuron connections
    for (auto& [id, neuron] : neurons){
        neuron.incoming_connections.clear();
    }

    // evaluate connections
    for (const auto& [id, connection] : connections){ // go over every connection

        try{
            neurons.at(connection.id.to).incoming_connections.push_back(connection); // add the input of the connection to the output of the connection

        }
        catch(...){
            cerr << "Error: Neuron " << connection.id.to << " not found." << endl;
            print_neurons(*this);
            print_connections(*this);
            exit(1);
        }
    }
}

// updates the topological order of the network using kahn's algorithm.
void NeuralNet::update_order(uint32_t n_inputs){
    // topological sort of neuron connections using kahns algorithm
    order.clear();
    order.reserve(neurons.size());

    unordered_map<uint32_t, int> in_degree;
    unordered_map<uint32_t, vector<uint32_t> > adj_list;

    // initialite data structures
    for (const auto& [id, neuron] : neurons){
        in_degree[id] = 0;
    }

    // Build adjacency list and calculate in-degrees
    for (const auto& [id, connection] : connections) {
        if (connection.enabled) {
            adj_list[connection.id.from].push_back(connection.id.to);
            in_degree[connection.id.to]++;
        }
    }

    queue<uint32_t> q;

    for (const auto& [id, neuron] : neurons) {
        if (in_degree[id] == 0) {
            q.push(id);
        }
    }
    
    // Kahn's algorithm
    while (!q.empty()) {
        uint32_t current = q.front();
        q.pop();
        order.push_back(current);

        for (uint32_t neighbor : adj_list[current]) {
            if (--in_degree[neighbor] == 0) {
                q.push(neighbor);
            }
        }
    }

    size_t non_stagnant_count = 0; // number of non-stagnant neurons
    for (const auto& [id, neuron] : neurons) {
        if (in_degree[id] > 0 || !adj_list[id].empty()) {
            non_stagnant_count++;
        }
    }


    // Check for cycles
    if (order.size() != neurons.size()) {
        print_neurons(*this);
        print_connections(*this);
        throw runtime_error("Network contains cycles!");
    }
}

// prepares the network for evaluation. This includes reevaluating connections and updating the topological order.
void NeuralNet::prepare_evaluation(uint32_t n_inputs){
    evaluate_connections();
    update_order(n_inputs);
}

bool ConnectionId::operator==(const ConnectionId &other) const {
    return from == other.from && to == other.to;
}

size_t ConnectionIdHash::operator()(const ConnectionId &id) const {
    return (static_cast<uint64_t>(id.from) << 32) | id.to;
}
