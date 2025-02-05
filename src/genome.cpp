Neuron crossover_neuron(const Neuron& dominant_neuron, const Neuron& recessive_neuron){
    assert(dominant_neuron.iid == recessive_neuron.iid);

    // return one of the neurons at random

    bool randbool = rng.next_bool();

    if (randbool){
        return dominant_neuron;
    } else {
        return recessive_neuron;
    }

}

Connection crossover_connection(const Connection& dominant_connection, const Connection& recessive_connection, PopulationConfig& config){
    assert(dominant_connection.id == recessive_connection.id);

    Connection new_connection = Connection(dominant_connection.id);

    // choose a weight of one of the connecions at random
    bool randbool = rng.next_bool();
    if (randbool){
        new_connection.weight = dominant_connection.weight;
    } else {
        new_connection.weight = recessive_connection.weight;
    }

    // choose whether it is enabled randomly
    new_connection.enabled = !rng.next_with_probability(config.crossover_connection_toggling_rate);

    return new_connection;
}

Genome crossover(const Genome &dominant_genome, const Genome &recessive_genome, PopulationConfig& config){
    Genome offspring = dominant_genome;
    offspring.nn.neurons.clear();
    offspring.nn.connections.clear();

    // crossover neurons
    for (auto& [id, dominant_neuron] : dominant_genome.nn.neurons){
        // check if this neuron exists in the recessive genome
        auto it = recessive_genome.nn.neurons.find(id);

        if (it != recessive_genome.nn.neurons.end()){ // if the neuron is present
            // create a new neuron using crossover_neuron
            //cout << id << " " << it->second.id << endl;
            if (dominant_neuron.iid != it->second.iid){
                print_neurons(dominant_genome.nn);
                print_connections(dominant_genome.nn);

                print_neurons(recessive_genome.nn);
                print_connections(recessive_genome.nn);

                cout << "sdsdh" << dominant_genome.id << "  " << recessive_genome.id << endl;
            }
            offspring.nn.neurons.insert({id, crossover_neuron(dominant_neuron, it->second)});
        }
        else{
            offspring.nn.neurons.insert({id, dominant_neuron});
        }
    }

    // crossover connections
    for (auto& [id, dominant_connection] : dominant_genome.nn.connections){
        // check if this connection exists in the recessive genome
        auto it = recessive_genome.nn.connections.find(id);

        if (it != recessive_genome.nn.connections.end()){ // if the connection is present
            offspring.nn.connections.insert({id, crossover_connection(dominant_connection, it->second, config)});
        }
        else{
            offspring.nn.connections.insert({id, dominant_connection});
        }
    }

    return offspring;
}

// selects a random connection
Connection& random_connection(NeuralNet& nn){
    uint32_t random_connection_index = rng.next_in_range(0, nn.connections.size() - 1);

    auto it = nn.connections.begin();
    advance(it, random_connection_index);

    return it->second;
}

// selects a random connection
Neuron& random_neuron(NeuralNet& nn){
    uint32_t random_connection_index = rng.next_in_range(0, nn.neurons.size() - 1);

    auto it = nn.neurons.begin();
    advance(it, random_connection_index);

    return it->second;
}

inline void perturb_value(float& value, PopulationConfig& config){
    if (rng.next_with_probability(config.replacement_rate)){
        value = rng.next_gaussian_range(1.0f, config.stddev);
        value = clamp(value, config.min_value, config.max_value);
        return;
    }

    if (rng.next_with_probability(config.perturbation_rate)){
        value += rng.next_float_range(-config.perturbation_magnitude, config.perturbation_magnitude);
        value = clamp(value, config.min_value, config.max_value);
    }
}

void Genome::mutate(PopulationConfig& config, uint32_t n_inputs, uint32_t n_outputs){
    if (rng.next_with_probability(config.add_neuron_rate)){
        add_random_neuron(config);
    }

    if (rng.next_with_probability(config.remove_neuron_rate)){
        remove_random_neuron(n_inputs, n_outputs);
    }

    if (rng.next_with_probability(config.add_connection_rate)){
        add_random_connection();
    }

    
    if (rng.next_with_probability(config.toggle_connection_rate)){
        toggle_random_connection();
    }

    if (rng.next_with_probability(config.perturb_biases_rate)){
        perturb_biases(config);
    }

    if (rng.next_with_probability(config.perturb_weights_rate)){
        perturb_weights(config);
    }
}




void Genome::toggle_random_connection(){
    // select a random connection
    Connection& connection = random_connection(nn);

    // toggle it's enabled flag
    connection.enabled = !connection.enabled;
}

void Genome::perturb_weights(PopulationConfig config){
    for (auto& [id, connection] : nn.connections) {
        // perturb the weight
        perturb_value(connection.weight, config);
    }
}

void Genome::perturb_biases(PopulationConfig config){
    for (auto& [id, neuron] : nn.neurons) {
        // perturb the bias
        perturb_value(neuron.bias, config);
    }

}

void Genome::add_random_neuron(PopulationConfig& config){
    // select a random connection
    Connection& connection = random_connection(nn);

    // copy the connection (remove reference)
    Connection tmp = connection;

    // delete the connection
    nn.connections.erase(tmp.id);

    // add a new neuron
    Neuron new_neuron = Neuron(*nn.iid_counter);


    // insert the neuron
    nn.neurons.insert({new_neuron.iid, new_neuron});

    (*nn.iid_counter)++;

    // add new connections
    // create a new connection from the new neuron to the connection's from neuron
    Connection c1 = Connection({tmp.id.from, new_neuron.iid});
    c1.weight = tmp.weight;

    Connection c2 = Connection({new_neuron.iid, tmp.id.to});
    c2.weight = 1.0f;
    perturb_value(c2.weight, config);

    nn.connections.insert({c1.id, c1});
    nn.connections.insert({c2.id, c2});

    // update the neuron's incoming connections
    nn.neurons.at(new_neuron.iid).incoming_connections.push_back(c1);
    nn.neurons.at(tmp.id.to).incoming_connections.push_back(c2);

}

void Genome::remove_random_neuron(uint32_t n_inputs, uint32_t n_outputs){
    // select a random neuron
    Neuron neuron = random_neuron(nn);

    // make sure neuron is not an input or output neuron
    if (find(nn.input_neuron_iids.begin(), nn.input_neuron_iids.end(), neuron.iid) != nn.input_neuron_iids.end() || 
        find(nn.output_neuron_iids.begin(), nn.output_neuron_iids.end(), neuron.iid) != nn.output_neuron_iids.end()){
        
        return;
    }

    // delete incoming connections from neuron
    vector<Connection> incoming_connections;

    for (const auto& [id, connection] : nn.connections){
        if (connection.id.to == neuron.iid){
            incoming_connections.push_back(connection);
        }
    }

    // delete outgoing connections to neuron
    vector<Connection> outgoing_connections;

    for (const auto& [id, connection] : nn.connections){
        if (connection.id.from == neuron.iid){
            outgoing_connections.push_back(connection);
        }
    }
    
    for (Connection& connection : outgoing_connections){
        nn.connections.erase(connection.id);
    }

    for (Connection& connection : incoming_connections){
        nn.connections.erase(connection.id);
    }

    auto it = nn.neurons.find(neuron.iid);

    if (it == nn.neurons.end()){
        cout << neuron.iid << endl;
        print_neurons(nn);
        print_connections(nn);
        throw runtime_error("Could not find neuron to delete.");
    }

    // delete the neuron
    nn.neurons.erase(it);
}

void Genome::add_random_connection(){
    uint32_t connection_depth = rng.next_in_range(0, 3);

    uint32_t from;
    uint32_t to;


    // select a random connection
    Connection& connection = random_connection(nn);
    from = connection.id.from;

    // follow the connection to the next neuron
    Neuron cneuron = nn.neurons.at(connection.id.to);
    to = cneuron.iid;

    // follow a random connection path for depth
    for (uint32_t i = 0; i < connection_depth; i++){ /// SOMEWHERE IN HERE
        // find any connections that start at this neuron
        vector<Connection> outgoing_connections;
        for (const auto& [id, connection] : nn.connections){
            if (connection.id.from == cneuron.iid){
                outgoing_connections.push_back(connection);
            }
        }

        if (outgoing_connections.empty()){
            break;
        }

        // select a random connection from the outgoing connections
        uint32_t next_connection_index = rng.next_in_range(0, outgoing_connections.size() - 1);

        Connection next_connection = outgoing_connections[next_connection_index];
        to = next_connection.id.to;
        cneuron = nn.neurons.at(to);
    }

    // add the connection
    Connection new_connection = Connection({from, to});
    new_connection.weight = 0.05f;
    nn.connections.insert({new_connection.id, new_connection});

    // update the neuron's incoming connections
    nn.neurons.at(from).incoming_connections.push_back(new_connection);

}