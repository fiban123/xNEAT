template<typename T>
void print_vector(vector<T> v){
    for (size_t i = 0; i < v.size(); i++){
        cout << v[i] << " ";
    }
    cout << endl;
}

// TODO: implement innovation Ids below

void print_neurons(NeuralNet nn){
    cout << "input neurons:" << endl;

    for (uint32_t& iid : nn.input_neuron_iids){
        cout << "    Neuron " << iid << " / " << nn.neurons.at(iid).iid << ": " << nn.neurons.at(iid).output << " (" << nn.neurons.at(iid).bias << ")" << endl;
    }

    cout << "hidden neurons:" << endl;

    for (const auto& [iid, neuron] : nn.neurons){
        if (find(nn.input_neuron_iids.begin(), nn.input_neuron_iids.end(), iid) == nn.input_neuron_iids.end() &&
        find(nn.output_neuron_iids.begin(), nn.output_neuron_iids.end(), iid) == nn.output_neuron_iids.end()){

            cout << "    Neuron " << iid << " / " << nn.neurons.at(iid).iid << ": " << nn.neurons.at(iid).output << " (" << nn.neurons.at(iid).bias << ")" << endl;
        }
    }

    cout << "output neurons:" << endl;

    for (uint32_t& iid : nn.output_neuron_iids){
        cout << "    Neuron " << iid << " / " << nn.neurons.at(iid).iid << ": " << nn.neurons.at(iid).output << " (" << nn.neurons.at(iid).bias << ")" << endl;
    }

}

void print_connections(NeuralNet nn){
    for (auto const& [id, connection] : nn.connections){
        if (connection.enabled){
            cout << "Connection " << ": from=" << connection.id.from << ", to=" << connection.id.to << ", weight=" << connection.weight << endl;
        }
    }

    cout << endl;

}