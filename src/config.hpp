struct PopulationConfig{
    // offspring generation settingd
    float elite_factor; // amount of best genomes that will be kept without crossover / mutation through generations.
    float crossover_rate; // chance of crossover occurring
    float mutation_rate; // chance of mutation occurring

    // mutation settings
    float add_neuron_rate; // chance of neuron being added at mutation
    float remove_neuron_rate; // chance of neuron being removed at mutation
    float add_connection_rate; // chance of connection being added at mutation

    float toggle_connection_rate; // chance of a random connection being toggled at mutation
    float perturb_weights_rate; // chance of connection weights being perturbed at mutation
    float perturb_biases_rate; // chance of neuron bias biases perturbed at mutation

    // weight / bias settings
    float perturbation_rate; // chance of perturbation occurring
    float perturbation_magnitude; // maximum deviation of perturbation
    float replacement_rate; // chance of replacement occurring
    float min_value; // minimum value
    float max_value; // maximum value
    float stddev; // standard deviation for weights / biases

    // crossover settings
    float crossover_connection_toggling_rate; // chance of a connection being toggled at crossover
};