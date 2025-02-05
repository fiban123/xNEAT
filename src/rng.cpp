
FastRNG::FastRNG(uint64_t seed){
    state = seed;
}

inline uint64_t FastRNG::next(){
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return state * 2685821657736338717ULL;
}

inline uint32_t FastRNG::next_in_range(uint32_t min, uint32_t max){
    return min + (next() % (max - min + 1));
}

inline float FastRNG::next_float_range(float min, float max){
    // Generate a random integer
    uint64_t random_value = next();

    // Normalize the random integer to the range [0, 1)
    float normalized = static_cast<float>(random_value) / static_cast<float>(UINT64_MAX);

    // Scale and shift the normalized value to the desired range [min, max]
    return min + normalized * (max - min);
}

float FastRNG::next_gaussian_range(float mean, float stddev){
    // Generate a Gaussian random number using Box-Muller transform

    // We need two random values to generate two Gaussian numbers
    float u1 = static_cast<float>(next()) / UINT64_MAX;
    float u2 = static_cast<float>(next()) / UINT64_MAX;

    // Box-Muller transformation to get a standard normal value
    float z0 = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);

    // Map the Gaussian number to the desired range
    float normalized = (z0 * stddev) + mean;  // Scale and shift

    return normalized;
}

inline bool FastRNG::next_with_probability(double probability){
    // Ensure probability is within valid range [0.0, 1.0]
    probability = clamp(probability, 0.0, 1.0);

    // Generate a random number and compare it to the threshold
    uint64_t rand_value = next();  // Get a random 64-bit number
    return (rand_value / static_cast<double>(UINT64_MAX)) < probability;
}

inline bool FastRNG::next_bool(){
    return next() & 1;
}