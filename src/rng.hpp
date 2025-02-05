struct FastRNG{
    uint64_t state;

    FastRNG(uint64_t seed);
    FastRNG() = default;

    // Returns a random 64-bit number and increments the RNG state)
    inline uint64_t next();

    // Returns a random number in the range [min, max)
    inline uint32_t next_in_range(uint32_t min, uint32_t max);

    // Method to return a float in the range [min, max]
    inline float next_float_range(float min, float max);

    float next_gaussian_range(float mean, float stddev);

    // Method that returns true with a certain probability (0.0 to 1.0)
    inline bool next_with_probability(double probability);
    
    // returns true / false with a uniform distribution
    inline bool next_bool();
};