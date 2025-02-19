#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint32_t state128[4];

static inline uint32_t rotl32(const uint32_t x, int k)
{
    return (x << k) | (x >> (32 - k));
}

static inline uint32_t xoshiro128pp(void)
{
    uint32_t *s = state128;
    uint32_t result = rotl32(s[0] + s[3], 7) + s[0];
    uint32_t t = s[1] << 9;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl32(s[3], 11);

    return result;
}

static inline float random_float(void)
{
    return (xoshiro128pp() >> 8) * (1.0f / 16777216.0f);
}

void seed_xoshiro128(uint32_t seed)
{
    uint32_t z = seed;

    for (int i = 0; i < 4; ++i)
    {
        z += 0x9e3779b9;
        z = (z ^ (z >> 15)) * 0x85ebca6b;
        z = (z ^ (z >> 13)) * 0xc2b2ae35;
        state128[i] = z ^ (z >> 16);
    }
}
