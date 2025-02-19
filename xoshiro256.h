#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t state256[4];

static inline uint64_t rotl64(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t xoshiro256pp(void)
{
    uint64_t *s = state256;
    uint64_t result = rotl64(s[0] + s[3], 23) + s[0];
    uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl64(s[3], 45);

    return result;
}

static inline double random_double(void)
{
    return (xoshiro256pp() >> 11) * (1.0 / 9007199254740992.0);
}

void seed_xoshiro256(uint64_t seed)
{
    uint64_t z = seed;

    for (int i = 0; i < 4; ++i)
    {
        z += 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        state256[i] = z ^ (z >> 31);
    }
}
