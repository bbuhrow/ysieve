/*
MIT License

Copyright (c) 2021 Ben Buhrow

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef YAFU_SOE_H
#define YAFU_SOE_H

#include <stdint.h>
#include "threadpool.h"
#include "immintrin.h"
#include "gmp.h"
#include "util.h"

#define USE_SOE_THREADPOOL
#define BITSINBYTE 8
#define MAXSIEVEPRIMECOUNT 100000000	//# primes less than ~2e9: limit of 2e9^2 = 4e18

enum soe_command {
	SOE_COMMAND_INIT,
	SOE_COMMAND_WAIT,
	SOE_COMMAND_SIEVE_AND_COUNT,
	SOE_COMMAND_SIEVE_AND_COMPUTE,
	SOE_COMPUTE_ROOTS,
	SOE_COMPUTE_PRIMES,
	SOE_COMPUTE_PRPS,
	SOE_COMMAND_END
};

#ifdef __INTEL_COMPILER
// leading and trailing zero count are ABM instructions
// that require haswell or later on Intel or ABM on AMD.
// The same basic functionality exists with the 
// bsf and bsr instructions that are standard x86, if
// those requirements are not met.
#if defined( USE_BMI2 ) || defined (TARGET_KNL) || defined( USE_AVX512F )
#define _reset_lsb(x) _blsr_u32(x)
#define _reset_lsb64(x) _blsr_u64(x)
#define _lead_zcnt64 __lzcnt64
#define _trail_zcnt _tzcnt_u32
#define _trail_zcnt64 _tzcnt_u64
#else
#define _reset_lsb(x) ((x) &= ((x) - 1))
#define _reset_lsb64(x) ((x) &= ((x) - 1))
__inline uint32_t _trail_zcnt(uint32_t x)
{
    uint32_t pos;
    if (_BitScanForward(&pos, x))
        return pos;
    else
        return 32;
}
__inline uint64_t _trail_zcnt64(uint64_t x)
{
    uint64_t pos;
    if (_BitScanForward64(&pos, x))
        return pos;
    else
        return 64;
}
__inline uint64_t _lead_zcnt64(uint64_t x)
{
    uint64_t pos;
    if (_BitScanReverse64(&pos, x))
        return pos;
    else
        return 64;
}
#endif
#elif defined(__GNUC__)
#if defined( USE_BMI2 ) || defined (TARGET_KNL) || defined( USE_AVX512F )
#define _reset_lsb(x) _blsr_u32(x)
#define _reset_lsb64(x) _blsr_u64(x)
#define _lead_zcnt64 __builtin_clzll
#define _trail_zcnt __builtin_ctzl
#define _trail_zcnt64 __builtin_ctzll
#else
#define _reset_lsb(x) ((x) &= ((x) - 1))
#define _reset_lsb64(x) ((x) &= ((x) - 1))
#define _lead_zcnt64 __builtin_clzll
#define _trail_zcnt __builtin_ctzl
#define _trail_zcnt64 __builtin_ctzll

#endif
#elif defined(_MSC_VER)
#include <intrin.h>
#ifdef USE_BMI2
#define _lead_zcnt64 __lzcnt64
#define _trail_zcnt _tzcnt_u32
#define _trail_zcnt64 _tzcnt_u64
#define _reset_lsb(x) ((x) &= ((x) - 1))
#define _reset_lsb64(x) ((x) &= ((x) - 1))
#else
__inline uint32_t _trail_zcnt(uint32_t x)
{
    uint32_t pos;
    if (_BitScanForward(&pos, x))
        return pos;
    else
        return 32;
}
__inline uint64_t _trail_zcnt64(uint64_t x)
{
    uint64_t pos;
    if (_BitScanForward64(&pos, x))
        return pos;
    else
        return 64;
}
__inline uint64_t _lead_zcnt64(uint64_t x)
{
    uint64_t pos;
    if (_BitScanReverse64(&pos, x))
        return pos;
    else
        return 64;
}
#define _reset_lsb(x) ((x) &= ((x) - 1))
#define _reset_lsb64(x) ((x) &= ((x) - 1))
#endif

#else

__inline uint64_t _lead_zcnt64(uint64_t x)
{
    uint64_t pos;
    if (x)
    {
        pos = 0;
        for (pos = 0; ; pos++)
        {
            if (x & (1ULL << (63 - pos)))
                break;
        }
    }
    else
    {
#ifdef CHAR_BIT
        pos = CHAR_BIT * sizeof(x);
#else
        pos = 8 * sizeof(x);
#endif
    }
    return pos;
}

__inline uint32_t _trail_zcnt(uint32_t x)
{
    uint32_t pos;
    if (x)
    {
        x = (x ^ (x - 1)) >> 1;  // Set x's trailing 0s to 1s and zero rest
        for (pos = 0; x; pos++)
        {
            x >>= 1;
        }
    }
    else
    {
#ifdef CHAR_BIT
        pos = CHAR_BIT * sizeof(x);
#else
        pos = 8 * sizeof(x);
#endif
    }
    return pos;
}

__inline uint64_t _trail_zcnt64(uint64_t x)
{
    uint64_t pos;
    if (x)
    {
        x = (x ^ (x - 1)) >> 1;  // Set x's trailing 0s to 1s and zero rest
        for (pos = 0; x; pos++)
        {
            x >>= 1;
        }
    }
    else
    {
#ifdef CHAR_BIT
        pos = CHAR_BIT * sizeof(x);
#else
        pos = 8 * sizeof(x);
#endif
    }
    return pos;
}
#define _reset_lsb(x) ((x) &= ((x) - 1))
#define _reset_lsb64(x) ((x) &= ((x) - 1))

#endif

#ifdef USE_AVX2

#ifdef USE_AVX512F
extern ALIGNED_MEM uint64_t presieve_largemasks[16][173][8];
extern ALIGNED_MEM uint32_t presieve_steps[32];
extern ALIGNED_MEM uint32_t presieve_primes[32];
extern ALIGNED_MEM uint32_t presieve_p1[32];

#else
// for storage of presieving lists from prime index 24 to 40 (97 to 173 inclusive)
extern ALIGNED_MEM uint64_t presieve_largemasks[16][173][4];
extern ALIGNED_MEM uint32_t presieve_steps[32];
extern ALIGNED_MEM uint32_t presieve_primes[32];
extern ALIGNED_MEM uint32_t presieve_p1[32];
#endif

// macros for Montgomery arithmetic - helpful for computing 
// division-less offsets once we have enough reuse (number of
// classes) to justify the setup costs.
#define _mm256_cmpge_epu32(a, b) \
        _mm256_cmpeq_epi32(_mm256_max_epu32(a, b), a)

#define _mm256_cmple_epu32(a, b) _mm256_cmpge_epu32(b, a)


static __inline __m256i CLEAR_HIGH_VEC(__m256i x)
{
    __m256i chi = _mm256_set1_epi64x(0x00000000ffffffff);
    return _mm256_and_si256(chi, x);
}

static __inline __m256i vec_redc(__m256i x64e, __m256i x64o, __m256i pinv, __m256i p)
{
    // uint32_t m = (uint32_t)x * pinv;
    __m256i t1 = _mm256_shuffle_epi32(pinv, 0xB1);      // odd-index pinv in lo words
    __m256i even = _mm256_mul_epu32(x64e, pinv);
    __m256i odd = _mm256_mul_epu32(x64o, t1);
    __m256i t2;

    // x += (uint64_t)m * (uint64_t)p;
    t1 = _mm256_shuffle_epi32(p, 0xB1);      // odd-index p in lo words
    even = _mm256_add_epi64(x64e, _mm256_mul_epu32(even, p));
    odd = _mm256_add_epi64(x64o, _mm256_mul_epu32(odd, t1));

    // m = x >> 32;
    t1 = _mm256_blend_epi32(odd, _mm256_shuffle_epi32(even, 0xB1), 0x55);

    // if (m >= p) m -= p;
    t2 = _mm256_cmpge_epu32(t1, p); //_mm256_or_si256(_mm256_cmpgt_epi32(t1, p), _mm256_cmpeq_epi32(t1, p));
    t2 = _mm256_and_si256(p, t2);

    return _mm256_sub_epi32(t1, t2);
}

static __inline __m256i vec_to_monty(__m256i x, __m256i r2, __m256i pinv, __m256i p)
{
    //uint64_t t = (uint64_t)x * (uint64_t)r2;
    __m256i t1 = _mm256_shuffle_epi32(x, 0xB1);
    __m256i t2 = _mm256_shuffle_epi32(r2, 0xB1);
    __m256i even = _mm256_mul_epu32(x, r2);
    __m256i odd = _mm256_mul_epu32(t1, t2);

    //return redc_loc(t, pinv, p);
    return vec_redc(even, odd, pinv, p);
}

#endif

typedef struct
{
	//uint32_t prime;		// the prime, so that we don't have to also look in the
						// main prime array
	uint32_t bitloc;		// bit location of the current hit
	uint32_t next_pid;	// index of the next prime that hits in the current sieve
	uint32_t p_div;		// prime / prodN
	uint8_t p_mod;		// prime % prodN
	uint8_t eacc;			// accumulated error
} soe_bitmap_p;

typedef struct
{
    int VFLAG;
    int THREADS;
    int sync_count;
	uint32_t *sieve_p;
    uint32_t num_sp;
	int *root;
	uint32_t *lower_mod_prime;

    uint32_t *pinv;       // montgomery inverse
    uint32_t *r2modp;     // to go out of montgomery rep

	uint64_t blk_r;
	uint64_t blocks;
	uint64_t partial_block_b;
	uint64_t prodN;
	uint64_t startprime;
	uint64_t orig_hlimit;
	uint64_t orig_llimit;
	uint64_t pbound;
	uint64_t pboundi;

	uint32_t bucket_start_id;
	uint32_t large_bucket_start_prime;
	uint32_t num_bucket_primes;
    uint64_t bitmap_lower_bound;
	uint32_t bitmap_start_id;
	uint32_t num_bitmap_primes;

	uint64_t lowlimit;
	uint64_t highlimit;
	uint64_t numlinebytes;
	uint32_t numclasses;
	uint32_t *rclass;
	uint32_t *special_count;
	uint32_t num_special_bins;
	uint8_t **lines;
	uint32_t bucket_alloc;
	uint32_t large_bucket_alloc;
	uint64_t num_found;
#if defined(bitmap_BUCKET)
	soe_bitmap_p *bitmap_data;
	int **bitmap_ptrs;
#endif
	int only_count;
	mpz_t *offset;
	int sieve_range;
	uint64_t min_sieved_val;

    // presieving stuff
    int presieve_max_id;

    // for small ranges we have only 2 residue classes which
    // is not enough calls to get_offsets() to justify the
    // Montgomery arithmetic setup costs (reduction for each prime
    // is only performed twice).
    int use_monty;

    // masks for removing or reading single bits in a byte.  nmasks are simply
    // the negation of these masks, and are filled in within the spSOE function.
    uint8_t masks[8];
    uint8_t nmasks[8];
    uint32_t masks32[32];
    uint32_t nmasks32[32];
    uint32_t max_bucket_usage;
    uint64_t GLOBAL_OFFSET;
    int NO_STORE;
    uint32_t SOEBLOCKSIZE;
    uint32_t FLAGSIZE;
    uint32_t FLAGSIZEm1;
    uint32_t FLAGBITS;
    uint32_t BUCKETSTARTI;

} soe_staticdata_t;

typedef struct
{
	uint64_t *pbounds;
	uint32_t *offsets;
	uint64_t lblk_b;
	uint64_t ublk_b;
	uint64_t blk_b_sqrt;
	uint32_t bucket_depth;
    uint32_t blockstart;
    uint32_t blockstop;

	uint32_t bucket_alloc;
	uint32_t *bucket_hits;
    uint64_t **sieve_buckets;
	
	uint32_t *special_count;
	uint32_t num_special_bins;

	uint32_t **large_sieve_buckets;
	uint32_t *large_bucket_hits;
	uint32_t bucket_alloc_large;
	
	uint64_t *primes;
	uint32_t largep_offset;
	uint64_t min_sieved_val;

    // presieving stuff
    uint32_t *presieve_scratch;

} soe_dynamicdata_t;

typedef struct {
	soe_dynamicdata_t ddata;
	soe_staticdata_t sdata;
	uint64_t linecount;
	uint32_t current_line;

    int tindex;
    int tstartup;

	// start and stop for computing roots
	uint32_t startid, stopid;

	// stuff for computing PRPs
	mpz_t offset, lowlimit, highlimit, tmpz;

	/* fields for thread pool synchronization */
	volatile enum soe_command command;

#ifdef USE_SOE_THREADPOOL
    /* fields for thread pool synchronization */
    volatile int *thread_queue, *threads_waiting;

#if defined(WIN32) || defined(_WIN64)
    HANDLE thread_id;
    HANDLE run_event;

    HANDLE finish_event;
    HANDLE *queue_event;
    HANDLE *queue_lock;

#else
    pthread_t thread_id;
    pthread_mutex_t run_lock;
    pthread_cond_t run_cond;

    pthread_mutex_t *queue_lock;
    pthread_cond_t *queue_cond;
#endif

#else

#if defined(WIN32) || defined(_WIN64)
    HANDLE thread_id;
    HANDLE run_event;
    HANDLE finish_event;
#else
    pthread_t thread_id;
    pthread_mutex_t run_lock;
    pthread_cond_t run_cond;
#endif

#endif

} thread_soedata_t;

// for use with threadpool
typedef struct
{
    soe_staticdata_t *sdata;
    thread_soedata_t *ddata;
} soe_userdata_t;

static __inline uint32_t redc_loc(uint64_t x, uint32_t pinv, uint32_t p)
{
    uint32_t m = (uint32_t)x * pinv;
    x += (uint64_t)m * (uint64_t)p;
    m = x >> 32;
    if (m >= p) m -= p;
    return m;
}

static __inline uint32_t to_monty_loc(uint32_t x, uint32_t r2, uint32_t pinv, uint32_t p)
{
    uint64_t t = (uint64_t)x * (uint64_t)r2;
    return redc_loc(t, pinv, p);
}


// interface functions
extern soe_staticdata_t* soe_init(int vflag, int threads, int blocksize);
extern void soe_finalize(soe_staticdata_t* sdata);
extern uint64_t* soe_wrapper(soe_staticdata_t* sdata, uint64_t lowlimit, uint64_t highlimit,
    int count, uint64_t* num_p, int PRIMES_TO_FILE, int PRIMES_TO_SCREEN);
extern uint64_t* sieve_to_depth(soe_staticdata_t* sdata,
    mpz_t lowlimit, mpz_t highlimit, int count, int num_witnesses, uint64_t* num_p,
    int PRIMES_TO_FILE, int PRIMES_TO_SCREEN);

// thread ready sieving functions
void sieve_line(thread_soedata_t *thread_data);
void sieve_line_avx2_32k(thread_soedata_t *thread_data);
void sieve_line_avx2_64k(thread_soedata_t *thread_data);
void sieve_line_avx2_128k(thread_soedata_t *thread_data);
void sieve_line_avx2_256k(thread_soedata_t *thread_data);
void sieve_line_avx2_512k(thread_soedata_t* thread_data);
void sieve_line_avx512_32k(thread_soedata_t *thread_data);
void sieve_line_avx512_64k(thread_soedata_t *thread_data);
void sieve_line_avx512_128k(thread_soedata_t *thread_data);
void sieve_line_avx512_256k(thread_soedata_t *thread_data);
void sieve_line_avx512_512k(thread_soedata_t *thread_data);
void(*sieve_line_ptr)(thread_soedata_t *);


uint64_t count_line(soe_staticdata_t *sdata, uint32_t current_line);
void count_line_special(thread_soedata_t *thread_data);
uint32_t compute_32_bytes(soe_staticdata_t *sdata,
    uint32_t pcount, uint64_t *primes, uint64_t byte_offset);
uint64_t primes_from_lineflags(soe_staticdata_t *sdata, thread_soedata_t *thread_data,
	uint32_t start_count, uint64_t *primes);
void get_offsets(thread_soedata_t *thread_data);
void getRoots(soe_staticdata_t *sdata, thread_soedata_t *thread_data);
void stop_soe_worker_thread(thread_soedata_t *t);
void start_soe_worker_thread(thread_soedata_t *t);
#if defined(WIN32) || defined(_WIN64)
DWORD WINAPI soe_worker_thread_main(LPVOID thread_data);
#else
void *soe_worker_thread_main(void *thread_data);
#endif

// routines for finding small numbers of primes; seed primes for main SOE
uint32_t tiny_soe(uint32_t limit, uint32_t *primes);

// top level sieving routines
uint64_t* GetPRIMESRange(soe_staticdata_t* sdata, 
    mpz_t* offset, uint64_t lowlimit, uint64_t highlimit, uint64_t* num_p);
uint64_t spSOE(soe_staticdata_t* sdata, mpz_t* offset,
    uint64_t lowlimit, uint64_t* highlimit, int count, uint64_t* primes);

// misc and helper functions
uint64_t estimate_primes_in_range(uint64_t lowlimit, uint64_t highlimit);
void get_numclasses(uint64_t highlimit, uint64_t lowlimit, soe_staticdata_t *sdata);
int check_input(uint64_t highlimit, uint64_t lowlimit, uint32_t num_sp, uint32_t *sieve_p,
	soe_staticdata_t *sdata, mpz_t offset);
uint64_t init_sieve(soe_staticdata_t *sdata);
void set_bucket_depth(soe_staticdata_t *sdata);
uint64_t alloc_threaddata(soe_staticdata_t *sdata, thread_soedata_t *thread_data);
void do_soe_sieving(soe_staticdata_t *sdata, thread_soedata_t *thread_data, int count);
void finalize_sieve(soe_staticdata_t *sdata,
	thread_soedata_t *thread_data, int count, uint64_t *primes);

uint32_t modinv_1(uint32_t a, uint32_t p);
uint32_t modinv_1b(uint32_t a, uint32_t p);
uint32_t modinv_1c(uint32_t a, uint32_t p);
uint64_t spGCD(uint64_t x, uint64_t y);

void pre_sieve(soe_dynamicdata_t *ddata, soe_staticdata_t *sdata, uint8_t *flagblock);
void pre_sieve_avx2(soe_dynamicdata_t *ddata, soe_staticdata_t *sdata, uint8_t *flagblock);
void pre_sieve_avx512(soe_dynamicdata_t *ddata, soe_staticdata_t *sdata, uint8_t *flagblock);
void (*pre_sieve_ptr)(soe_dynamicdata_t *, soe_staticdata_t *, uint8_t *);

uint32_t compute_8_bytes(soe_staticdata_t *sdata,
    uint32_t pcount, uint64_t *primes, uint64_t byte_offset);
uint32_t compute_8_bytes_bmi2(soe_staticdata_t *sdata,
    uint32_t pcount, uint64_t *primes, uint64_t byte_offset);
uint32_t (*compute_8_bytes_ptr)(soe_staticdata_t *, uint32_t, uint64_t *, uint64_t);




#endif // YAFU_SOE_H
