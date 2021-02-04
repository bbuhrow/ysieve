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

#ifndef _YSIEVE_UTIL_H_
#define _YSIEVE_UTIL_H_



/* system-specific stuff ---------------------------------------*/

#ifdef __cplusplus
extern "C" {
#endif


#ifdef _MSC_VER
#include <winsock.h>        // timeval
#endif

#include <stdint.h>
#ifndef _MSC_VER
#include <sys/time.h>       // gettimeofday
#endif

/* useful functions ---------------------------------------------------*/

#define MIN(a,b) ((a) < (b)? (a) : (b))
#define MAX(a,b) ((a) > (b)? (a) : (b))

#define INLINE __inline
#if defined(_MSC_VER)
	#define getpid _getpid
#endif

#if defined(__GNUC__) && __GNUC__ >= 3
	#define PREFETCH(addr) __builtin_prefetch(addr) 
#elif defined(_MSC_VER) && _MSC_VER >= 1400
	#define PREFETCH(addr) PreFetchCacheLine(PF_TEMPORAL_LEVEL_1, addr)
#else
	#define PREFETCH(addr) /* nothing */
#endif


#if defined(_MSC_VER)

#define align_free _aligned_free	
#define ALIGNED_MEM __declspec(align(64))     

#elif defined(__GNUC__) || defined(__INTEL_COMPILER)

#if defined(__MINGW64__) || defined(__MINGW32__) || defined(__MSYS__)
#define align_free _aligned_free //_mm_free
#else
#define align_free free
#endif

#if defined (__INTEL_COMPILER)
#define ALIGNED_MEM __declspec(align(64))
#else
#define ALIGNED_MEM __attribute__((aligned(64)))
#endif

#endif



void* xmalloc_align(size_t len);
void* xmalloc(size_t len);
void* xcalloc(size_t num, size_t len);
void* xrealloc(void* iptr, size_t len);

//user dimis:
//http://cboard.cprogramming.com/cplusplus-programming/
//101085-how-measure-time-multi-core-machines-pthreads.html
//
double yafu_difftime (struct timeval *, struct timeval *);

//http://www.openasthra.com/c-tidbits/gettimeofday-function-for-windows/
#if defined (_MSC_VER)
	int gettimeofday(struct timeval *tv, struct timezone *tz);
#endif


uint64_t spRand(uint64_t lower, uint64_t upper, uint64_t* lcg_state);

int qcomp_int(const void *x, const void *y);
int qcomp_int(const void *x, const void *y);
int qcomp_uint16_t(const void *x, const void *y);
int qcomp_uint32_t(const void *x, const void *y);
int qcomp_uint64(const void *x, const void *y);
int qcomp_double(const void *x, const void *y);
int bin_search_uint32_t(int idp, int idm, uint32_t q, uint32_t *input);


typedef struct
{
    uint8_t** hashBins;
    uint64_t** hashKey;
    uint32_t* binSize;
    uint32_t numBins;
    uint32_t numBinsPow2;
    uint32_t numStored;
    uint32_t elementSizeB;
} hash_t;

hash_t* initHash(uint32_t elementSizeB, uint32_t pow2numElements);
void deleteHash(hash_t* hash);
void hashPut(hash_t* hash, uint8_t* element, uint64_t key);
void hashGet(hash_t* hash, uint64_t key, uint8_t* element);



#ifdef __cplusplus
}
#endif


#endif /* _YSIEVE_UTIL_H_ */
