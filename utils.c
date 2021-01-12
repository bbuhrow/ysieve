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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <malloc.h>
#include "util.h"

#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
#define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif

const char* szFeatures[] =
{
    "x87 FPU On Chip",
    "Virtual-8086 Mode Enhancement",
    "Debugging Extensions",
    "Page Size Extensions",
    "Time Stamp Counter",
    "RDMSR and WRMSR Support",
    "Physical Address Extensions",
    "Machine Check Exception",
    "CMPXCHG8B Instruction",
    "APIC On Chip",
    "Unknown1",
    "SYSENTER and SYSEXIT",
    "Memory Type Range Registers",
    "PTE Global Bit",
    "Machine Check Architecture",
    "Conditional Move/Compare Instruction",
    "Page Attribute Table",
    "36-bit Page Size Extension",
    "Processor Serial Number",
    "CFLUSH Extension",
    "Unknown2",
    "Debug Store",
    "Thermal Monitor and Clock Ctrl",
    "MMX Technology",
    "FXSAVE/FXRSTOR",
    "SSE Extensions",
    "SSE2 Extensions",
    "Self Snoop",
    "Multithreading Technology",
    "Thermal Monitor",
    "Unknown4",
    "Pending Break Enable"
};

#if defined (_MSC_VER)
struct timezone
{
    int  tz_minuteswest; /* minutes W of Greenwich */
    int  tz_dsttime;     /* type of dst correction */
};
#endif

void* xmalloc_align(size_t len)
{
#if defined (_MSC_VER) || defined(__MINGW32__)
    void* ptr = _aligned_malloc(len, 64);

#elif defined (__APPLE__)
    void* ptr = malloc(len);

#elif defined (__GNUC__)
    void* ptr = memalign(64, len);

#else
    void* ptr = malloc(len);

#endif

    return ptr;
}

void* xmalloc(size_t len) {
    void* ptr = malloc(len);
    if (ptr == NULL) {
        printf("failed to allocate %u bytes\n", (uint32_t)len);
        exit(-1);
    }
    return ptr;
}

void* xcalloc(size_t num, size_t len) {
    void* ptr = calloc(num, len);
    if (ptr == NULL) {
        printf("failed to calloc %u bytes\n", (uint32_t)(num * len));
        exit(-1);
    }
    return ptr;
}

void* xrealloc(void* iptr, size_t len) {
    void* ptr = realloc(iptr, len);
    if (ptr == NULL) {
        printf("failed to reallocate %u bytes\n", (uint32_t)len);
        exit(-1);
    }
    return ptr;
}

hash_t* initHash(uint32_t elementSizeB, uint32_t pow2numElements)
{
    hash_t *hashTable;
    int i;

    hashTable = (hash_t*)xmalloc(sizeof(hash_t));
    hashTable->hashKey = (uint64_t**)xmalloc((1 << pow2numElements) * sizeof(uint64_t*));
    hashTable->hashBins = (uint8_t **)xmalloc((1 << pow2numElements) * sizeof(uint8_t*));
    hashTable->binSize = (uint32_t*)xcalloc((1 << pow2numElements), sizeof(uint32_t));
    hashTable->elementSizeB = elementSizeB;
    hashTable->numStored = 0;
    hashTable->numBinsPow2 = pow2numElements;
    hashTable->numBins = 1 << pow2numElements;

    printf("initialized hash array of size %u with elements of size %u\n",
        hashTable->numBins, hashTable->elementSizeB);

    for (i = 0; i < hashTable->numBins; i++)
    {
        hashTable->hashBins[i] = (uint8_t*)xmalloc(elementSizeB);
        hashTable->hashKey[i] = (uint64_t*)xmalloc(sizeof(uint64_t));
    }

    return hashTable;
}

void deleteHash(hash_t* hash)
{
    int i;

    for (i = 0; i < hash->numBins; i++)
    {
        free(hash->hashBins[i]);
        free(hash->hashKey[i]);
    }

    free(hash->hashBins);
    free(hash->hashKey);
    free(hash->binSize);
    hash->numStored = 0;
    hash->numBins = 0;
    free(hash);
}

void hashPut(hash_t* hash, uint8_t* element, uint64_t key)
{
    uint32_t binNum = (uint32_t)((((key)+18932479UL) * 2654435761UL) >> (64 - hash->numBinsPow2));
    //printf("hashPut into bin %u with key %u\n", binNum, key);

    if (hash->binSize[binNum] > 0)
    {
        //printf("growing bin %u size to %u\n", binNum + 1, hash->binSize[binNum] + 1);
        hash->hashBins[binNum] = (uint8_t*)xrealloc(hash->hashBins[binNum],
            (hash->binSize[binNum] + 1) * hash->elementSizeB);
        hash->hashKey[binNum] = (uint64_t*)xrealloc(hash->hashKey[binNum],
            (hash->binSize[binNum] + 1) * sizeof(uint64_t));
        memcpy(&hash->hashBins[binNum][hash->elementSizeB * hash->binSize[binNum]],
            element, hash->elementSizeB);
        hash->hashKey[binNum][hash->binSize[binNum]] = key;
        hash->binSize[binNum]++;
    }
    else
    {
        memcpy(hash->hashBins[binNum], element, hash->elementSizeB);
        hash->hashKey[binNum][hash->binSize[binNum]] = key;
        hash->binSize[binNum]++;
    }

    hash->numStored++;

    return;
}

void hashGet(hash_t* hash, uint64_t key, uint8_t*element)
{
    uint32_t binNum = (uint32_t)((((key)+18932479UL) * 2654435761UL) >> (64 - hash->numBinsPow2));
    int i;
    
    for (i = 0; i < hash->binSize[binNum]; i++)
    {
        if (hash->hashKey[binNum][i] == key)
        {
            memcpy(element, &hash->hashBins[binNum][hash->elementSizeB * i],
                hash->elementSizeB);
            return;
        }
    }
    return;
}

uint64_t spRand(uint64_t lower, uint64_t upper, uint64_t *lcg_state)
{
	// advance the state of the LCG and return the appropriate result

#if BITS_PER_DIGIT == 64
	//we need to do some gymnastics to prevent the potentially 64 bit value
	//of (upper - lower) from being truncated to a 53 bit double

	uint32_t n = spBits(upper-lower);

	if (n > 32)
	{
		uint64_t boundary = 4294967296ULL;
		uint64_t l,u;
		l = spRand(lower,boundary-1,lcg_state);
		u = spRand(0,upper>>32,lcg_state);
		return  l + (u << 32);
	}
	else
	{
        *lcg_state = 6364136223846793005ULL * (*lcg_state) + 1442695040888963407ULL;
		return lower + (uint64_t)(
			(double)(upper - lower) * (double)(LCGSTATE >> 32) / 4294967296.0);
	}

	
#else

    *lcg_state = 6364136223846793005ULL * (*lcg_state) + 1442695040888963407ULL;
	return lower + (uint64_t)(
			(double)(upper - lower) * (double)((*lcg_state) >> 32) / 4294967296.0);
#endif

}

//user dimis:
//http://cboard.cprogramming.com/cplusplus-programming/
//101085-how-measure-time-multi-core-machines-pthreads.html
//
double yafu_difftime (struct timeval * start, struct timeval * end)
{
    double secs;
    double usecs;

	if (start->tv_sec == end->tv_sec) {
		secs = 0;
		usecs = end->tv_usec - start->tv_usec;
	}
	else {
		usecs = 1000000 - start->tv_usec;
		secs = end->tv_sec - (start->tv_sec + 1);
		usecs += end->tv_usec;
		if (usecs >= 1000000) {
			usecs -= 1000000;
			secs += 1;
		}
	}
	
	return secs + usecs / 1000000.;
}

//http://www.openasthra.com/c-tidbits/gettimeofday-function-for-windows/
#if defined (_MSC_VER)
	int gettimeofday(struct timeval *tv, struct timezone *tz)
	{
	  FILETIME ft;
	  unsigned __int64 tmpres = 0;
	  static int tzflag;
	 
	  if (NULL != tv)
	  {
		GetSystemTimeAsFileTime(&ft);
	 
		tmpres |= ft.dwHighDateTime;
		tmpres <<= 32;
		tmpres |= ft.dwLowDateTime;
	 
		/*converting file time to unix epoch*/
		tmpres /= 10;  /*convert into microseconds*/
		tmpres -= DELTA_EPOCH_IN_MICROSECS; 
		tv->tv_sec = (long)(tmpres / 1000000UL);
		tv->tv_usec = (long)(tmpres % 1000000UL);
	  }
	 
	  if (NULL != tz)
	  {
		if (!tzflag)
		{
		  _tzset();
		  tzflag++;
		}
		tz->tz_minuteswest = _timezone / 60;
		tz->tz_dsttime = _daylight;
	  }
	 
	  return 0;
	}
#endif

int qcomp_uint16_t(const void *x, const void *y)
{
	uint16_t *xx = (uint16_t *)x;
	uint16_t *yy = (uint16_t *)y;
	
	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

int qcomp_uint32_t(const void *x, const void *y)
{
	uint32_t *xx = (uint32_t *)x;
	uint32_t *yy = (uint32_t *)y;
	
	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

int qcomp_uint64_t(const void *x, const void *y)
{
	uint64_t *xx = (uint64_t *)x;
	uint64_t *yy = (uint64_t *)y;
	
	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

int qcomp_int(const void *x, const void *y)
{
	int *xx = (int *)x;
	int *yy = (int *)y;
	
	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

int qcomp_double(const void *x, const void *y)
{
	double *xx = (double *)x;
	double *yy = (double *)y;

	if (*xx > *yy)
		return 1;
	else if (*xx == *yy)
		return 0;
	else
		return -1;
}

uint32_t * mergesort(uint32_t *a, uint32_t *b, int sz_a, int sz_b)
{
    uint32_t *c = (uint32_t *)malloc((sz_a + sz_b) * sizeof(uint32_t));
    int i = 0, j = 0, k = 0;

    while ((i < sz_a) && (j < sz_b)) {
        if (a[i] < b[j]) {
            c[k++] = a[i++];
        }
        else if (a[i] > b[j]) {
            c[k++] = b[j++];
        }
        else {
            c[k++] = a[i++];
            c[k++] = b[j++];
        }
    }

    while (i < sz_a)
        c[k++] = a[i++];

    while (j < sz_b)
        c[k++] = b[j++];

    return c;
}

int bin_search_uint32_t(int idp, int idm, uint32_t q, uint32_t *input)
{
	int next = (idp + idm) / 2;
	
	while ((idp - idm) > 10)
	{
		if (input[next] > q)
		{
			idp = next;
			next = (next + idm) / 2;							
		}
		else					
		{
			idm = next;
			next = (idp + next) / 2;							
		}
	}

	for (next = idm; next < idm + 10; next++)
		if (input[next] == q)
			return next;

	if (input[next] != q)
		next = -1;

	return next;
}


