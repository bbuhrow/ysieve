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

#include "soe.h"
#include "soe_impl.h"
#include "ytools.h"
#include <stdint.h>
#include <immintrin.h>

uint64_t count_line(soe_staticdata_t *sdata, uint32_t current_line)
{
	//extract stuff from the thread data structure
	uint8_t *line = sdata->lines[current_line];
	uint64_t numlinebytes = sdata->numlinebytes;
	uint64_t lowlimit = sdata->lowlimit;
	uint64_t prodN = sdata->prodN;
	uint8_t *flagblock = line;
	uint64_t i, it = 0;
    uint64_t stopcount;
	int ix;
	int done, kx;
	uint64_t prime;
    uint8_t* masks = sdata->masks;
    uint8_t* nmasks = sdata->nmasks;

#ifdef USE_AVX2

    __m256i v5, v3, v0f, v3f;
    uint32_t *tmp;

    v5 = _mm256_set1_epi32(0x55555555);
    v3 = _mm256_set1_epi32(0x33333333);
    v0f = _mm256_set1_epi32(0x0F0F0F0F);
    v3f = _mm256_set1_epi32(0x0000003F);
    tmp = (uint32_t *)xmalloc_align(8 * sizeof(uint32_t));

	uint64_t numchunks = (sdata->orig_hlimit - lowlimit) / (512 * prodN) + 1;

	stopcount = numchunks * 2; // i / 32;
    for (i = 0; i < stopcount; i += 2)
    {
        __m256i t1, t2, t3, t4;
        __m256i x = _mm256_load_si256((__m256i *)(&flagblock[32 * i]));
        __m256i y = _mm256_load_si256((__m256i *)(&flagblock[32 * i + 32]));
        t1 = _mm256_srli_epi64(x, 1);
        t3 = _mm256_srli_epi64(y, 1);
        t1 = _mm256_and_si256(t1, v5);
        t3 = _mm256_and_si256(t3, v5);
        x = _mm256_sub_epi64(x, t1);
        y = _mm256_sub_epi64(y, t3);
        t1 = _mm256_and_si256(x, v3);
        t3 = _mm256_and_si256(y, v3);
        t2 = _mm256_srli_epi64(x, 2);
        t4 = _mm256_srli_epi64(y, 2);
        t2 = _mm256_and_si256(t2, v3);
        t4 = _mm256_and_si256(t4, v3);
        x = _mm256_add_epi64(t2, t1);
        y = _mm256_add_epi64(t4, t3);
        t1 = _mm256_srli_epi64(x, 4);
        t3 = _mm256_srli_epi64(y, 4);
        x = _mm256_add_epi64(x, t1);
        y = _mm256_add_epi64(y, t3);
        x = _mm256_and_si256(x, v0f);
        y = _mm256_and_si256(y, v0f);
        t1 = _mm256_srli_epi64(x, 8);
        t3 = _mm256_srli_epi64(y, 8);
        x = _mm256_add_epi64(x, t1);
        y = _mm256_add_epi64(y, t3);
        t1 = _mm256_srli_epi64(x, 16);
        t3 = _mm256_srli_epi64(y, 16);
        x = _mm256_add_epi64(x, t1);
        y = _mm256_add_epi64(y, t3);
        t1 = _mm256_srli_epi64(x, 32);
        t3 = _mm256_srli_epi64(y, 32);
        x = _mm256_add_epi64(x, t1);
        y = _mm256_add_epi64(y, t3);
        x = _mm256_and_si256(x, v3f);
        y = _mm256_and_si256(y, v3f);
        _mm256_store_si256((__m256i *)tmp, x);
        it += tmp[0] + tmp[2] + tmp[4] + tmp[6];
        _mm256_store_si256((__m256i *)tmp, y);
        it += tmp[0] + tmp[2] + tmp[4] + tmp[6];
       
    }

    align_free(tmp);

#else

	// process 64 bits at a time by using Warren's algorithm
	uint64_t numchunks = (sdata->orig_hlimit - lowlimit) / (64 * prodN) + 1;
    uint64_t* flagblock64 = (uint64_t*)line;

    for (i = 0; i < numchunks; i++)
	{
		/* Convert to 64-bit unsigned integer */    
		uint64_t x = flagblock64[i];
		    
		/*  Employ bit population counter algorithm from Henry S. Warren's
			*  "Hacker's Delight" book, chapter 5.   Added one more shift-n-add
			*  to accomdate 64 bit values.
			*/
        
		x = x - ((x >> 1) & 0x5555555555555555ULL);
		x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
		x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
		x = x + (x >> 8);
		x = x + (x >> 16);
		x = x + (x >> 32);

		it += (x & 0x000000000000003FULL);
        
	}

#endif

	return it;
}

void count_line_special(thread_soedata_t *thread_data)
{
	//extract stuff from the thread data structure
	soe_staticdata_t *sdata = &thread_data->sdata;
	uint32_t current_line = thread_data->current_line;
	uint8_t *line = sdata->lines[current_line];	
	uint64_t numlinebytes = sdata->numlinebytes;
	uint64_t lowlimit = sdata->lowlimit;
	uint64_t prodN = sdata->prodN;
	uint64_t *flagblock64 = (uint64_t *)line;
	uint64_t i, k, it, lower, upper;
	int ix;
	int64_t start, stop;
    uint8_t* masks = sdata->masks;
    uint8_t* nmasks = sdata->nmasks;

	//zero out any bits below the requested range
	for (i=lowlimit + sdata->rclass[current_line], ix=0; i < sdata->orig_llimit; i += prodN, ix++)
		line[ix >> 3] &= masks[ix & 7];
	
	//and any high bits above the requested range
	for (i=sdata->highlimit + sdata->rclass[current_line] - prodN, ix=0; i > sdata->orig_hlimit; i -= prodN, ix++)
		line[numlinebytes - 1 - (ix >> 3)] &= masks[7 - (ix & 7)];

	//count each block of 1e9
	lower = sdata->orig_llimit;
	upper = lower;
	k = 0;
	thread_data->linecount = 0;
	while (upper != sdata->orig_hlimit)
	{
		//set the bounds for the next batch
		upper = upper + 1000000000; 
		if (upper > sdata->orig_hlimit)
			upper = sdata->orig_hlimit;

		//find the starting byte number.  first find the number of bits between the current lower
		//limit and the start of the line.
		start = (int64_t)((lower - lowlimit) / prodN);

		//we'll be counting in 64 bit chunks, so compute how many 64 bit chunks this is
		start /= 64;
		
		//start a little before the range, to account for rounding errors
		start -= 2;

		if (start < 0) start = 0;

		//find the stopping byte number: first find the number of bits between the current upper
		//limit and the start of the line.
		stop = (int64_t)((upper - lowlimit) / prodN);

		//we'll be counting in 64 bit chunks, so compute how many 64 bit chunks this is
		stop /= 64;
		
		//stop a little after the range, to account for rounding errors
		stop += 2;

		if (stop > (numlinebytes >> 3)) stop = (numlinebytes >> 3);

		//count these bytes
		it = 0;
		for (ix = start; ix < stop; ix++)
		{
			/* Convert to 64-bit unsigned integer */    
			uint64_t x = flagblock64[ix];
		    
			/*  Employ bit population counter algorithm from Henry S. Warren's
			 *  "Hacker's Delight" book, chapter 5.   Added one more shift-n-add
			 *  to accomdate 64 bit values.
			 */
			x = x - ((x >> 1) & 0x5555555555555555ULL);
			x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
			x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
			x = x + (x >> 8);
			x = x + (x >> 16);
			x = x + (x >> 32);

			it += (x & 0x000000000000003FULL);
		}

		//then correct the counts
		//zero out any bits below the requested range
		for (i= (start * 64) * prodN + sdata->rclass[current_line] + lowlimit, ix=0; i < lower; i += prodN, ix++)
		{
			if (line[(ix >> 3) + (start << 3)] & nmasks[ix & 7])
				it--;
		}
		
		//and any high bits above the requested range
		for (i=(stop * 64) * prodN + sdata->rclass[current_line] - prodN + lowlimit, ix=0; i > upper; i -= prodN, ix++)
		{
			if (line[(stop << 3) - 1 - (ix >> 3)] & nmasks[7 - (ix & 7)])
				it--;

		}
		
		//add the count to the special array
		thread_data->ddata.special_count[k] = it;
		thread_data->linecount += it;
		k++;
		lower = upper;
	}

}
