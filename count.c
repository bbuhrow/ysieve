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

uint64_t count_8_bytes(soe_staticdata_t* sdata,
    uint64_t pcount, uint64_t byte_offset);
uint64_t count_8_bytes_bmi2(soe_staticdata_t* sdata,
    uint64_t pcount, uint64_t byte_offset);

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

uint64_t count_twins(soe_staticdata_t* sdata)
{
    int i;
    uint64_t lowlimit = sdata->lowlimit;
    uint64_t count = 0;
    uint64_t prodN = sdata->prodN;
    uint64_t numchunks = (sdata->orig_hlimit - lowlimit) / (64 * prodN) + 1;

	// counting twins and other prime constellations requires
	// all of the sieve lines in memory for reording.
#if defined(USE_BMI2) || defined(USE_AVX512F)
    if ((sdata->has_bmi2) && (sdata->numclasses <= 48))
    {
        for (i = 0; i < numchunks; i++)
        {
            count = count_8_bytes_bmi2(sdata, count, (uint64_t)i * 8);

            // if searching for prime constellations, then we need to look at the last 
            // flags of this block of 8 bytes and the first flags of the next one.
            // we do this by loading the trailing bits of this block into a carry
            // register.  The rest is handled by the block analysis function.
            // For the first block in a multi-threaded run, we should also
            // be receiving carry data from the previous thread in order to
            // maintain continuity across threads.

            if ((sdata->analysis == 2) && (sdata->is_main_sieve == 1))
            {
                uint8_t* lastline = sdata->lines[sdata->numclasses - 1];
                uint8_t* firstline = sdata->lines[0];

                if ((i + 1) < numchunks)
                {
                    uint8_t lastflag = lastline[i * 8 + 7] & 0x80;
                    uint8_t firstflag = firstline[i * 8 + 8] & 0x1;

                    if (lastflag && firstflag)
                    {
                        count++;
                    }
                }
                else if ((i + 1) < sdata->numlinebytes)
                {
                    // this thread is done but if there is more data after
                    // this thread's chunk then check between thread boundaries.
                    uint8_t lastflag = lastline[i * 8 + 7] & 0x80;
                    uint8_t firstflag = firstline[i * 8 + 8] & 0x1;

                    if (lastflag && firstflag)
                    {
                        count++;
                    }
                }
            }
        }
    }
    else
    {
        // if we don't have BMI2, or numclasses > 48, or we're doing a more
        // complicated analysis, then we should end up here.
        
        for (i = 0; i < numchunks; i++)
        {
            count = count_8_bytes(sdata, count, (uint64_t)i * 8);

            // if searching for prime constellations, then we need to look at the last 
            // flags of this block of 8 bytes and the first flags of the next one.
            // we do this by loading the trailing bits of this block into a carry
            // register.  The rest is handled by the block analysis function.
            // For the first block in a multi-threaded run, we should also
            // be receiving carry data from the previous thread in order to
            // maintain continuity across threads.

            if ((sdata->analysis == 2) && (sdata->is_main_sieve == 1))
            {
                uint8_t* lastline = sdata->lines[sdata->numclasses - 1];
                uint8_t* firstline = sdata->lines[0];

                if ((i + 1) < numchunks)
                {
                    uint8_t lastflag = lastline[i * 8 + 7] & 0x80;
                    uint8_t firstflag = firstline[i * 8 + 8] & 0x1;

                    if (lastflag && firstflag)
                    {
                        count++;
                    }
                }
                else if ((i + 1) < sdata->numlinebytes)
                {
                    // this thread is done but if there is more data after
                    // this thread's chunk then check between thread boundaries.
                    uint8_t lastflag = lastline[i * 8 + 7] & 0x80;
                    uint8_t firstflag = firstline[i * 8 + 8] & 0x1;

                    if (lastflag && firstflag)
                    {
                        count++;
                    }
                }
            }
        }
    }
#else
    for (i = t->startid; i < t->stopid; i += 8)
    {
        t->linecount = compute_8_bytes(sdata, t->linecount, t->ddata.primes, i);
    }
#endif

	return count;
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

uint64_t count_8_bytes(soe_staticdata_t* sdata,
    uint64_t pcount, uint64_t byte_offset)
{
    uint32_t current_line;
    // re-ordering queues supporting arbitrary residue classes.
    uint64_t** pqueues; // [64][48]
    uint32_t pcounts[64];
    int i, j;
    uint32_t nc = sdata->numclasses;
    uint64_t lowlimit = sdata->lowlimit;
    uint64_t prodN = sdata->prodN;
    uint8_t** lines = sdata->lines;
    uint64_t olow = sdata->orig_llimit;
    uint64_t ohigh = sdata->orig_hlimit;
    int GLOBAL_OFFSET = sdata->GLOBAL_OFFSET;

    if ((byte_offset & 32767) == 0)
    {
        if (sdata->VFLAG > 1)
        {
            printf("computing: %d%%\r", (int)
                ((double)byte_offset / (double)(sdata->numlinebytes) * 100.0));
            fflush(stdout);
        }
    }

    pqueues = (uint64_t**)xmalloc(64 * sizeof(uint64_t*));
    for (i = 0; i < 64; i++)
    {
        pqueues[i] = (uint64_t*)xmalloc(sdata->numclasses * sizeof(uint64_t));
    }

    // Compute the primes using ctz on the 64-bit words but push the results
    // into 64 different queues depending on the bit position.  Then
    // we pull from the queues in order while storing into the primes array.
    // This time the bottleneck is mostly in the queue-based sorting
    // and associated memory operations, so we don't bother with
    // switching between branch-free inner loops or not.
    memset(pcounts, 0, 64 * sizeof(uint32_t));

    lowlimit += byte_offset * 8 * prodN;
    for (current_line = 0; current_line < nc; current_line++)
    {
        uint64_t* line64 = (uint64_t*)lines[current_line];
        uint64_t flags64 = line64[byte_offset / 8];

        while (flags64 > 0)
        {
            uint64_t pos = _trail_zcnt64(flags64);
            uint64_t prime = lowlimit + pos * prodN + sdata->rclass[current_line];

            if ((prime >= olow) && (prime <= ohigh))
            {
                pqueues[pos][pcounts[pos]] = prime;
                pcounts[pos]++;
            }
            flags64 ^= (1ULL << pos);
        }
    }

    for (i = 0; i < 64; i++)
    {

        // search for twins and only load the leading element/prime.
        // if depth-based sieving then these are candidate twins.
        if (pcounts[i] > 0)
        {

            for (j = 0; j < pcounts[i] - 1; j++)
            {
                if ((pqueues[i][j + 1] - pqueues[i][j]) == 2)
                {
                    pcount++;
                }
            }
            if (i < 63)
            {
                if (pcounts[i + 1] > 0)
                {
                    if ((pqueues[i + 1][0] - pqueues[i][j]) == 2)
                    {
                        pcount++;
                    }
                }
            }
        }

    }

    for (i = 0; i < 64; i++)
    {
        free(pqueues[i]);
    }
    free(pqueues);

    return pcount;
}



#if defined(USE_BMI2) || defined(USE_AVX512F)

__inline uint64_t interleave_pdep2x32(uint32_t x1, uint32_t x2)
{
    return _pdep_u64(x1, 0x5555555555555555)
        | _pdep_u64(x2, 0xaaaaaaaaaaaaaaaa);
}

__inline uint64_t interleave_pdep_8x8(uint8_t x1,
    uint8_t x2,
    uint8_t x3,
    uint8_t x4,
    uint8_t x5,
    uint8_t x6,
    uint8_t x7,
    uint8_t x8)
{
    return _pdep_u64(x1, 0x0101010101010101ull) |
        _pdep_u64(x2, 0x0202020202020202ull) |
        _pdep_u64(x3, 0x0404040404040404ull) |
        _pdep_u64(x4, 0x0808080808080808ull) |
        _pdep_u64(x5, 0x1010101010101010ull) |
        _pdep_u64(x6, 0x2020202020202020ull) |
        _pdep_u64(x7, 0x4040404040404040ull) |
        _pdep_u64(x8, 0x8080808080808080ull);
}

#define BIT0 0x1
#define BIT1 0x2
#define BIT2 0x4
#define BIT3 0x8
#define BIT4 0x10
#define BIT5 0x20
#define BIT6 0x40
#define BIT7 0x80

uint64_t count_8_bytes_bmi2(soe_staticdata_t* sdata,
    uint64_t pcount, uint64_t byte_offset)
{
    uint32_t nc = sdata->numclasses;
    uint64_t lowlimit = sdata->lowlimit;
    uint8_t** lines = sdata->lines;

    if ((byte_offset & 32767) == 0)
    {
        if (sdata->VFLAG > 1)
        {
            printf("computing: %d%%\r", (int)
                ((double)byte_offset / (double)(sdata->numlinebytes) * 100.0));
            fflush(stdout);
        }
    }

    // AVX2 version, new instructions help quite a bit:
    // use _pdep_u64 to align/interleave bits from multiple bytes, 
    // _blsr_u64 to clear the last set bit, and depending on the 
    // number of residue classes, AVX2 vector load/store operations.

    // here is the 2 line version
    if (nc == 2)
    {
        int i;
        uint32_t last_bit = 0;
        uint32_t* lines32a = (uint32_t*)lines[0];
        uint32_t* lines32b = (uint32_t*)lines[1];
        
        // align the current bytes in next 2 residue classes
        for (i = 0; i < 2; i++)
        {
            uint64_t aligned_flags;

            aligned_flags = interleave_pdep2x32(
                lines32a[byte_offset / 4 + i],
                lines32b[byte_offset / 4 + i]);


            // alternate bits encode potential primes
            // in residue classes 1 and 5.  So twins can 
            // only exist with flags in class 5 followed by 1.
            uint64_t twins = aligned_flags & (aligned_flags >> 1);

            twins &= 0xaaaaaaaaaaaaaaaaULL;

            pcount += _mm_popcnt_u64(twins);
            pcount += (last_bit & aligned_flags);

            last_bit = (aligned_flags >> 63);
        }
    }
    else if (nc == 8)
    {
        int i;
        uint32_t last_bit = 0;

        // align the current bytes in next 8 residue classes
        for (i = 0; i < 8; i++)
        {
            uint64_t aligned_flags;

            aligned_flags = interleave_pdep_8x8(lines[0][byte_offset + i],
                lines[1][byte_offset + i],
                lines[2][byte_offset + i],
                lines[3][byte_offset + i],
                lines[4][byte_offset + i],
                lines[5][byte_offset + i],
                lines[6][byte_offset + i],
                lines[7][byte_offset + i]);

            uint64_t twins = aligned_flags & (aligned_flags >> 1);

            twins &= 0x9494949494949494ULL;

            pcount += _mm_popcnt_u64(twins);
            pcount += (last_bit & aligned_flags);
            
            last_bit = (aligned_flags >> 63);
        }
    }
    else if (nc == 48)
    {
        int i;
        uint32_t last_bit = 0;

        // align the current bytes in next 8 residue classes
        for (i = 0; i < 8; i++)
        {
            uint64_t aligned_flags;

            aligned_flags = interleave_pdep_8x8(lines[0][byte_offset + i],
                lines[1][byte_offset + i],
                lines[2][byte_offset + i],
                lines[3][byte_offset + i],
                lines[4][byte_offset + i],
                lines[5][byte_offset + i],
                lines[6][byte_offset + i],
                lines[7][byte_offset + i]);

            uint64_t twins = aligned_flags & (aligned_flags >> 1);

            twins &= 0x9494949494949494ULL;

            pcount += _mm_popcnt_u64(twins);
            pcount += (last_bit & aligned_flags);

            last_bit = (aligned_flags >> 63);
        }
    }
    else
    {
        uint64_t** pqueues; // [64][48]
        uint32_t pcounts[64];
        int i, j;
        uint64_t prodN = sdata->prodN;
        uint32_t current_line;

        // ordering the bits becomes inefficient with 48 lines because
        // they would need to be dispersed over too great a distance.
        // instead we compute the primes as before but push the results
        // into 64 different queues depending on the bit position.  Then
        // we pull from the queues in order while storing into the primes array.
        // This time the bottleneck is mostly in the queue-based sorting
        // and associated memory operations, so we don't bother with
        // switching between branch-free inner loops or not.
        pqueues = (uint64_t**)xmalloc(64 * sizeof(uint64_t*));
        for (i = 0; i < 64; i++)
        {
            pqueues[i] = (uint64_t*)xmalloc(sdata->numclasses * sizeof(uint64_t));
        }

        // Compute the primes using ctz on the 64-bit words but push the results
        // into 64 different queues depending on the bit position.  Then
        // we pull from the queues in order while storing into the primes array.
        // This time the bottleneck is mostly in the queue-based sorting
        // and associated memory operations, so we don't bother with
        // switching between branch-free inner loops or not.
        memset(pcounts, 0, 64 * sizeof(uint32_t));

        lowlimit += byte_offset * 8 * prodN;
        for (current_line = 0; current_line < nc; current_line++)
        {
            uint64_t* line64 = (uint64_t*)lines[current_line];
            uint64_t flags64 = line64[byte_offset / 8];

            while (flags64 > 0)
            {
                uint64_t pos = _trail_zcnt64(flags64);
                uint64_t prime = lowlimit + pos * prodN + sdata->rclass[current_line];

                pqueues[pos][pcounts[pos]] = prime;
                pcounts[pos]++;
                flags64 ^= (1ULL << pos);
            }
        }

        for (i = 0; i < 64; i++)
        {
            // search for twins and only load the leading element/prime.
            // if depth-based sieving then these are candidate twins.
            if (pcounts[i] > 0)
            {
                for (j = 0; j < pcounts[i] - 1; j++)
                {
                    if ((pqueues[i][j + 1] - pqueues[i][j]) == 2)
                    {
                        pcount++;
                    }
                }
                if (i < 63)
                {
                    if (pcounts[i + 1] > 0)
                    {
                        if ((pqueues[i + 1][0] - pqueues[i][j]) == 2)
                        {
                            pcount++;
                        }
                    }
                }
            }

        }

        for (i = 0; i < 64; i++)
        {
            free(pqueues[i]);
        }
        free(pqueues);


    }

    return pcount;
}

#endif
