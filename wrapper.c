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
#include "gmp.h"
#include "ytools.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include "threadpool.h"

void compute_prps_dispatch(void *vptr)
{
    tpool_t *tdata = (tpool_t *)vptr;
    soe_userdata_t *t = (soe_userdata_t *)tdata->user_data;
    soe_staticdata_t *sdata = t->sdata;

    if (sdata->sync_count < sdata->THREADS)
    {
        tdata->work_fcn_id = 0;
        sdata->sync_count++;
    }
    else
    {
        tdata->work_fcn_id = tdata->num_work_fcn;
    }

    return;
}

void compute_prps_work_fcn(void *vptr)
{
    tpool_t *tdata = (tpool_t *)vptr;
    soe_userdata_t *udata = (soe_userdata_t *)tdata->user_data;
    soe_staticdata_t *sdata = udata->sdata;
    thread_soedata_t *t = &udata->ddata[tdata->tindex];
    int i;
    
    t->linecount = 0;
    for (i = t->startid; i < t->stopid; i++)
    {
        if (((i & 127) == 0) && (sdata->VFLAG > 0))
        {
            printf("thread %d progress: %d%%\r", tdata->tindex, 
                (int)((double)(i - t->startid) / (double)(t->stopid - t->startid) * 100.0));
            fflush(stdout);
        }

        mpz_add_ui(t->tmpz, t->offset, t->ddata.primes[i - t->startid]);
        if ((mpz_cmp(t->tmpz, t->lowlimit) >= 0) && (mpz_cmp(t->highlimit, t->tmpz) >= 0))
        {
            //if (mpz_extrastrongbpsw_prp(t->tmpz))
            if (mpz_probab_prime_p(t->tmpz, 1))
            {
                t->ddata.primes[t->linecount++] = t->ddata.primes[i - t->startid];
            }
        }
    }

    return;
}

soe_staticdata_t* soe_init(int vflag, int threads, int blocksize)
{
    soe_staticdata_t* sdata;

    sdata = (soe_staticdata_t*)malloc(sizeof(soe_staticdata_t));

    // bootstrap the sieve
    sdata->sieve_p = (uint32_t*)xmalloc(65536 * sizeof(uint32_t));
    sdata->num_sp = tiny_soe(65536, sdata->sieve_p);

    sdata->VFLAG = vflag;
    sdata->THREADS = threads;
    if (blocksize > 1024)
        sdata->SOEBLOCKSIZE = blocksize;
    else
        sdata->SOEBLOCKSIZE = blocksize << 10;
    return sdata;
}

void soe_finalize(soe_staticdata_t* sdata)
{
    free(sdata->sieve_p);
    return;
}

uint64_t *GetPRIMESRange(soe_staticdata_t* sdata, 
	mpz_t *offset, uint64_t lowlimit, uint64_t highlimit, uint64_t *num_p)
{
	uint64_t i;
	uint64_t hi_est, lo_est;
	uint64_t maxrange = 10000000000ULL;
	uint64_t *primes = NULL;
	
	//reallocate output array based on conservative estimate of the number of 
	//primes in the interval
	if (offset != NULL)
	{
		i = (highlimit - lowlimit);
		primes = (uint64_t *)realloc(primes, (size_t) (i * sizeof(uint64_t)));
		if (primes == NULL)
		{
            if (offset == NULL)
            {
                printf("unable to allocate %" PRIu64 " bytes for range %" PRIu64 " to %" PRIu64 "\n",
                    (uint64_t)(i * sizeof(uint64_t)), lowlimit, highlimit);
            }
            else
            {
                printf("unable to allocate %" PRIu64 " bytes \n",
                    (uint64_t)(i * sizeof(uint64_t)));
            }
			exit(1);
		}
	}
	else
	{
		hi_est = (uint64_t)(highlimit/log((double)highlimit));
		if (lowlimit > 1)
			lo_est = (uint64_t)(lowlimit/log((double)lowlimit));
		else
			lo_est = 0;

		i = (uint64_t)((double)(hi_est - lo_est) * 1.25);
		primes = (uint64_t *)xrealloc(primes, (size_t) (i * sizeof(uint64_t)));
	}

	//check for really big ranges ('big' is different here than when we are counting
	//primes because there are higher memory demands when computing primes)
	if ((highlimit - lowlimit) > maxrange)
	{
		uint64_t tmpl, tmph, tmpcount = 0;
		uint32_t num_ranges = (uint32_t)((highlimit - lowlimit) / maxrange);
		uint64_t remainder = (highlimit - lowlimit) % maxrange;
		uint32_t j;
				
		sdata->GLOBAL_OFFSET = 0;
		tmpl = lowlimit;
        // maxrange - 1, so that we don't count the upper
        // limit twice (again on the next iteration's lower bound).
		tmph = lowlimit + maxrange - 1;
		for (j = 0; j < num_ranges; j++)
		{
			tmpcount += spSOE(sdata, offset, tmpl, &tmph, 0, primes);
			tmpl += maxrange;
			tmph = tmpl + maxrange - 1;
            sdata->GLOBAL_OFFSET = tmpcount;
		}
				
		tmph = tmpl + remainder;
		tmpcount += spSOE(sdata, offset, tmpl, &tmph, 0, primes);
		*num_p = tmpcount;
	}
	else
	{
		//find the primes in the interval
        sdata->GLOBAL_OFFSET = 0;
        if (sdata->VFLAG > 1)
        {
            printf("generating primes in range %" PRIu64 " : %" PRIu64 "\n", 
                lowlimit, highlimit);
        }
		*num_p = spSOE(sdata, offset, lowlimit, &highlimit, 0, primes);
	}

	return primes;
}

uint64_t *soe_wrapper(soe_staticdata_t* sdata, uint64_t lowlimit, uint64_t highlimit, 
    int count, uint64_t* num_p, int PRIMES_TO_FILE, int PRIMES_TO_SCREEN)
{
	//public interface to the sieve.  
	uint64_t retval, tmpl, tmph, i;
	uint32_t max_p;	
	
	uint64_t *primes = NULL;

    sdata->only_count = count;

    if (highlimit < lowlimit)
    {
        printf("error: lowlimit must be less than highlimit\n");
        *num_p = 0;
        return primes;
    }

	if (highlimit > (sdata->sieve_p[sdata->num_sp-1] * sdata->sieve_p[sdata->num_sp-1]))
	{
		//then we need to generate more sieving primes
		uint32_t range_est;

		//allocate array based on conservative estimate of the number of 
		//primes in the interval	
		max_p = (uint32_t)sqrt((int64_t)(highlimit)) + 65536;
		range_est = (uint32_t)estimate_primes_in_range(0, (uint64_t)max_p);

        if (sdata->VFLAG > 1)
        {
            printf("generating more sieving primes in range 0 : %u \n", max_p);
            printf("allocating %u bytes \n", range_est);
        }

        sdata->sieve_p = (uint32_t *)xrealloc(sdata->sieve_p, 
            (size_t) (range_est * sizeof(uint32_t)));

		//find the sieving primes using the seed primes
        sdata->NO_STORE = 0;
		primes = GetPRIMESRange(sdata, NULL, 0, max_p, &retval);

        if (sdata->VFLAG > 1)
        {
            printf("found %u sieving primes\n", (uint32_t)retval);
        }

        for (i = 0; i < retval; i++)
        {
            sdata->sieve_p[i] = (uint32_t)primes[i];
        }

        sdata->num_sp = (uint32_t)retval;
		free(primes);
		primes = NULL;
	}

	if (count)
	{
		//this needs to be a range of at least 1e6
		if ((highlimit - lowlimit) < 1000000)
		{
			//go and get a new range.
			tmpl = lowlimit;
			tmph = tmpl + 1000000;

			//since this is a small range, we need to 
			//find a bigger range and count them.
			primes = GetPRIMESRange(sdata, NULL, tmpl, tmph, &retval);

			*num_p = 0;
			//count how many are in the original range of interest
			for (i = 0; i < retval; i++)
			{
                if ((primes[i] >= lowlimit) && (primes[i] <= highlimit))
                {
                    (*num_p)++;
                }
			}
			free(primes);
			primes = NULL;
		}
		else
		{
			//check for really big ranges
			uint64_t maxrange = 100000000000ULL;

			if ((highlimit - lowlimit) > maxrange)
			{
				uint32_t num_ranges = (uint32_t)((highlimit - lowlimit) / maxrange);
				uint64_t remainder = (highlimit - lowlimit) % maxrange;
				uint32_t j;
				//to get time per range
				double t_time;
				struct timeval start, stop;
				
				*num_p = 0;
				tmpl = lowlimit;
                // maxrange - 1, so that we don't count the upper
                // limit twice (again on the next iteration's lower bound).
				tmph = lowlimit + maxrange - 1;
				gettimeofday (&start, NULL);

				for (j = 0; j < num_ranges; j++)
				{
					*num_p += spSOE(sdata, NULL, tmpl, &tmph, count, NULL);

					gettimeofday (&stop, NULL);
                    t_time = ytools_difftime(&start, &stop);

                    if (sdata->VFLAG > 1)
                    {
                        printf("so far, found %" PRIu64 " primes in %1.1f seconds\n", *num_p, t_time);
                    }
					tmpl += maxrange;
					tmph = tmpl + maxrange - 1;
				}
				
				if (remainder > 0)
				{
					tmph = tmpl + remainder;
					*num_p += spSOE(sdata, NULL, tmpl, &tmph, count, NULL);
				}
                if (sdata->VFLAG > 1)
                {
                    printf("so far, found %" PRIu64 " primes\n", *num_p);
                }
			}
			else
			{
				//we're in a sweet spot already, just get the requested range
				*num_p = spSOE(sdata, NULL, lowlimit, &highlimit, count, NULL);
			}
		}

	}
	else
	{
		tmpl = lowlimit;
		tmph = highlimit;

		//this needs to be a range of at least 1e6
		if ((tmph - tmpl) < 1000000)
		{
			//there is slack built into the sieve limit, so go ahead and increase
			//the size of the interval to make it at least 1e6.
			tmph = tmpl + 1000000;

			//since this is a small range, we need to 
			//find a bigger range and count them.
			primes = GetPRIMESRange(sdata, NULL, tmpl, tmph, &retval);
			*num_p = 0;
			for (i = 0; i < retval; i++)
			{
				if (primes[i] >= lowlimit && primes[i] <= highlimit)
					(*num_p)++;
			}

		}
		else
		{
			//we don't need to mess with the requested range,
			//so GetPRIMESRange will return the requested range directly
			//and the count will be in NUM_P
			primes = GetPRIMESRange(sdata, NULL, lowlimit, highlimit, num_p);
		}

		// now dump the requested range of primes to a file, or the
		// screen, both, or neither, depending on the state of a couple
		// global configuration variables
		if (PRIMES_TO_FILE)
		{
			FILE *out;
			out = fopen("primes.dat","w");
			if (out == NULL)
			{
				printf("fopen error: %s\n", strerror(errno));
				printf("can't open primes.dat for writing\n");
			}
			else
			{
				for (i = 0; i < *num_p; i++)
				{
                    if ((primes[i] >= lowlimit) && (primes[i] <= highlimit))
                    {
                        fprintf(out, "%" PRIu64 "\n", primes[i]);
                    }
				}
				fclose(out);
			}
		}

		if (PRIMES_TO_SCREEN)
		{
			for (i = 0; i < *num_p; i++)
			{
                if ((primes[i] >= lowlimit) && (primes[i] <= highlimit))
                {
                    printf("%" PRIu64 " ", primes[i]);
                }
			}
			printf("\n");
		}			
	}

	return primes;
}

uint64_t *sieve_to_depth(soe_staticdata_t* sdata,
	mpz_t lowlimit, mpz_t highlimit, int count, int num_witnesses, 
    uint64_t sieve_limit, uint64_t *num_p,
    int PRIMES_TO_FILE, int PRIMES_TO_SCREEN)
{
	// public interface to a routine which will sieve a range of integers
	// with the supplied primes and either count or compute the values
	// that survive.  Basically, it is just the sieve, but with no
	// guareentees that what survives the sieving is prime.  The idea is to 
	// remove cheap composites.
	uint64_t retval, i, range, tmpl, tmph;
	uint64_t *values = NULL;
	mpz_t tmpz;
	mpz_t *offset;

	if (mpz_cmp(highlimit, lowlimit) <= 0)
	{
		printf("error: lowlimit must be less than highlimit\n");
		*num_p = 0;
		return values;
	}	

	offset = (mpz_t *)malloc(sizeof(mpz_t));
	mpz_init(tmpz);
	mpz_init(*offset);
	mpz_set(*offset, lowlimit);
	mpz_sub(tmpz, highlimit, lowlimit);
	range = mpz_get_ui(tmpz);

	if (count)
	{
		//this needs to be a range of at least 1e6
		if (range < 1000000)
		{
			//go and get a new range.
			tmpl = 0;
			tmph = 1000000;

			//since this is a small range, we need to 
			//find a bigger range and count them.
			values = GetPRIMESRange(sdata, offset, tmpl, tmph, &retval);

			*num_p = 0;
			//count how many are in the original range of interest
			for (i = 0; i < retval; i++)
			{
				mpz_add_ui(tmpz, *offset, values[i]);
                if ((mpz_cmp(tmpz, lowlimit) >= 0) && (mpz_cmp(highlimit, tmpz) >= 0))
                {
                    (*num_p)++;
                }
			}
			free(values);
			values = NULL;
		}
		else
		{
			//check for really big ranges
			uint64_t maxrange = 100000000000ULL;

			if (range > maxrange)
			{
				uint32_t num_ranges = (uint32_t)(range / maxrange);
				uint64_t remainder = range % maxrange;
				uint32_t j;
				
				*num_p = 0;
				tmpl = 0;
                // maxrange - 1, so that we don't count the upper
                // limit twice (again on the next iteration's lower bound).
				tmph = tmpl + maxrange - 1;
				for (j = 0; j < num_ranges; j++)
				{
					*num_p += spSOE(sdata, offset, tmpl, &tmph, 1, NULL);
                    if (sdata->VFLAG > 1)
                    {
                        printf("so far, found %" PRIu64 " primes\n", *num_p);
                    }
					tmpl += maxrange;
					tmph = tmpl + maxrange - 1;
				}
				
				if (remainder > 0)
				{
					tmph = tmpl + remainder;
					*num_p += spSOE(sdata, offset, tmpl, &tmph, 1, NULL);
				}
                if (sdata->VFLAG > 1)
                {
                    printf("so far, found %" PRIu64 " primes\n", *num_p);
                }
			}
			else
			{
				//we're in a sweet spot already, just get the requested range
				*num_p = spSOE(sdata, offset, 0, &range, 1, NULL);
			}
		}

	}
	else
	{
		// this needs to be a range of at least 1e6
		if (range < 1000000)
		{
			//there is slack built into the sieve limit, so go ahead and increase
			//the size of the interval to make it at least 1e6.
			tmpl = 0;
			tmph = tmpl + 1000000;

			// since this is a small range, we need to 
			// find a bigger range and count them.
			values = GetPRIMESRange(sdata, offset, tmpl, tmph, &retval);
			*num_p = 0;
			for (i = 0; i < retval; i++)
			{
				mpz_add_ui(tmpz, *offset, values[i]);
                if ((mpz_cmp(tmpz, lowlimit) >= 0) && (mpz_cmp(highlimit, tmpz) >= 0))
                {
                    (*num_p)++;
                }
			}
		}
		else
		{
			//we don't need to mess with the requested range,
			//so GetPRIMESRange will return the requested range directly
			//and the count will be in NUM_P
			values = GetPRIMESRange(sdata, offset, 0, range, num_p);
		}

		if (num_witnesses > 0)
		{
			thread_soedata_t *thread_data;		//an array of thread data objects
			uint32_t lastid;
			int j;

            // threading structures
            tpool_t *tpool_data;
            soe_userdata_t udata;

			//allocate thread data structure
			thread_data = (thread_soedata_t *)malloc(sdata->THREADS * sizeof(thread_soedata_t));
			
			// conduct PRP tests on all surviving values
            if (sdata->VFLAG > 0)
            {
                printf("starting PRP tests with %d witnesses on "
                    "%" PRIu64 " surviving candidates using %d threads\n",
                    num_witnesses, *num_p, sdata->THREADS);
            }

			range = *num_p / sdata->THREADS;
			lastid = 0;

			// divvy up the range
			for (j = 0; j < sdata->THREADS; j++)
			{
				thread_soedata_t *t = thread_data + j;
				
				t->startid = lastid;
				t->stopid = t->startid + range;
				lastid = t->stopid;

                if (sdata->VFLAG > 2)
                {
                    printf("thread %d computing PRPs from %u to %u\n",
                        (int)j, t->startid, t->stopid);
                }
			}

			// the last one gets any leftover
            if (thread_data[sdata->THREADS - 1].stopid != (uint32_t)*num_p)
            {
                thread_data[sdata->THREADS - 1].stopid = (uint32_t)*num_p;
            }

			// allocate space for stuff in the threads
			for (j = 0; j < sdata->THREADS; j++)
			{
				thread_soedata_t *t = thread_data + j;

				mpz_init(t->tmpz);
				mpz_init(t->offset);
				mpz_init(t->lowlimit);
				mpz_init(t->highlimit);
				mpz_set(t->offset, *offset);
				mpz_set(t->lowlimit, lowlimit);
				mpz_set(t->highlimit, highlimit);
				t->current_line = (uint64_t)num_witnesses;

				t->ddata.primes = (uint64_t *)malloc((t->stopid - t->startid) * sizeof(uint64_t));
                for (i = t->startid; i < t->stopid; i++)
                {
                    t->ddata.primes[i - t->startid] = values[i];
                }
			}

			// now run with the threads.  don't really need the 
            // threadpool since we are statically dividing up the range
            // to test, but it is easy so we use it.
            udata.sdata = &thread_data->sdata;
            udata.ddata = thread_data;
            tpool_data = tpool_setup(sdata->THREADS, NULL, NULL, NULL,
                &compute_prps_dispatch, &udata);

            thread_data->sdata.sync_count = 0;
            tpool_add_work_fcn(tpool_data, &compute_prps_work_fcn);
            tpool_go(tpool_data);

            free(tpool_data);

			// combine results and free stuff
			retval = 0;
            for (i = 0; i < sdata->THREADS; i++)
            {
                thread_soedata_t* t = thread_data + i;

                for (j = 0; j < t->linecount; j++)
                {
                    values[retval++] = t->ddata.primes[j];
                }

                free(t->ddata.primes);
                mpz_clear(t->tmpz);
                mpz_clear(t->offset);
                mpz_clear(t->lowlimit);
                mpz_clear(t->highlimit);
            }

			free(thread_data);

			*num_p = retval;
            if (sdata->VFLAG > 0)
            {
                printf("found %" PRIu64 " PRPs\n", *num_p);
            }
			
		}

		if (mpz_cmp(*offset, lowlimit) != 0)
		{
			// sieving needed to change the lower sieve limit.  adjust the returned
			// values accordingly.
			uint64_t a;
			mpz_sub(tmpz, lowlimit, *offset);
			a = mpz_get_ui(tmpz);

            for (i = 0; i < *num_p; i++)
            {
                values[i] -= a;
            }
		}

		// now dump the requested range of primes to a file, or the
		// screen, both, or neither, depending on the state of a couple
		// global configuration variables
		if (PRIMES_TO_FILE)
		{
			FILE *out;
			if (num_witnesses > 0)
				out = fopen("prp_values.dat", "w");
			else
				out = fopen("sieved_values.dat","w");

			if (out == NULL)
			{
				printf("fopen error: %s\n", strerror(errno));
				printf("can't open file for writing\n");
			}
			else
			{
				for (i = 0; i < *num_p; i++)
				{
					//mpz_add_ui(tmpz, *offset, values[i]);
					mpz_add_ui(tmpz, lowlimit, values[i]);
                    if ((mpz_cmp(tmpz, lowlimit) >= 0) && (mpz_cmp(highlimit, tmpz) >= 0))
                    {
                        char* buf = mpz_get_str(NULL, 10, tmpz);
                        fprintf(out, "%s\n", buf);
                        free(buf);
                    }
				}
				fclose(out);
			}
		}

		if (PRIMES_TO_SCREEN)
		{
			for (i = 0; i < *num_p; i++)
			{
				//mpz_add_ui(tmpz, *offset, values[i]);
				mpz_add_ui(tmpz, lowlimit, values[i]);
                if ((mpz_cmp(tmpz, lowlimit) >= 0) && (mpz_cmp(highlimit, tmpz) >= 0))
                {
                    gmp_printf("%Zd\n", tmpz);
                }
			}
			printf("\n");
		}			
	}

	mpz_clear(tmpz);
	mpz_clear(*offset);
	free(offset);

	return values;
}


