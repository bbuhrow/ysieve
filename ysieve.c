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

/*

*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include "calc.h"
#include "cmdOptions.h"
#include "soe.h"
#include "util.h"

// gcc -O2  calc.h calc.c demo.c -o demo  -lgmp -lm


int main(int argc, char** argv)
{
    options_t* options;
    uint64_t start;
    uint64_t stop;
    uint64_t *primes;
    int count;
    uint64_t num_found;
    int haveFile;
    // timing
    double t;
    struct timeval tstart, tstop;
    int SOEBLOCKSIZE = 32768;       // this should be an option in the interface
    soe_staticdata_t* sdata;


    options = initOpt();
    processOpts(argc, argv, options);


    calc_init();
    process_expression(options->startExpression, NULL, 1, 0);
    process_expression(options->stopExpression, NULL, 1, 0);
    calc_finalize();

    sscanf(options->startExpression, "%" PRIu64 "", &start);
    sscanf(options->stopExpression, "%" PRIu64 "", &stop);

    if (strlen(options->outFile) > 0)
    {
        haveFile = 1;
    }
    else
    {
        haveFile = 0;
    }

    if (options->outScreen || haveFile)
    {
        count = 0;
    }
    else
    {
        count = 1;
    }

    printf("starting sieve on bounds %" PRIu64 " : %" PRIu64 "\n", start, stop);

    sdata = soe_init(options->verbosity, options->threads, SOEBLOCKSIZE);

    gettimeofday(&tstart, NULL);
    primes = soe_wrapper(sdata, start, stop, count, &num_found,
        haveFile, options->outScreen);
    printf("Num primes found: %" PRIu64 "\n", num_found);
    gettimeofday(&tstop, NULL);
    t = yafu_difftime(&tstart, &tstop);
    printf("Elapsed time    : %1.6f seconds\n", t);

    soe_finalize(sdata);
    free(sdata);
    align_free(primes);

    return 0;
}
