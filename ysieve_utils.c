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

//user dimis:
//http://cboard.cprogramming.com/cplusplus-programming/
//101085-how-measure-time-multi-core-machines-pthreads.html
//
double ysieve_difftime (struct timeval * start, struct timeval * end)
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

