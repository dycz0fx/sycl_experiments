/* bigbuf.c
 * lawrencex.stewart@intel.com
 */
#define _GNU_SOURCE             /* See feature_test_macros(7) */

#include "bigbuf.h"
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "amfunctions.h"

int bigbuf_initialized = 0;
int bigbuf_fd = 0;
void *bigbuf_base = NULL;
void *bigbuf_end = NULL;
void *bigbuf_brk = NULL;
int bigbuf_zerofill_needed = 0;

void bigbuf_init()
{
  char *hugefs;
  char hugefn[128];
  if (bigbuf_initialized != 0) return;
  hugefs = getenv("HUGEFS");
  if (hugefs == NULL) hugefs = "/mnt/huge";
  snprintf(hugefn, 128, "%s/bigbuf", hugefs);
  bigbuf_fd = open(hugefn, O_CREAT | O_RDWR, 0755);
  if (bigbuf_fd > 0) {
    bigbuf_base = mmap(NULL, 1L<<30,
		       PROT_READ | PROT_WRITE,
		       MAP_PRIVATE |MAP_HUGETLB,
		       bigbuf_fd, 0L);
    if (bigbuf_base != MAP_FAILED) {
      assert(bigbuf_base != NULL);
      bigbuf_end = (void *) ((uintptr_t) bigbuf_base + (1L<<30));
      bigbuf_initialized = 1;
      bigbuf_brk = bigbuf_base;
      memset(bigbuf_base, 0, 4096);  /* force memory allocation */
      raw_flush((__m512i *) bigbuf_base, 4096);
      printf("bigbuf hugetlbfs suceeded\n");
      bigbuf_zerofill_needed = 0;
      return;
    }
    perror("bigbuf hugetlbfs failed");
  }
  bigbuf_base = mmap(NULL, 1L<<28,
		     PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
		     -1, 0L);
  if (bigbuf_base != MAP_FAILED) {
    assert(bigbuf_base != NULL);
    bigbuf_end = (void *) ((uintptr_t) bigbuf_base + (1L<<28));
    bigbuf_initialized = 1;
    bigbuf_brk = bigbuf_base;
    printf("bigbuf map_anon map_hugetlb suceeded\n");
    bigbuf_zerofill_needed = 1;  // maybe not, but be safe
    return;
  }
  bigbuf_base = mmap(NULL, 1L<<28,
		     PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS,
		     -1, 0L);
  if (bigbuf_base != MAP_FAILED) {
    assert(bigbuf_base != NULL);
    bigbuf_end = (void *) ((uintptr_t) bigbuf_base + (1L<<28));
    bigbuf_initialized = 1;
    bigbuf_brk = bigbuf_base;
    printf("bigbuf map_anon suceeded\n");
    bigbuf_zerofill_needed = 1;
    return;
  }
  perror("bigbuf mmap failed");
}


/* shuts down the buffer and frees the memory, no further references will
 * work
 */
void bigbuf_free()
{
  assert(bigbuf_base != NULL);
  munmap(bigbuf_base, 1L<<30);
  close(bigbuf_fd);
  bigbuf_fd = -1;
  bigbuf_base = NULL;
  bigbuf_brk = NULL;
  bigbuf_end = NULL;
  
}


/* really an sbrk allocator.  There is no free. */

void* bigbuf_alloc(size_t size, size_t align)
{
  void *new = (void *) (((uintptr_t) bigbuf_brk + align - 1L) & ~(align - 1L));
  if (((uintptr_t) new + size) > (uintptr_t) bigbuf_end) return (NULL);
  bigbuf_brk = (void *) ((uintptr_t) new + size);
  if (bigbuf_zerofill_needed) {
    memset(new, 0, size);
    raw_flush((__m512i *) new, size);
  }
  return(new);
}

