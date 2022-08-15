/* bigbuf.h
   lawrencex.stewart@intel.com
*/
#ifndef BIGBUF_H
#define BIGBUF_H

#include <stdlib.h>

/* uses mmap to map a 1 GB hunk of virtual memory with contiguous 
 * physical memory
 */

void bigbuf_init();

/* shuts down the buffer and frees the memory, no further references will
 * work
 */
void bigbuf_free();

/* really an sbrk allocator.  There is no free. */

void* bigbuf_alloc(size_t size, size_t align);

#endif
