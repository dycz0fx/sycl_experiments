KEIKO:= ../../keiko
PAPIDIR = /home/lstewa4x/w/papi-5.6.0


#PAPIDIREC = $(PAPIDIR)

CC=icc

INCLUDES:=-I. -I../common_includes/
ifneq (${PAPIDIREC},)
INCLUDES+= -I${PAPIDIREC}/src
endif

# regular 
# The -wd line ignores certain compiler diagnostics
# 177 is "declared but never referenced"
# 593 is "set but never used"
CFLAGS:=-Wall -std=c99 -axCORE-AVX512 -Wextra -w3 -wd1418,981,1419,177,593 -g -O1 \
-pthread ${EXTRA_PARAMS} ${INCLUDES}
ifneq (${PAPIDIREC},)
CFLAGS+= -DUSE_PAPI
endif
# debug


LDFLAGS=${PAPIDIREC}/src/libpapi.a ${PAPIDIREC}/src/libpfm4/lib/libpfm.a -lpthread -lrt


all: libmovget.a

# builds a version of the library with Simics magic markers 0x4321 and 0x5321
# instead of SSC marks 0x44332211 and 0x55332211 - useful for Calypso EDS
simx: libmovget-simx.a

HFILES:=ActiveMessage.h amfunctions.h barrier.h bigbuf.h LWT.h magic-instruction.h magic_marker.h misc.h movget.h movgetmemcpy.h rao.h shadowmemcpy.h wtime.h
ifneq (${PAPIDIREC},)
HFILES+= papiutil.h 
endif

LIBCOMMON:= ActiveMessage.o barrier.o amfunctions.o movget.o LWT.o movgetmemcpy.o shadowmemcpy.o bigbuf.o
LIBFILES:= ${LIBCOMMON} misc.o
LIBSIMICS:= ${LIBCOMMON} misc-simx.o
ifneq (${PAPIDIREC},)
LIBFILES+= papiutil.o
LIBSIMICS+= papiutil.o
endif

$(LIBFILES): $(HFILES)

misc-simx.o: misc.c
	$(CC) -c ${CFLAGS} -DSIMICS -I${KEIKO}/calypso/lib/magic $^ -o $@

libmovget.a: ${LIBFILES}
	-rm -f $@
	ar r $@ $^

libmovget-simx.a: ${LIBSIMICS}
	-rm -f $@
	ar r $@ $^

movgetmemcpytest.x: movgetmemcpytest.o movgetmemcpy.o
		$(CC) $^ -o $@

clean:
	rm -rf *.o *.s *.a *~
