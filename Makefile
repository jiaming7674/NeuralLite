CC=/usr/bin/g++
SDIR=./neural/src
ODIR=./bin
INC=-I./neural/inc -I/ucrt64/include/eigen3
TARGET=run
CFLAGS=-lpthread -O4
SRCS=network.cpp core.cpp layers/activation_layer.cpp layers/fc_layer.cpp
_OBJS=$(patsubst %.cpp, ${ODIR}/%.o, $(notdir ${SRCS}))

MAIN=xor

.PHONY: clean

${TARGET}: ${SRCS} ${MAIN}
	${CC} ${CFLAGS} -o ${TARGET} ${_OBJS} ./bin/${MAIN}.o
	@echo "Create TARGET done !!"


%.cpp:
	${CC} ${CFLAGS} -c ${INC} ${SDIR}/$@ -o ${ODIR}/$(notdir $*).o


${MAIN}:
	${CC} ${CFLAGS} -c ${INC} ./examples/$@.cpp -o ${ODIR}/$(notdir $@).o


clean:
	-rm -rf ${ODIR}/*.o
	-rm ${TARGET}