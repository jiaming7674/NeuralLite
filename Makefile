CC=/ucrt64/bin/g++
SDIR=./neural/src
ODIR=./bin
INC=-I./neural/inc -I/ucrt64/include/eigen3 -I/ucrt64/include
TARGET=run
CFLAGS=-O4
SRCS=network.cpp core.cpp layers/activation_layer.cpp layers/fc_layer.cpp
_OBJS=$(patsubst %.cpp, ${ODIR}/%.o, $(notdir ${SRCS}))
LIB=-lpthread -lraylib -lopengl32 -lwinmm -lgdi32

MAIN=game

.PHONY: clean

${TARGET}: ${SRCS} ${MAIN}
	${CC} ${CFLAGS} -o ${TARGET} ${_OBJS} ./bin/${MAIN}.o -L/ucrt64/lib ${LIB}
	@echo "Create TARGET done !!"


%.cpp:
	${CC} ${CFLAGS} -c ${INC} ${SDIR}/$@ -o ${ODIR}/$(notdir $*).o


${MAIN}:
	${CC} ${CFLAGS} -L/ucrt64/lib -lraylib -lopengl32 -lwinmm -lgdi32 -c ${INC} ./examples/$@.cpp -o ${ODIR}/$(notdir $@).o


clean:
	-rm -rf ${ODIR}/*.o
	-rm ${TARGET}