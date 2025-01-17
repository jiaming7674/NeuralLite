CC=/usr/bin/g++
SDIR=./neural/src
ODIR=./bin
INC=-I./neural/inc -I./loader/inc -I/ucrt64/include/eigen3 -I/ucrt64/include
TARGET=run
CFLAGS= -O4
SRCS=network.cpp core.cpp layers/activation_layer.cpp layers/fc_layer.cpp layers/conv_layer.cpp
_OBJS=$(patsubst %.cpp, ${ODIR}/%.o, $(notdir ${SRCS}))
LIB=-lpthread -lraylib -lopengl32 -lwinmm -lgdi32

MAIN=mnist_conv

.PHONY: clean

${TARGET}: ${SRCS} ${MAIN} mnist
	${CC} ${CFLAGS} -o ${TARGET} ${_OBJS} ./bin/mnist.o ./bin/${MAIN}.o -L/ucrt64/lib ${LIB}
	@echo "Create TARGET done !!"


%.cpp:
	${CC} ${CFLAGS} -c ${INC} ${SDIR}/$@ -o ${ODIR}/$(notdir $*).o

mnist:
	${CC} ${CFLAGS} -c ${INC} -I./loader/inc ./loader/src/mnist.cpp -o ${ODIR}/$(notdir $@).o

${MAIN}:
	${CC} ${CFLAGS} -L/ucrt64/lib -lraylib -lopengl32 -lwinmm -lgdi32 -c ${INC} ./examples/$@.cpp -o ${ODIR}/$(notdir $@).o


clean:
	-rm -rf ${ODIR}/*.o
	-rm ${TARGET}