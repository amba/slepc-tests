default: 1d-josephson 1d-josephson-spin

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common

ex1: ex1.o
	-${CLINKER} -o ex1 ex1.o ${SLEPC_EPS_LIB}
	${RM} ex1.o
ex1f: ex1f.o
	-${FLINKER} -o ex1f ex1f.o ${SLEPC_EPS_LIB}
	${RM} ex1f.o
