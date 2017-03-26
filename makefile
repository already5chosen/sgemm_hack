CC   = g++
CPPOPT     = -std=c++11 -Wall -mavx -O2
CPPOPT_AVX = -std=c++11 -Wall -mavx -O1
CPPOPT_FMA = -std=c++11 -Wall -mavx -O1 -mfma

avx%.o : avx%.cpp
	$(CC) -c $(CPPOPT_AVX) $< -o $@
	${CC} -MM ${CPPOPT_AVX} -MF $(patsubst %.o,%.d,$@) $<

fma%.o : fma%.cpp
	$(CC) -c $(CPPOPT_FMA) $< -o $@
	${CC} -MM ${CPPOPT_FMA} -MF $(patsubst %.o,%.d,$@) $<

%.o : %.cpp
	$(CC) -c $(CPPOPT) $< -o $@
	${CC} -MM ${CPPOPT} -MF $(patsubst %.o,%.d,$@) $<

ALL_EXE :=
ALL_EXE += tst1.exe
all: ${ALL_EXE}


OBJLIST :=
OBJLIST += tst1.o
OBJLIST += ref_noncblas_sgemm.o

OBJLIST_AVX :=
OBJLIST_AVX += avxscalar_noncblas_sgemm.o
OBJLIST_AVX += avx128_noncblas_sgemm.o
OBJLIST_AVX += avx256_noncblas_sgemm.o
OBJLIST_AVX += avx128_noncblas_sgemm_m.o
OBJLIST_AVX += avx256_noncblas_sgemm_m.o

OBJLIST_FMA :=
OBJLIST_FMA += fma128_noncblas_sgemm.o
OBJLIST_FMA += fma256_noncblas_sgemm.o
OBJLIST_FMA += fma256_noncblas_sgemm_4x3.o
OBJLIST_FMA += fma256_noncblas_sgemm_3x4.o
OBJLIST_FMA += fma256_noncblas_sgemm_4x2.o
OBJLIST_FMA += fma256_noncblas_sgemm_5x2.o
OBJLIST_FMA += fma128_noncblas_sgemm_m.o
OBJLIST_FMA += fma256_noncblas_sgemm_m.o


-include $(OBJLIST:.o=.d)
-include $(OBJLIST_AVX:.o=.d)
-include $(OBJLIST_FMA:.o=.d)

tst1.exe : $(OBJLIST) $(OBJLIST_AVX) $(OBJLIST_FMA)
	${CC} $+ -o$@


clean:
	rm *.o *.d $(ALL_EXE)