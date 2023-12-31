#include <cstdio>
#include <omp.h>
#include <cblas.h>

// void sgemm_core(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
//                 const CBLAS_TRANSPOSE TransB, const int M, const int N,
//                 const int K, const float alpha, const float  *A,
//                 const int lda, const float  *B, const int ldb,
// 				const float beta, float  *C, const int ldc, const char* name) {
// 	std::printf("SGEMM(%d, %d, %d) is called [%s]\n", M, N, K, name);
// }

// extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, 
// 						float *alpha, const float *a, int *lda, const float *b, 
// 						int *ldb, float *beta, float *c, int *ldc) {
// 	sgemm_core(
// 		CblasColMajor,
// 		*transa == 'N' ? CblasNoTrans : CblasTrans,
// 		*transb == 'N' ? CblasNoTrans : CblasTrans,
// 		*m, *n, *k,
// 		*alpha,
// 		a, *lda,
// 		b, *ldb,
// 		*beta,
// 		c, *ldc,
// 		__func__
// 	);
// }

extern "C" int use_batch() {
	const int num_threads = omp_get_max_threads();
	printf("threads : %d\n", num_threads);

	return 0;
}
