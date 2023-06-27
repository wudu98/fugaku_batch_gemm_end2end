#include <cstdio>
#include <omp.h>
#include <cblas.h>

// void my_blas_batch_sgemm(const int parallel_mode, const int batch_count, const int *batch_size, const int *batch_head, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, const int* m, const int* n, const int* k, const float* alpha, const float ** a, const int* lda, const float ** b, const int* ldb, const float* beta, float ** c, const int* ldc)
// {
//     // const int num_threads = omp_get_max_threads();

// 	#pragma omp parallel for collapse(2) if (parallel_mode==1)
//     for(int i = 0; i < batch_count; i++){
// 		for(int j = 0; j < batch_size[i]; j++){
// 			cblas_sgemm(layout, transa, transb, m[i], n[i], k[i], alpha[i], a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], beta[i], c[batch_head[i]+j], ldc[i]);
// 		}
// 	}
// }

void sgemm_core(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA,
                const CBLAS_TRANSPOSE TransB, const int M, const int N,
                const int K, const float alpha, const float  *A,
                const int lda, const float  *B, const int ldb,
				const float beta, float  *C, const int ldc, const char* name) {
	std::printf("SGEMM(%d, %d, %d) is called [%s]\n", M, N, K, name);
}

extern "C" void sgemm_(char *transa, char *transb, int *m, int *n, int *k, 
						float *alpha, const float *a, int *lda, const float *b, 
						int *ldb, float *beta, float *c, int *ldc) {
	sgemm_core(
		CblasColMajor,
		*transa == 'N' ? CblasNoTrans : CblasTrans,
		*transb == 'N' ? CblasNoTrans : CblasTrans,
		*m, *n, *k,
		*alpha,
		a, *lda,
		b, *ldb,
		*beta,
		c, *ldc,
		__func__
	);
}

namespace {
	int use_batch() {
	const int num_threads = omp_get_max_threads();
	printf("threads : %d\n", omp_get_max_threads);

	if(num_threads > 1)
		return 1;
	else
		return 0;
	}
}
