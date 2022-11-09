#include <iostream>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverSp.h>

int main() {
    cusparseHandle_t cusparseHandle;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if (cusparseStatus != 0)
        std::cout << "status create cusparse handle: " << cusparseStatus << std::endl;

    cusolverSpHandle_t cusolverHandle;
    cusolverStatus_t cusolverStatus;
    cusolverStatus = cusolverSpCreate(&cusolverHandle);
    if (cusolverStatus != 0)
        std::cout << "status create cusolver handle: " << cusolverStatus << std::endl;

    int nNonZero = 3;
    int n = 2, m = 2;
    float* valA = new float[nNonZero];
    int* rowIndA = new int[nNonZero];
    int* colIndA = new int[nNonZero];
    valA[0] = valA[1] = valA[2] = 1.0f;
    rowIndA[0] = rowIndA[1] = 0;
    rowIndA[2] = 1;
    colIndA[0] = 0;
    colIndA[1] = colIndA[2] = 1;


    float* csrValAGpu;
    cudaMalloc(&csrValAGpu, sizeof(float) * nNonZero);
    cudaMemcpy(csrValAGpu, valA, sizeof(float) * nNonZero, cudaMemcpyHostToDevice);
    int* csrRowIndAGpu;
    cudaMalloc(&csrRowIndAGpu, sizeof(int) * nNonZero);
    cudaMemcpy(csrRowIndAGpu, rowIndA, sizeof(int) * nNonZero, cudaMemcpyHostToDevice);
    int *csrColIndAGpu;
    cudaMalloc(&csrColIndAGpu, sizeof(int) * nNonZero);
    cudaMemcpy(csrColIndAGpu, colIndA, sizeof(int) * nNonZero, cudaMemcpyHostToDevice);

    int* csrRowPtrAGpu;
    cudaMalloc(&csrRowPtrAGpu, sizeof(int) * (m + 1));
    cusparseStatus = cusparseXcoo2csr(cusparseHandle, csrRowIndAGpu, nNonZero, m, csrRowPtrAGpu, CUSPARSE_INDEX_BASE_ZERO);
     if (cusparseStatus != 0)
        std::cout << "coo2csr error" << std::endl;

    float* csrValA = new float[nNonZero];
    cudaMemcpy(csrValA, csrValAGpu, sizeof(float) * nNonZero, cudaMemcpyDeviceToHost);
    int* csrColIndA = new int[nNonZero];
    cudaMemcpy(csrColIndA, csrColIndAGpu, sizeof(int) * nNonZero, cudaMemcpyDeviceToHost);
    int* csrRowPtrA = new int[n + 1];
    cudaMemcpy(csrRowPtrA, csrRowPtrAGpu, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nNonZero; i++)
        std::cout << csrValA[i] << ' ';
    std::cout << std::endl;
    for (int i = 0; i < m + 1; i++)
        std::cout << csrRowPtrA[i] << ' ';
    std::cout << std::endl;
    for (int i = 0; i < nNonZero; i++)
        std::cout << csrColIndA[i] << ' ' ;
    std::cout << std::endl;

    float* cscValAGpu;
    cudaMalloc(&cscValAGpu, sizeof(float) * nNonZero);
    int* cscRowIndAGpu;
    cudaMalloc(&cscRowIndAGpu, sizeof(int) * nNonZero);
    int* cscColPtrAGpu;
    cudaMalloc(&cscColPtrAGpu, sizeof(int) * (n + 1));
    size_t bufferSize;
    cusparseStatus = cusparseCsr2cscEx2_bufferSize(cusparseHandle, m, n, nNonZero, csrValAGpu, csrRowPtrAGpu, csrColIndAGpu, cscValAGpu, cscColPtrAGpu, cscRowIndAGpu, CUDA_R_32F, CUSPARSE_ACTION_SYMBOLIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);
    if (cusparseStatus != 0)
        std::cout << "buffersize error" << std::endl;

    int* bufferGpu;
    cudaMalloc(&bufferGpu, sizeof(int) * bufferSize);
    cusparseStatus = cusparseCsr2cscEx2(cusparseHandle, m, n, nNonZero, csrValAGpu, csrRowPtrAGpu, csrColIndAGpu, cscValAGpu, cscColPtrAGpu, cscRowIndAGpu, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, bufferGpu);
    if (cusparseStatus != 0)
        std::cout << "csr2csc error" << std::endl;

    float* cscValA = new float[nNonZero];
    cudaMemcpy(cscValA, cscValAGpu, sizeof(float) * nNonZero, cudaMemcpyDeviceToHost);
    int* cscRowIndA = new int[nNonZero];
    cudaMemcpy(cscRowIndA, cscRowIndAGpu, sizeof(int) * nNonZero, cudaMemcpyDeviceToHost);
    int* cscColPtrA = new int[n + 1];
    cudaMemcpy(cscColPtrA, cscColPtrAGpu, sizeof(int) * (n + 1), cudaMemcpyDeviceToHost);

    for (int i = 0; i < nNonZero; i++)
        std::cout << cscValA[i] << ' ';
    std::cout << std::endl;
    for (int i = 0; i < n + 1; i++)
        std::cout << cscColPtrA[i] << ' ';
    std::cout << std::endl;
    for (int i = 0; i < nNonZero; i++)
        std::cout << cscRowIndA[i] << ' ' ;
    std::cout << std::endl;

    float* b = new float[n];
    b[0] = 1.5f;
    b[1] = 0.7f;
    float* bGpu;
    cudaMalloc(&bGpu, sizeof(float) * n);
    cudaMemcpy(bGpu, b, sizeof(float) * n, cudaMemcpyHostToDevice);
    float* xGpu;
    cudaMalloc(&xGpu, sizeof(float) * n);
    cusparseMatDescr_t descr;
    cusparseCreateMatDescr(&descr);
    int singularity;
    cusolverSpScsrlsvqr(cusolverHandle, n, nNonZero, descr, csrValAGpu, csrRowPtrAGpu, csrColIndAGpu, bGpu, 1e-5f, 0, xGpu, &singularity);
    cudaDeviceSynchronize();
    if (singularity != -1)
        std::cout << "Not invertible!" << std::endl;

    float* x = new float[n];
    cudaMemcpy(x, xGpu, sizeof(float) * n, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++)
        std::cout << x[i] << ' ';
    std::cout << std::endl;

//     // --- prepare matrix:
//     int Nrows = 4;
//     int Ncols = 4;
//     std::vector<float> csrVal;
//     std::vector<int> cooRow;
//     std::vector<int> csrColInd;
//     std::vector<float> b;

//     assemble_poisson_matrix_coo(csrVal, cooRow, csrColInd, b, Nrows, Ncols);

//     int nnz = csrVal.size();
//     int m = Nrows * Ncols;
//     std::vector<int> csrRowPtr(m+1);

//     // --- prepare solving and copy to GPU:
//     std::vector<float> x(m);
//     float tol = 1e-5;
//     int reorder = 0;
//     int singularity = 0;

//     float *db, *dcsrVal, *dx;
//     int *dcsrColInd, *dcsrRowPtr, *dcooRow;
//     cudaMalloc((void**)&db, m*sizeof(float));
//     cudaMalloc((void**)&dx, m*sizeof(float));
//     cudaMalloc((void**)&dcsrVal, nnz*sizeof(float));
//     cudaMalloc((void**)&dcsrColInd, nnz*sizeof(int));
//     cudaMalloc((void**)&dcsrRowPtr, (m+1)*sizeof(int));
//     cudaMalloc((void**)&dcooRow, nnz*sizeof(int));

//     cudaMemcpy(db, b.data(), b.size()*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dcsrVal, csrVal.data(), csrVal.size()*sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(dcsrColInd, csrColInd.data(), csrColInd.size()*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(dcooRow, cooRow.data(), cooRow.size()*sizeof(int), cudaMemcpyHostToDevice);

//     cusparse_status = cusparseXcoo2csr(cusparse_handle, dcooRow, nnz, m,
//                                        dcsrRowPtr, CUSPARSE_INDEX_BASE_ZERO);
//     std::cout << "status cusparse coo2csr conversion: " << cusparse_status << std::endl;

//     cudaDeviceSynchronize(); // matrix format conversion has to be finished!

//     // --- everything ready for computation:

//     cusparseMatDescr_t descrA;

//     cusparse_status = cusparseCreateMatDescr(&descrA);
//     std::cout << "status cusparse createMatDescr: " << cusparse_status << std::endl;

//     // optional: print dense matrix that has been allocated on GPU

//     std::vector<float> A(m*m, 0);
//     float *dA;
//     cudaMalloc((void**)&dA, A.size()*sizeof(float));

//     cusparseScsr2dense(cusparse_handle, m, m, descrA, dcsrVal,
//                        dcsrRowPtr, dcsrColInd, dA, m);

//     cudaMemcpy(A.data(), dA, A.size()*sizeof(float), cudaMemcpyDeviceToHost);
//     std::cout << "A: \n";
//     for (int i = 0; i < m; ++i) {
//         for (int j = 0; j < m; ++j) {
//             std::cout << A[i*m + j] << " ";
//         }
//         std::cout << std::endl;
//     }

//     cudaFree(dA);

//     std::cout << "b: \n";
//     cudaMemcpy(b.data(), db, (m)*sizeof(int), cudaMemcpyDeviceToHost);
//     for (auto a : b) {
//         std::cout << a << ",";
//     }
//     std::cout << std::endl;


//     // --- solving!!!!

// //    cusolver_status = cusolverSpScsrlsvchol(cusolver_handle, m, nnz, descrA, dcsrVal,
// //                       dcsrRowPtr, dcsrColInd, db, tol, reorder, dx,
// //                       &singularity);

//      cusolver_status = cusolverSpScsrlsvqr(cusolver_handle, m, nnz, descrA, dcsrVal,
//                         dcsrRowPtr, dcsrColInd, db, tol, reorder, dx,
//                         &singularity);

//     cudaDeviceSynchronize();

//     std::cout << "singularity (should be -1): " << singularity << std::endl;

//     std::cout << "status cusolver solving (!): " << cusolver_status << std::endl;

//     cudaMemcpy(x.data(), dx, m*sizeof(float), cudaMemcpyDeviceToHost);

//     // relocated these 2 lines from above to solve (2):
//     cusparse_status = cusparseDestroy(cusparse_handle);
//     std::cout << "status destroy cusparse handle: " << cusparse_status << std::endl;

//     cusolver_status = cusolverSpDestroy(cusolver_handle);
//     std::cout << "status destroy cusolver handle: " << cusolver_status << std::endl;

//     for (auto a : x) {
//         std::cout << a << " ";
//     }
//     std::cout << std::endl;



//     cudaFree(db);
//     cudaFree(dx);
//     cudaFree(dcsrVal);
//     cudaFree(dcsrColInd);
//     cudaFree(dcsrRowPtr);
//     cudaFree(dcooRow);

    return 0;
}