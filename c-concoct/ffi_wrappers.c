/* Thin FFI wrappers for proptest oracle calls.
 *
 * Functions that take t_Cluster* are wrapped here to avoid mirroring
 * the full 25-field struct (which includes gsl_matrix pointers) in Rust.
 * Wrappers build the struct on the stack from flat arguments.
 */

#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>
#include <omp.h>
#include "c_vbgmm_fit.h"

/* Forward declaration — mstep is not in the header */
double mstep(int k, double *adMu, double* adM, int nN, int nD,
             double** aadZ, double** aadX,
             double *pdBeta, double *pdNu, double *pdLDet,
             t_VBParams *ptVBParams, gsl_matrix *ptCovarK, gsl_matrix *ptSigmaMatrix);

void ffi_mstep(int k, int nN, int nD, int nK,
               double **aadZ, double **aadX,
               double dBeta0, double dNu0, double **aadInvW0,
               /* outputs: */
               double *adMu, double *adM,
               double *pdPi, double *pdBeta, double *pdNu, double *pdLDet,
               double *covarOut, double *sigmaOut)
{
    int nD2 = nD;
    gsl_matrix *ptInvW0 = gsl_matrix_alloc(nD, nD);
    gsl_matrix *ptCovarK = gsl_matrix_alloc(nD, nD);
    gsl_matrix *ptSigmaMatrix = gsl_matrix_alloc(nD, nD);
    t_VBParams vbParams;
    int l, m2;

    for (l = 0; l < nD; l++)
        for (m2 = 0; m2 < nD; m2++)
            gsl_matrix_set(ptInvW0, l, m2, aadInvW0[l][m2]);

    vbParams.dBeta0 = dBeta0;
    vbParams.dNu0 = dNu0;
    vbParams.ptInvW0 = ptInvW0;
    vbParams.dLogWishartB = 0.0; /* not used by mstep */

    *pdPi = mstep(k, adMu, adM, nN, nD, aadZ, aadX,
                  pdBeta, pdNu, pdLDet, &vbParams, ptCovarK, ptSigmaMatrix);

    for (l = 0; l < nD; l++)
        for (m2 = 0; m2 < nD; m2++) {
            covarOut[l * nD + m2] = gsl_matrix_get(ptCovarK, l, m2);
            sigmaOut[l * nD + m2] = gsl_matrix_get(ptSigmaMatrix, l, m2);
        }

    gsl_matrix_free(ptInvW0);
    gsl_matrix_free(ptCovarK);
    gsl_matrix_free(ptSigmaMatrix);
}

void calcZ_MP(t_Cluster* ptCluster, t_Data *ptData);
void initKMeans(gsl_rng *ptGSLRNG, t_Cluster *ptCluster, t_Data *ptData);
void allocateCluster(t_Cluster *ptCluster, int nN, int nK, int nD, t_Data *ptData, long lSeed, int nMaxIter, double dEpsilon, char *szCOutFile);
void destroyCluster(t_Cluster* ptCluster);
void setVBParams(t_VBParams *ptVBParams, t_Data *ptData);

void gmmTrainVB_MP(t_Cluster *ptCluster, t_Data *ptData);

/* Run initKMeans + gmmTrainVB_MP and return final assignments */
void ffi_trainFull(double **aadX, int nN, int nK, int nD,
                   unsigned long seed, int maxIter, double dEpsilon,
                   int *anAssign_out)
{
    t_Data data;
    t_VBParams vbParams;
    t_Cluster *ptCluster;
    gsl_rng *ptGSLRNG;
    const gsl_rng_type *ptGSLRNGType;
    int i;

    data.nN = nN;
    data.nD = nD;
    data.aadX = aadX;

    vbParams.dBeta0 = 1.0e-3;
    vbParams.dNu0 = (double)nD;
    vbParams.ptInvW0 = gsl_matrix_alloc(nD, nD);

    {
        double adVar[nD], adMu[nD];
        calcSampleVar(&data, adVar, adMu);
        gsl_matrix_set_zero(vbParams.ptInvW0);
        for (i = 0; i < nD; i++)
            gsl_matrix_set(vbParams.ptInvW0, i, i, adVar[i] * (double)nD);
    }
    vbParams.dLogWishartB = dLogWishartB(vbParams.ptInvW0, nD, vbParams.dNu0, 1);

    ptCluster = malloc(sizeof(t_Cluster));
    allocateCluster(ptCluster, nN, nK, nD, &data, seed, maxIter, dEpsilon, NULL);
    ptCluster->ptVBParams = &vbParams;
    ptCluster->bAssign = 0;

    /* Pin OMP threads to match what driverMP does:
       nThreads = min(nThreads, nN/32 + 1) */
    {
        int nT = nN / 32 + 1;
        int nThreads = omp_get_max_threads();
        if (nT < nThreads) nThreads = nT;
        omp_set_num_threads(nThreads);
    }

    gsl_rng_env_setup();
    ptGSLRNGType = gsl_rng_default;
    ptGSLRNG = gsl_rng_alloc(ptGSLRNGType);
    gsl_rng_set(ptGSLRNG, seed);

    initKMeans(ptGSLRNG, ptCluster, &data);
    gmmTrainVB_MP(ptCluster, &data);

    for (i = 0; i < nN; i++)
        anAssign_out[i] = ptCluster->anMaxZ[i];

    gsl_rng_free(ptGSLRNG);
    destroyCluster(ptCluster);
    free(ptCluster);
    gsl_matrix_free(vbParams.ptInvW0);
}

/* Run initKMeans and return the resulting Z matrix and assignments */
void ffi_initKMeans(double **aadX, int nN, int nK, int nD,
                    unsigned long seed, int maxIter,
                    double dBeta0, double dNu0,
                    double **aadZ_out, int *anMaxZ_out,
                    double *adPi_out)
{
    t_Data data;
    t_VBParams vbParams;
    t_Cluster *ptCluster;
    gsl_rng *ptGSLRNG;
    const gsl_rng_type *ptGSLRNGType;
    int i, k;

    data.nN = nN;
    data.nD = nD;
    data.aadX = aadX;

    /* Set up VB params the same way the driver does */
    vbParams.dBeta0 = dBeta0;
    vbParams.dNu0 = dNu0;
    vbParams.ptInvW0 = gsl_matrix_alloc(nD, nD);

    {
        double adVar[nD], adMu[nD];
        calcSampleVar(&data, adVar, adMu);
        gsl_matrix_set_zero(vbParams.ptInvW0);
        for (i = 0; i < nD; i++) {
            gsl_matrix_set(vbParams.ptInvW0, i, i, adVar[i] * (double)nD);
        }
    }
    vbParams.dLogWishartB = dLogWishartB(vbParams.ptInvW0, nD, dNu0, 1);

    /* Allocate cluster */
    ptCluster = malloc(sizeof(t_Cluster));
    allocateCluster(ptCluster, nN, nK, nD, &data, seed, maxIter, 1.0e-4, NULL);
    ptCluster->ptVBParams = &vbParams;
    ptCluster->bAssign = 0;

    /* RNG */
    gsl_rng_env_setup();
    ptGSLRNGType = gsl_rng_default;
    ptGSLRNG = gsl_rng_alloc(ptGSLRNGType);
    gsl_rng_set(ptGSLRNG, seed);

    initKMeans(ptGSLRNG, ptCluster, &data);

    /* Copy results out */
    for (i = 0; i < nN; i++) {
        anMaxZ_out[i] = ptCluster->anMaxZ[i];
        for (k = 0; k < nK; k++) {
            aadZ_out[i][k] = ptCluster->aadZ[i][k];
        }
    }
    for (k = 0; k < nK; k++) {
        adPi_out[k] = ptCluster->adPi[k];
    }

    gsl_rng_free(ptGSLRNG);
    destroyCluster(ptCluster);
    free(ptCluster);
    gsl_matrix_free(vbParams.ptInvW0);
}
double calcVBL_MP(t_Cluster* ptCluster);

double ffi_calcVBL(double **aadX, int nN, int nK, int nD,
                   double **aadZ, double **aadMu, double **aadM,
                   gsl_matrix **aptCovar, gsl_matrix **aptSigma,
                   double *adPi, double *adBeta, double *adNu, double *adLDet,
                   double dBeta0, double dNu0, gsl_matrix *ptInvW0, double dLogWishartB)
{
    t_Data data;
    t_Cluster cluster;
    t_VBParams vbParams;

    data.nN = nN;
    data.nD = nD;
    data.aadX = aadX;

    vbParams.dBeta0 = dBeta0;
    vbParams.dNu0 = dNu0;
    vbParams.ptInvW0 = ptInvW0;
    vbParams.dLogWishartB = dLogWishartB;

    memset(&cluster, 0, sizeof(cluster));
    cluster.nN = nN;
    cluster.nK = nK;
    cluster.nD = nD;
    cluster.ptData = &data;
    cluster.ptVBParams = &vbParams;
    cluster.aadZ = aadZ;
    cluster.aadMu = aadMu;
    cluster.aadM = aadM;
    cluster.aptCovar = aptCovar;
    cluster.aptSigma = aptSigma;
    cluster.adPi = adPi;
    cluster.adBeta = adBeta;
    cluster.adNu = adNu;
    cluster.adLDet = adLDet;

    return calcVBL_MP(&cluster);
}

void ffi_calcZ(double **aadX, int nN, int nK, int nD,
               double **aadZ, double **aadM,
               gsl_matrix **aptSigma,
               double *adPi, double *adNu, double *adLDet, double *adBeta)
{
    t_Data data;
    t_Cluster cluster;

    data.nN = nN;
    data.nD = nD;
    data.aadX = aadX;

    memset(&cluster, 0, sizeof(cluster));
    cluster.nN = nN;
    cluster.nK = nK;
    cluster.nD = nD;
    cluster.aadZ = aadZ;
    cluster.aadM = aadM;
    cluster.aptSigma = aptSigma;
    cluster.adPi = adPi;
    cluster.adNu = adNu;
    cluster.adLDet = adLDet;
    cluster.adBeta = adBeta;

    calcZ_MP(&cluster, &data);
}

void ffi_updateMeans(double **aadX, int nN, int nK, int nD,
                     int *anMaxZ, int *anW, double **aadMu)
{
    t_Data data;
    t_Cluster cluster;

    data.nN = nN;
    data.nD = nD;
    data.aadX = aadX;

    /* Zero the struct, then set only the fields updateMeans reads */
    memset(&cluster, 0, sizeof(cluster));
    cluster.nN = nN;
    cluster.nK = nK;
    cluster.nD = nD;
    cluster.anMaxZ = anMaxZ;
    cluster.anW = anW;
    cluster.aadMu = aadMu;

    updateMeans(&cluster, &data);
}
