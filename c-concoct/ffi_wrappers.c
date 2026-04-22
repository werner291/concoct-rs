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
