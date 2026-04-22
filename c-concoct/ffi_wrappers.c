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
