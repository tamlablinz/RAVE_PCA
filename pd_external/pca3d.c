    /* PCA-based 3D reduction and inverse mapping */
#include "m_pd.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* if you have LAPACK/BLAS, include headers here. Otherwise, use a simple SVD/eigen implementation.
 * For now, we'll use a simple Jacobi eigenvalue algorithm for small matrices (for covariance). */

#define MAX_DIM 128
#define MAX_POINTS 1000000

static t_class *pca3d_class;

typedef struct _pca3d {
    t_object x_obj;
    int dimensions;
    t_symbol *array_name;
    int num_points;
    double mean[MAX_DIM];
    double components[3][MAX_DIM];  /* 3 principal components */
    int pca_ready;
    void *outlet_list;  /* left outlet */
    void *outlet_status;  /* right outlet */
    t_symbol *output_base;  /* output base name */
} t_pca3d;

/* ------------------------------ helper functions ------------------------------ */

static int get_pd_array(t_symbol *name, t_word **buf, int *size) {
    t_garray *a = (t_garray *)pd_findbyclass(name, garray_class);
    if (!a) return 0;
    if (!garray_getfloatwords(a, size, buf)) return 0;
    return 1;
}

static void compute_mean(double *data, int num_points, int dim, double *mean) {
    for (int d = 0; d < dim; ++d) mean[d] = 0;
    for (int i = 0; i < num_points; ++i)
        for (int d = 0; d < dim; ++d)
            mean[d] += data[i*dim + d];
    for (int d = 0; d < dim; ++d) mean[d] /= num_points;
}

static void center_data(double *data, int num_points, int dim, double *mean) {
    for (int i = 0; i < num_points; ++i)
        for (int d = 0; d < dim; ++d)
            data[i*dim + d] -= mean[d];
}

    /* compute covariance matrix (dim x dim) */
static void compute_covariance(double *data, int num_points, int dim, double *cov) {
    for (int i = 0; i < dim*dim; ++i) cov[i] = 0;
    for (int i = 0; i < num_points; ++i) {
        for (int d1 = 0; d1 < dim; ++d1) {
            for (int d2 = 0; d2 < dim; ++d2) {
                cov[d1*dim + d2] += data[i*dim + d1] * data[i*dim + d2];
            }
        }
    }
    for (int i = 0; i < dim*dim; ++i) cov[i] /= (num_points - 1);
}

    /* Jacobi eigenvalue algorithm for symmetric matrices (for small dim)
     * Only gets top 3 eigenvectors (principal components) */
static void jacobi_eigen(double *cov, int dim, double components[3][MAX_DIM]) {
    /* for simplicity, use a naive power iteration for top 3 components */
    double v[MAX_DIM];
    for (int k = 0; k < 3; ++k) {
            /* start with random vector */
        for (int i = 0; i < dim; ++i) v[i] = (i==k)?1.0:0.0;
        for (int iter = 0; iter < 30; ++iter) {
                /* multiply by cov */
            double tmp[MAX_DIM] = {0};
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    tmp[i] += cov[i*dim + j] * v[j];
                }
            }
                /* orthogonalize against previous */
            for (int prev = 0; prev < k; ++prev) {
                double dot = 0;
                for (int i = 0; i < dim; ++i) dot += tmp[i] * components[prev][i];
                for (int i = 0; i < dim; ++i) tmp[i] -= dot * components[prev][i];
            }
                /* normalize */
            double norm = 0;
            for (int i = 0; i < dim; ++i) norm += tmp[i]*tmp[i];
            norm = sqrt(norm);
            if (norm < 1e-12) break;
            for (int i = 0; i < dim; ++i) v[i] = tmp[i]/norm;
        }
            /* store */
        for (int i = 0; i < dim; ++i) components[k][i] = v[i];
    }
}

    /* resize Pd array to a given size */
static int resize_pd_array(t_symbol *name, int newsize) {
    t_garray *a = (t_garray *)pd_findbyclass(name, garray_class);
    if (!a) return 0;
    garray_resize_long(a, newsize);
    garray_redraw(a);
    return 1;
}

    /* write projected 3D coordinates to arrays */
static void write_projected_arrays(t_pca3d *x, double *data) {
    char xname[256], yname[256], zname[256];
    snprintf(xname, sizeof(xname), "%s-x", x->output_base->s_name);
    snprintf(yname, sizeof(yname), "%s-y", x->output_base->s_name);
    snprintf(zname, sizeof(zname), "%s-z", x->output_base->s_name);
    t_symbol *sx = gensym(xname);
    t_symbol *sy = gensym(yname);
    t_symbol *sz = gensym(zname);
    int npts = x->num_points;
    resize_pd_array(sx, npts);
    resize_pd_array(sy, npts);
    resize_pd_array(sz, npts);
    t_word *xbuf, *ybuf, *zbuf;
    int nptsx, nptsy, nptsz;
    int ok = 1;
    if (!get_pd_array(sx, &xbuf, &nptsx) || nptsx < npts) { pd_error(x, "pca3d: output array %s too small or missing", xname); ok = 0; }
    if (!get_pd_array(sy, &ybuf, &nptsy) || nptsy < npts) { pd_error(x, "pca3d: output array %s too small or missing", yname); ok = 0; }
    if (!get_pd_array(sz, &zbuf, &nptsz) || nptsz < npts) { pd_error(x, "pca3d: output array %s too small or missing", zname); ok = 0; }
    if (!ok) return;
    for (int i = 0; i < npts; ++i) {
        double px = 0, py = 0, pz = 0;
        for (int d = 0; d < x->dimensions; ++d) {
            double centered = data[i*x->dimensions + d];
            px += centered * x->components[0][d];
            py += centered * x->components[1][d];
            pz += centered * x->components[2][d];
        }
        xbuf[i].w_float = px;
        ybuf[i].w_float = py;
        zbuf[i].w_float = pz;
    }
    garray_redraw((t_garray *)pd_findbyclass(sx, garray_class));
    garray_redraw((t_garray *)pd_findbyclass(sy, garray_class));
    garray_redraw((t_garray *)pd_findbyclass(sz, garray_class));
}

/* ------------------------------ PCA computation on bang ------------------------------ */

static void pca3d_bang(t_pca3d *x) {
    t_word *buf;
    int size;
    if (!get_pd_array(x->array_name, &buf, &size)) {
        pd_error(x, "pca3d: could not find array %s", x->array_name->s_name);
        return;
    }
    if (x->dimensions < 1 || x->dimensions > MAX_DIM) {
        pd_error(x, "pca3d: invalid dimensions");
        return;
    }
    if (size % x->dimensions != 0) {
        pd_error(x, "pca3d: array size not divisible by dimensions");
        return;
    }
    x->num_points = size / x->dimensions;
    if (x->num_points < 3) {
        pd_error(x, "pca3d: need at least 3 data points");
        return;
    }
    if (x->num_points > MAX_POINTS) {
        pd_error(x, "pca3d: too many data points");
        return;
    }
        /* output size message */
    t_atom size_atom;
    SETFLOAT(&size_atom, x->num_points);
    outlet_anything(x->outlet_status, gensym("size"), 1, &size_atom);
        /* copy data and compute mean, center data, covariance, eigenvectors */
    double *data = (double *)malloc(sizeof(double)*size);
    for (int i = 0; i < size; ++i) data[i] = buf[i].w_float;
    compute_mean(data, x->num_points, x->dimensions, x->mean);
    center_data(data, x->num_points, x->dimensions, x->mean);
    double *cov = (double *)malloc(sizeof(double)*x->dimensions*x->dimensions);
    compute_covariance(data, x->num_points, x->dimensions, cov);
    jacobi_eigen(cov, x->dimensions, x->components);
    write_projected_arrays(x, data);
    free(data), free(cov);
    x->pca_ready = 1;
    outlet_anything(x->outlet_status, gensym("done"), 0, NULL);
}

/* ------------------------------ inverse mapping: 3D list to n-dim list ------------------------------ */

static void pca3d_list(t_pca3d *x, t_symbol *s, int argc, t_atom *argv) {
    if (!x->pca_ready) return;
    if (argc != 3) return;
    double input3[3];
    for (int i = 0; i < 3; ++i) input3[i] = atom_getfloat(argv+i);
    t_atom out[MAX_DIM];
    for (int d = 0; d < x->dimensions; ++d) {
        double val = x->mean[d];
        for (int k = 0; k < 3; ++k) val += x->components[k][d] * input3[k];
        SETFLOAT(&out[d], val);
    }
    outlet_list(x->outlet_list, &s_list, x->dimensions, out);
    (void)s;
}

static void *pca3d_new(t_floatarg f_dim, t_symbol *in, t_symbol *outbase) {
    t_pca3d *x = (t_pca3d *)pd_new(pca3d_class);
    x->dimensions = (int)f_dim;
    x->array_name = in;
    x->output_base = outbase;
    x->pca_ready = 0;
    x->outlet_list = outlet_new(&x->x_obj, &s_list);
    x->outlet_status = outlet_new(&x->x_obj, &s_anything);
    return (x);
}

void pca3d_setup(void) {
    pca3d_class = class_new(gensym("pca3d"),
        (t_newmethod)pca3d_new,
        0, sizeof(t_pca3d),
        CLASS_DEFAULT,
        A_DEFFLOAT, A_SYMBOL, A_SYMBOL, 0);
    class_addbang(pca3d_class, (t_method)pca3d_bang);
    class_addlist(pca3d_class, (t_method)pca3d_list);
} 