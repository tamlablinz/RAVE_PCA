/* closest -- find the stored vector with closest position using k-d tree */

#include "m_pd.h"
#include <math.h>
#include <stdlib.h>

static t_class *closest_class;
#define DIMENSION 10
#define MAX_POINTS 1000000  // k-d tree can handle 100k+ points efficiently

typedef struct _point {
    t_float coords[DIMENSION];
    int index;
    t_float age;
} t_point;

typedef struct _kdnode {
    t_point *point;
    struct _kdnode *left;
    struct _kdnode *right;
    int depth;
} t_kdnode;

typedef struct _closest
{
    t_object x_obj;
    t_point *x_points;
    int x_n;
    int x_nonrepeat;
    t_kdnode *x_root;
    int x_needs_rebuild;
} t_closest;

// Helper function to calculate Euclidean distance
static t_float distance(t_float *a, t_float *b) {
    t_float sum = 0;
    for (int i = 0; i < DIMENSION; i++) {
        t_float diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

// Find median of points along a specific dimension
static t_point* find_median(t_point *points, int start, int end, int dim) {
    int n = end - start + 1;
    int median_idx = start + n / 2;
    
    // Simple selection sort to find median (could be optimized)
    for (int i = start; i <= end; i++) {
        for (int j = i + 1; j <= end; j++) {
            if (points[i].coords[dim] > points[j].coords[dim]) {
                t_point temp = points[i];
                points[i] = points[j];
                points[j] = temp;
            }
        }
    }
    
    return &points[median_idx];
}

// Build k-d tree recursively
static t_kdnode* build_kdtree(t_point *points, int start, int end, int depth) {
    if (start > end) return NULL;
    
    t_kdnode *node = (t_kdnode*)getbytes(sizeof(t_kdnode));
    int dim = depth % DIMENSION;
    
    if (start == end) {
        node->point = &points[start];
        node->left = node->right = NULL;
        node->depth = depth;
        return node;
    }
    
    // Find median and partition
    t_point *median = find_median(points, start, end, dim);
    int median_idx = median - points;
    
    // Partition around median
    t_point temp = points[median_idx];
    points[median_idx] = points[end];
    points[end] = temp;
    
    int pivot = start;
    for (int i = start; i < end; i++) {
        if (points[i].coords[dim] <= points[end].coords[dim]) {
            temp = points[i];
            points[i] = points[pivot];
            points[pivot] = temp;
            pivot++;
        }
    }
    
    temp = points[pivot];
    points[pivot] = points[end];
    points[end] = temp;
    
    node->point = &points[pivot];
    node->depth = depth;
    node->left = build_kdtree(points, start, pivot - 1, depth + 1);
    node->right = build_kdtree(points, pivot + 1, end, depth + 1);
    
    return node;
}

// Free k-d tree
static void free_kdtree(t_kdnode *node) {
    if (node) {
        free_kdtree(node->left);
        free_kdtree(node->right);
        freebytes(node, sizeof(t_kdnode));
    }
}

// Nearest neighbor search in k-d tree
static void nearest_neighbor_search(t_kdnode *node, t_float *query, 
                                   t_point **best, t_float *best_dist, int nonrepeat) {
    if (!node) return;
    
    t_float dist = distance(query, node->point->coords);
    if (nonrepeat) dist /= log(node->point->age);
    
    if (dist < *best_dist) {
        *best_dist = dist;
        *best = node->point;
    }
    
    int dim = node->depth % DIMENSION;
    t_float diff = query[dim] - node->point->coords[dim];
    
    // Search the closer subtree first
    t_kdnode *first = (diff <= 0) ? node->left : node->right;
    t_kdnode *second = (diff <= 0) ? node->right : node->left;
    
    nearest_neighbor_search(first, query, best, best_dist, nonrepeat);
    
    // Check if we need to search the other subtree
    if (fabs(diff) < *best_dist) {
        nearest_neighbor_search(second, query, best, best_dist, nonrepeat);
    }
}

static void *closest_new(t_float fnonrepeat)
{
    t_closest *x = (t_closest *)pd_new(closest_class);
    outlet_new(&x->x_obj, gensym("float"));
    x->x_points = (t_point *)getbytes(MAX_POINTS * sizeof(t_point));
    x->x_n = 0;
    x->x_nonrepeat = (fnonrepeat != 0);
    x->x_root = NULL;
    x->x_needs_rebuild = 0;
    return (x);
}

static void closest_clear(t_closest *x)
{
    if (x->x_root) {
        free_kdtree(x->x_root);
        x->x_root = NULL;
    }
    x->x_n = 0;
    x->x_needs_rebuild = 0;
}

static void closest_print(t_closest *x)
{
    for (int j = 0; j < x->x_n; j++) {
        t_point *p = &x->x_points[j];
        post("%2d age %2d w %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f",
            p->index, (int)(p->age), p->coords[0], p->coords[1], p->coords[2], p->coords[3], 
            p->coords[4], p->coords[5], p->coords[6], p->coords[7], p->coords[8], p->coords[9]);
    }
}

static void closest_add(t_closest *x, t_symbol *s, int argc, t_atom *argv)
{
    if (x->x_n >= MAX_POINTS) {
        pd_error(x, "closest: maximum number of points reached");
        return;
    }
    
    t_point *p = &x->x_points[x->x_n];
    p->index = x->x_n;
    p->age = 2;
    
    for (int i = 0; i < DIMENSION; i++) {
        p->coords[i] = atom_getfloatarg(i, argc, argv);
    }
    
    x->x_n++;
    x->x_needs_rebuild = 1;  // Mark for rebuild
    (void)s;
}

static void closest_list(t_closest *x, t_symbol *s, int argc, t_atom *argv)
{
    if (x->x_n == 0) {
        outlet_float(x->x_obj.ob_outlet, -1);
        return;
    }
    
    // Rebuild k-d tree if needed
    if (x->x_needs_rebuild) {
        if (x->x_root) free_kdtree(x->x_root);
        x->x_root = build_kdtree(x->x_points, 0, x->x_n - 1, 0);
        x->x_needs_rebuild = 0;
    }
    
    t_float query[DIMENSION];
    for (int i = 0; i < DIMENSION; i++) {
        query[i] = atom_getfloatarg(i, argc, argv);
    }
    
    t_point *best = NULL;
    t_float best_dist = 1e20;
    
    nearest_neighbor_search(x->x_root, query, &best, &best_dist, x->x_nonrepeat);
    
    if (best) {
        // Update ages
        for (int j = 0; j < x->x_n; j++) {
            x->x_points[j].age += 1.;
        }
        best->age = 1;
        outlet_float(x->x_obj.ob_outlet, (t_float)best->index);
    } else {
        outlet_float(x->x_obj.ob_outlet, -1);
    }
    (void)s;
}

static void closest_free(t_closest *x)
{
    if (x->x_root) free_kdtree(x->x_root);
    freebytes(x->x_points, MAX_POINTS * sizeof(t_point));
}

void closest_setup(void)
{
    closest_class = class_new(gensym("closest"), (t_newmethod)closest_new,
        (t_method)closest_free, sizeof(t_closest), 0, A_DEFFLOAT, 0);
    class_addmethod(closest_class, (t_method)closest_add, gensym("add"), A_GIMME, 0);
    class_addmethod(closest_class, (t_method)closest_clear, gensym("clear"), 0);
    class_addmethod(closest_class, (t_method)closest_print, gensym("print"), 0);
    class_addlist(closest_class, closest_list);
} 