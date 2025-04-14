#ifndef _FUNCTIONS_H_
#define _FUNCTIONS_H_

#include <functional>
#include <mpi.h>
#include <vector>

#define TAG_SIZE 0
#define TAG_DATA 1

typedef std::pair<std::pair<int, int>, int> coo_entry_t;
typedef std::vector<coo_entry_t> coo_matrix_t;

void spgemm_2d(
    int m,
    int p,
    int n,
    std::vector<std::pair<std::pair<int, int>, int>> &A,
    std::vector<std::pair<std::pair<int, int>, int>> &B,
    std::vector<std::pair<std::pair<int, int>, int>> &C,
    std::function<int(int, int)> plus,
    std::function<int(int, int)> times,
    MPI_Comm row_comm,
    MPI_Comm col_comm);

void apsp(
    int n,
    std::vector<std::pair<std::pair<int, int>, int>> &graph,
    std::vector<std::pair<std::pair<int, int>, int>> &result,
    MPI_Comm row_comm,
    MPI_Comm col_comm);

void distribute_matrix_2d(
    int m,
    int n,
    std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
    std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
    int root,
    MPI_Comm comm_1d);

#endif
