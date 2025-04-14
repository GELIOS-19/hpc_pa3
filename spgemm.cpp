#include "functions.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <mpi.h>
#include <utility>
#include <vector>

#ifndef SUBMIT
#define SUBMIT false
#if SUMBMIT

#define TAG_SIZE 0
#define TAG_DATA 1

typedef std::pair<std::pair<int, int>, int> coo_entry_t;
typedef std::vector<coo_entry_t> coo_matrix_t;

#endif
#endif

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
    MPI_Comm col_comm)
{
    // Get the rank of the processor
    int row_rank, col_rank;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_rank(col_comm, &col_rank);
}
