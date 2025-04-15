#include <mpi.h>

#include <array>
#include <cassert>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

#include "functions.h"

#ifndef TAG_SIZE
#define TAG_SIZE 0
#endif
#ifndef TAG_DATA
#define TAG_DATA 1
#endif

typedef std::pair<std::pair<int, int>, int> coo_entry_t;
typedef std::vector<coo_entry_t> coo_matrix_t;

void apsp(
    int n,
    std::vector<std::pair<std::pair<int, int>, int>> &graph,
    std::vector<std::pair<std::pair<int, int>, int>> &result,
    MPI_Comm row_comm,
    MPI_Comm col_comm)
{
    std::vector<std::pair<std::pair<int, int>, int>> L = graph;

    int max_iter = 1;
    while (max_iter < n)
    {
        std::vector<std::pair<std::pair<int, int>, int>> L_tmp = std::move(L);
        spgemm_2d(
            n,
            n,
            n,
            L_tmp,
            L_tmp,
            L,
            // TODO: Choose operation here
            [](int a, int b) { return std::min(a, b); },
            // TODO: Choose operation here
            [](int a, int b)
            {
                return a == std::numeric_limits<int>::max() / 2
                               || b == std::numeric_limits<int>::max() / 2
                           ? std::numeric_limits<int>::max() / 2
                           : a + b;
            },
            row_comm,
            col_comm);
        max_iter *= 2;
    }
    result = L;
}
