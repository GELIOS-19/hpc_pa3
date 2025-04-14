#include "functions.h"

#include <array>
#include <cassert>
#include <iostream>
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
            [](int a, int b) { return 0; },
            // TODO: Choose operation here
            [](int a, int b) { return 0; },
            row_comm,
            col_comm);
        max_iter *= 2;
    }
    result = L;
}
