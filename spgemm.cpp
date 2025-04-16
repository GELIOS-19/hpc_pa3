#include <mpi.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <map>
#include <unordered_map>
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

static inline void pack_coo_matrix(
    coo_matrix_t const &unpacked_matrix,
    std::vector<int> &packed_matrix)
{
    packed_matrix.resize(3 * unpacked_matrix.size());

    for (size_t i = 0; i < unpacked_matrix.size(); i++)
    {
        packed_matrix[3 * i] = unpacked_matrix[i].first.first;
        packed_matrix[3 * i + 1] = unpacked_matrix[i].first.second;
        packed_matrix[3 * i + 2] = unpacked_matrix[i].second;
    }
}

static inline void unpack_coo_matrix(
    coo_matrix_t &unpacked_matrix,
    std::vector<int> const &packed_matrix)
{
    unpacked_matrix.resize(packed_matrix.size() / 3);

    for (size_t i = 0; i < packed_matrix.size() / 3; i++)
    {
        unpacked_matrix[i] = std::make_pair(
            std::make_pair(packed_matrix[3 * i], packed_matrix[3 * i + 1]),
            packed_matrix[3 * i + 2]);
    }
}

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
    int row_rank, row_size, col_rank, col_size;
    MPI_Comm_rank(row_comm, &row_rank);
    MPI_Comm_size(row_comm, &row_size);
    MPI_Comm_rank(col_comm, &col_rank);
    MPI_Comm_size(col_comm, &col_size);
    // We own A[row_rank, col_rank] and B[row_rank, col_rank]

    // Both the row size and the column size should be equal, since the layout
    // is a square
    assert(row_size == col_size);
    const int k_sqrt_num_procs = row_size;

    // Set up an intermediate data structure to store the local computation of C
    std::unordered_map<int64_t, int> local_C;

    // MPI Requests for non-blocking communication
    MPI_Request reqs[4]; // 0: A_size, 1: A_data, 2: B_size, 3: B_data
    MPI_Status stats[4];

    // SUMMA loop
    for (int k = 0; k < k_sqrt_num_procs; k++)
    {
        // If we own the block at A[row_rank, k], broadcast it to all processors
        // in row row_rank. Only do so if the block has non-zero entries
        coo_matrix_t global_A;
        if (row_rank == k)
            global_A = A;

        std::vector<int> A_packed_matrix_buffer;
        pack_coo_matrix(global_A, A_packed_matrix_buffer);

        // If we own the block at A[k, col_rank], broadcast it to all processors
        // in col col_rank. Only do so if the block has non-zero entries
        coo_matrix_t global_B;
        if (col_rank == k)
            global_B = B;

        std::vector<int> B_packed_matrix_buffer;
        pack_coo_matrix(global_B, B_packed_matrix_buffer);

        // Broadcast sizes
        int A_packed_matrix_buffer_size = A_packed_matrix_buffer.size();
        int B_packed_matrix_buffer_size = B_packed_matrix_buffer.size();

        MPI_Ibcast(
            &A_packed_matrix_buffer_size,
            1,
            MPI_INT,
            k,
            row_comm,
            &reqs[0]);
        MPI_Ibcast(
            &B_packed_matrix_buffer_size,
            1,
            MPI_INT,
            k,
            col_comm,
            &reqs[2]);

        // Broadcast data
        MPI_Wait(&reqs[0], &stats[0]);
        if (row_rank != k)
            A_packed_matrix_buffer.resize(A_packed_matrix_buffer_size);

        MPI_Ibcast(
            A_packed_matrix_buffer.data(),
            A_packed_matrix_buffer_size,
            MPI_INT,
            k,
            row_comm,
            &reqs[1]);

        MPI_Wait(&reqs[2], &stats[2]);
        if (col_rank != k)
            B_packed_matrix_buffer.resize(B_packed_matrix_buffer_size);

        MPI_Ibcast(
            B_packed_matrix_buffer.data(),
            B_packed_matrix_buffer_size,
            MPI_INT,
            k,
            col_comm,
            &reqs[3]);

        // Recieve and unpack broadcasted data
        coo_matrix_t recv_A;
        MPI_Wait(&reqs[1], &stats[1]);
        unpack_coo_matrix(recv_A, A_packed_matrix_buffer);

        coo_matrix_t recv_B;
        MPI_Wait(&reqs[3], &stats[3]);
        unpack_coo_matrix(recv_B, B_packed_matrix_buffer);

        // Hande sparse matrix multiplication logic:
        //   We need to perform block matrix multiplication, which follows the
        //   same structure as regular matrix multiplication. We have the following
        //   blocks available to us here:
        //      A[row_rank, k], B[k, col_rank], C[row_rank, col_rank]

        // We can hash the B matrix for efficiency
        std::unordered_map<int, std::vector<std::pair<int, int>>>
            B_entry_lookup;
        for (auto const &[B_idx, B_value] : recv_B)
        {
            B_entry_lookup[B_idx.first]
                .push_back(std::make_pair(B_idx.second, B_value));
        }

        for (auto const &[A_idx, A_value] : recv_A)
        {
            int global_row_A = A_idx.first;
            int inner_dim_A = A_idx.second;

            std::vector<std::pair<int, int>> const &B_values =
                B_entry_lookup[inner_dim_A];

            for (auto const &[global_col_B, B_value] : B_values)
            {
                int product = times(A_value, B_value);

                // Calculate the key for the output C(i, j) entry
                // Use global row from A and global col from B
                int C_row = global_row_A;
                int C_col = global_col_B;

                int64_t key = static_cast<int64_t>(C_row) * n + C_col;

                // Accumulate the product into the map
                if (local_C.count(key))
                    local_C[key] = plus(local_C[key], product);
                else
                    local_C[key] = product;
            }
        }
    }

    // Load the local_C into C
    C.clear();
    for (auto const &kvp : local_C)
    {
        int C_row = kvp.first / n;
        int C_col = kvp.first % n;
        C.push_back({ { C_row, C_col }, kvp.second });
    }
}
