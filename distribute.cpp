#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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

void distribute_matrix_2d(
    int m,
    int n,
    std::vector<std::pair<std::pair<int, int>, int>> &full_matrix,
    std::vector<std::pair<std::pair<int, int>, int>> &local_matrix,
    int root,
    MPI_Comm comm_2d)
{
    // Get the rank of the processor
    int rank;
    MPI_Comm_rank(comm_2d, &rank);

    // Get the topology data
    int dims[2];
    [[maybe_unused]] int periods[2], coords[2];
    MPI_Cart_get(comm_2d, 2, dims, periods, coords);
    const int k_grid_rows = dims[0], k_grid_cols = dims[1];
    const int k_num_procs = k_grid_rows * k_grid_cols;

    // Clear the local matrix
    local_matrix.clear();

    // Root processor distributes the non-zero entries of the matrix
    if (rank == root)
    {
        // Set up destination rank data
        std::vector<coo_matrix_t> dest_buffers = std::vector<coo_matrix_t>(k_num_procs);

        for (const auto &[idx, value] : full_matrix)
        {
            int target_pr = (idx.first * k_grid_rows) / m,
                target_pc = (idx.second * k_grid_cols) / n;

            // Using this, we obtain the rank of the destination processor
            int dest_rank;
            int coords[2] = { target_pr, target_pc };
            MPI_Cart_rank(comm_2d, coords, &dest_rank);

            // Add the data that needs to be send to the destination buffer
            dest_buffers[dest_rank].push_back(std::make_pair(idx, value));
        }

        // Create destined data to be send to the destination rank
        std::vector<int> dest_buffer_sizes = std::vector<int>(k_num_procs);
        std::vector<std::vector<int>> packed_dest_buffers =
            std::vector<std::vector<int>>(k_num_procs);

        for (int proc_rank = 0; proc_rank < k_num_procs; proc_rank++)
        {
            // Compute the buffer size first
            dest_buffer_sizes[proc_rank] = dest_buffers[proc_rank].size();

            // Create the packed data
            std::vector<int> &packed_dest_buffer = packed_dest_buffers[proc_rank];
            packed_dest_buffer.resize(3 * dest_buffer_sizes[proc_rank]);

            for (int i = 0; i < dest_buffer_sizes[proc_rank]; i++)
            {
                packed_dest_buffer[3 * i + 0] = dest_buffers[proc_rank][i].first.first;
                packed_dest_buffer[3 * i + 1] = dest_buffers[proc_rank][i].first.second;
                packed_dest_buffer[3 * i + 2] = dest_buffers[proc_rank][i].second;
            }

            if (proc_rank == root)
            {
                local_matrix = dest_buffers[proc_rank];
            }
        }

        // Setup send request vector for non-blocking sends
        // We anticipate 2 send requests per grid rank:
        //     1 for the buffer size
        //     1 for the buffer data
        std::vector<MPI_Request> send_requests;
        send_requests.reserve(2 * (k_num_procs - 1));

        // Send destined data to the destination rank using non-blocking sends
        for (int proc_rank = 0; proc_rank < k_num_procs; proc_rank++)
        {
            if (proc_rank == root)
            {
                continue;
            }

            // Send the buffer size
            send_requests.emplace_back();
            MPI_Isend(
                &dest_buffer_sizes[proc_rank],
                1,
                MPI_INT,
                proc_rank,
                TAG_SIZE,
                comm_2d,
                &send_requests.back());

            // Send the buffer data
            if (dest_buffer_sizes[proc_rank] == 0)
            {
                continue;
            }

            send_requests.emplace_back();
            MPI_Isend(
                packed_dest_buffers[proc_rank].data(),
                3 * dest_buffer_sizes[proc_rank],
                MPI_INT,
                proc_rank,
                TAG_DATA,
                comm_2d,
                &send_requests.back());
        }

        // Wait for all the requests
        if (!send_requests.empty())
        {
            MPI_Waitall(static_cast<int>(send_requests.size()), send_requests.data(), MPI_STATUSES_IGNORE);
        }
    }

    if (rank != root)
    {
        // Recieve the data size
        int dest_buffer_size;
        MPI_Recv(&dest_buffer_size, 1, MPI_INT, root, TAG_SIZE, comm_2d, MPI_STATUS_IGNORE);

        if (dest_buffer_size > 0)
        {
            // Receive the data
            std::vector<int> packed_buffer = std::vector<int>(3 * dest_buffer_size);
            MPI_Recv(packed_buffer.data(), 3 * dest_buffer_size, MPI_INT, root, TAG_DATA, comm_2d, MPI_STATUS_IGNORE);

            // Distribute the data into the local matrix
            for (int i = 0; i < 3 * dest_buffer_size; i += 3)
            {
                local_matrix.push_back(
                    std::make_pair(
                        std::make_pair(packed_buffer[i], packed_buffer[i + 1]),
                        packed_buffer[i + 2]));
            }
        }
    }
}