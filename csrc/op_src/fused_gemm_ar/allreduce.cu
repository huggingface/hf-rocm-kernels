#include <string>
//#include <mpi.h>
#include <mscclpp/core.hpp>
#include <mscclpp/utils.hpp>
#include <mscclpp/port_channel.hpp>
#include <mscclpp/memory_channel.hpp>

#include "skinny_gemm.cu"

class AllReduceEngine {
public:
    AllReduceEngine(int rank, int worldSize,
                    int port,
                    torch::Tensor& comms_buff_A,
                    torch::Tensor& comms_buff_B)
        : rank_(rank), worldSize_(worldSize) {
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        bootstrap(port);

        comms_buff_bytes_ = comms_buff_A.numel() * comms_buff_A.element_size();

        hipExtMallocWithFlags((void**)&comms_buff_A_, comms_buff_bytes_, hipDeviceMallocUncached);
        hipExtMallocWithFlags((void**)&comms_buff_B_, comms_buff_bytes_, hipDeviceMallocUncached);
        cudaMemset(comms_buff_A_, 0, comms_buff_bytes_);
        cudaMemset(comms_buff_B_, 0, comms_buff_bytes_);

        registered_buff_A_ = communicator_->registerMemory(comms_buff_A_, comms_buff_bytes_, transport);
        registered_buff_B_ = communicator_->registerMemory(comms_buff_B_, comms_buff_bytes_, transport);
        printf("Registered memory\n");


        auto getChannelDeviceHandle = [](const std::vector<mscclpp::MemoryChannel>& in,
            std::vector<DeviceHandle<mscclpp::MemoryChannel>>& out) {
            return std::transform(in.begin(), in.end(), out.begin(), [](const mscclpp::MemoryChannel& memoryChannel) {
                return mscclpp::deviceHandle(memoryChannel);
            });
        };

        setupMeshConnections(channels_A_, registered_buff_A_, registered_buff_B_, comms_buff_bytes_, 0);
        std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles_A(channels_A_.size());
        getChannelDeviceHandle(channels_A_, memoryChannelDeviceHandles_A);
        CUDATHROW(cudaMemcpyToSymbol(constRingChannelsA, memoryChannelDeviceHandles_A.data(),
            sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles_A.size()));
        printf("Copied channels to device\n");


        setupMeshConnections(channels_B_, registered_buff_B_, registered_buff_A_, comms_buff_bytes_, 1);
        std::vector<DeviceHandle<mscclpp::MemoryChannel>> memoryChannelDeviceHandles_B(channels_B_.size());
        getChannelDeviceHandle(channels_B_, memoryChannelDeviceHandles_B);
        CUDATHROW(cudaMemcpyToSymbol(constRingChannelsB, memoryChannelDeviceHandles_B.data(),
            sizeof(DeviceHandle<mscclpp::MemoryChannel>) * memoryChannelDeviceHandles_B.size()));
        printf("Copied channels to device\n");

        printf("Setup mesh connections\n");
        startProxy();

        //cudaEventCreate(&allreduce_lock_event);
        //cudaEventRecord(allreduce_lock_event, at::cuda::getCurrentCUDAStream());
    }

    ~AllReduceEngine() {
        //if (deviceChannels_) {
        //    //hipFree(deviceChannels_);
        //}
    }

    torch::Tensor reduce(
        torch::Tensor& A,
        torch::Tensor& B,
        torch::Tensor& D,
        torch::Tensor& scale_tensor,
        int64_t b_lanes,
        int64_t split_k,
        int64_t nBlocks,
        int64_t compute_warps,
        bool is_capturing) {
        TORCH_CHECK(A.is_cuda(), "Input tensor must be a CUDA tensor");
        TORCH_CHECK(A.is_contiguous(), "Input tensor must be contiguous");

        // Setup mesh connections
        //allocateCommsBuffers(D.numel() * D.element_size());
        // printf("Allocated input buffers\n");
        // comms_buff_bytes_ = D.numel() * D.element_size();
        // setupMeshConnections(channels_A_, comms_buff_A.data_ptr(), comms_buff_B.data_ptr(), comms_buff_bytes_);
        // CUDATHROW(cudaMemcpyToSymbol(constRingChannelsA, channels_A_.data(),
        //     sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_A_.size()));
        // printf("Copied channels to device\n");
        // setupMeshConnections(channels_B_, comms_buff_B.data_ptr(), comms_buff_A.data_ptr(), comms_buff_bytes_);
        // CUDATHROW(cudaMemcpyToSymbol(constRingChannelsB, channels_B_.data(),
        //     sizeof(DeviceHandle<mscclpp::PortChannel>) * channels_B_.size()));
        // printf("Copied channels to device\n");

        // printf("Setup mesh connections\n");
        // startProxy();

        //CUDATHROW(hipMemsetD16(reinterpret_cast<uint8_t *>(comms_buff_A_), 0, A.numel()));
        //CUDATHROW(hipMemsetD16(reinterpret_cast<uint8_t *>(comms_buff_B_), 0, A.numel()));
        //CUDATHROW(cudaDeviceSynchronize());
        //communicator_->bootstrap()->barrier();

        skinny_gemm(A, B, D, scale_tensor, b_lanes, split_k, rank_, worldSize_, reinterpret_cast<uint8_t *>(comms_buff_A_), reinterpret_cast<uint8_t *>(comms_buff_B_), allreduce_lock_event, nBlocks, compute_warps, is_capturing);
        //CUDATHROW(cudaDeviceSynchronize());
        cudaMemset(comms_buff_A_, 0, comms_buff_bytes_);
        //communicator_->bootstrap()->barrier();
        return D;
    }

private:
    void bootstrap(int port) {
        // Use longer timeout for initialization

        //MPI_Init(NULL, NULL);
        //MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
        //MPI_Comm_rank(MPI_COMM_WORLD, &rank_);

        printf("Rank %d: World size %d\n", rank_, worldSize_);

        std::string ip_port = "localhost:";
        auto bootstrap = std::make_shared<mscclpp::TcpBootstrap>(rank_, worldSize_);

        // Initialize with options
        //bootstrap->initialize(ip_port, options);
        //mscclpp::UniqueId id;
        //if (bootstrap->getRank() == 0) id = bootstrap->createUniqueId();
        //MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
        //bootstrap->initialize(id);
        bootstrap->initialize(ip_port.append(std::to_string(port)));
        bootstrap->barrier();
        printf("Initialized comms\n");

        // Create communicator and wait for all processes
        communicator_ = std::make_shared<mscclpp::Communicator>(bootstrap);
        chanService_ = std::make_shared<mscclpp::ProxyService>();
    }

    void allocateCommsBuffers(size_t bytes) {
        comm_buff_A = mscclpp::GpuBuffer<uint8_t>(bytes).memory();
        comm_buff_B = mscclpp::GpuBuffer<uint8_t>(bytes).memory();
        comms_buff_bytes_ = bytes;
    }

    void setupMeshConnections(std::vector<mscclpp::MemoryChannel>& memoryChannels, mscclpp::RegisteredMemory& sendBufRegMem, mscclpp::RegisteredMemory& recvBufRegMem, size_t buff_size, int tag = 0) {
        mscclpp::Transport transport = mscclpp::Transport::CudaIpc;
        std::vector<mscclpp::NonblockingFuture<mscclpp::RegisteredMemory>> remoteRegMemories;
        std::vector<mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>> connectionFutures;
        std::vector<std::shared_ptr<mscclpp::Connection>> connections;

        printf("Rank %d: Setting up mesh connections\n", rank_);

        // Connect with all other ranks
        for (int r = 0; r < worldSize_; ++r) {
            if (r == rank_) continue;
            connectionFutures.push_back(communicator_->connectOnSetup(r, tag, transport));
            communicator_->sendMemoryOnSetup(recvBufRegMem, r, tag);
            remoteRegMemories.push_back(communicator_->recvMemoryOnSetup(r, tag));
        }
        printf("Connected with all other ranks\n");

        communicator_->setup();

        printf("Setup communicator\n");

        std::transform(
            connectionFutures.begin(), connectionFutures.end(), std::back_inserter(connections),
            [](const mscclpp::NonblockingFuture<std::shared_ptr<mscclpp::Connection>>& future) { return future.get(); });
        printf("Got connections\n");


        std::vector<std::shared_ptr<mscclpp::MemoryDevice2DeviceSemaphore>> memorySemaphores;
        for (size_t cid = 0; cid < connections.size(); ++cid) {
          if (connections[cid]->transport() == mscclpp::Transport::CudaIpc) {
            memorySemaphores.emplace_back(
                std::make_shared<mscclpp::MemoryDevice2DeviceSemaphore>(*communicator_, connections[cid]));
          }
        }
        communicator_->setup();

        for (size_t cid = 0; cid < connections.size(); ++cid) {
            memoryChannels.emplace_back(
                memorySemaphores[cid], remoteRegMemories[cid].get(), sendBufRegMem.data(), nullptr);
        }

        printf("Created channels: %d\n", memoryChannels.size());
    }

    void startProxy() {
        this->chanService_->startProxy();
        communicator_->bootstrap()->barrier();
        printf("Started proxy\n");
    }


    int rank_;
    int worldSize_;
    std::shared_ptr<mscclpp::Communicator> communicator_;
    std::vector<mscclpp::MemoryChannel> channels_A_;
    std::vector<mscclpp::MemoryChannel> channels_B_;
    std::shared_ptr<mscclpp::BaseProxyService> chanService_;
    cudaStream_t stream_;

    std::shared_ptr<uint8_t> comm_buff_A;
    std::shared_ptr<uint8_t> comm_buff_B;

    mscclpp::RegisteredMemory registered_buff_A_;
    mscclpp::RegisteredMemory registered_buff_B_;

    uint8_t* comms_buff_A_;
    uint8_t* comms_buff_B_;
    size_t comms_buff_bytes_;

    cudaEvent_t allreduce_lock_event;
};


int64_t all_reduce_init(
    int64_t rank, int64_t worldSize, int64_t port,
    torch::Tensor& comms_buff_A,
    torch::Tensor& comms_buff_B
) {
    return (int64_t) new AllReduceEngine(rank, worldSize, port, comms_buff_A, comms_buff_B);
}

torch::Tensor all_reduce(
    int64_t allreduce_engine_ptr,
    torch::Tensor& A,
    torch::Tensor& B,
    torch::Tensor& D,
    torch::Tensor& scale_tensor,
    int64_t b_lanes,
    int64_t split_k,
    int64_t n_blocks,
    int64_t compute_warps,
    bool is_capturing) {
    AllReduceEngine* allreduce_engine = reinterpret_cast<AllReduceEngine*>(allreduce_engine_ptr);
    return allreduce_engine->reduce(A, B, D, scale_tensor, b_lanes, split_k, n_blocks, compute_warps, is_capturing);
}
