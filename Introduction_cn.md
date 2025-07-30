# StepMesh：面向 AF 分离系统的通信库

## 前言

[Step-3技术报告](https://arxiv.org/abs/2507.19427)发布后，我们收到不少咨询Attention-FFN分离方案的细节问题，其中很多是关于我们开源的通信库[StepMesh](https://github.com/stepfun-ai/StepMesh)。本文针对StepMesh详细展开一些技术细节和选型讨论，分享我们对AF分离系统在通信层面的需求思考和观察。

AFD 对通信库提出了很高的延迟要求。对于一个 3 阶段的 AFD 流水线，如果要满足端到端 20 Tokens/s 的 SLA 约束，需要在 273us（本文第二部分会介绍 273us 的缘由）内完成所有 Attention 实例和 FFN 实例之间的数据传输。此外，像 NCCL 和 DeepEP 这样的通信库会引入额外的通信 SM 占用开销，影响 Attention 和 FFN 的计算速度。AF 分离还引入了一种新颖的二分图通信模式，这与现有的集合通信接口（如 AllReduce、AllToAll 等）不同。现有集合通信库对二分图通信没有良好的原生支持。虽然可以使用 ncclSend/ncclRecv 等接口组合满足功能需求，但它们不可避免地会牺牲性能。为了解决上述问题，我们开发了 StepMesh，一个基于 GPUDirect RDMA 的专门用于 AF 分离的通信库，它提供低延迟、零 SM 占用和灵活的二分图通信能力。

基于对 AFD 通信模式的深入理解，开源项目 [**BytePs**](https://github.com/stepfun-ai/StepMesh) 能够完美适配 AFD 的通信需求，因此我们在 BytePS 代码基础上做了二次开发。对于基础通信组件，我们还是希望基于前人的肩膀做增量式的工作，不重复造轮子。同时，MegaScale-Infer 也为 AFD 系统的 MxN 通信库提供了参考设计。
在这些上述已有工作的基础上，我们根据 Step-3 的推理流量特征需求，规划了 AFD 的通信时序（Timeline），以满足 AFD 系统对低延迟的严苛要求。此外，我们还在探索以及解决 AFD 系统的性能瓶颈——Straggler 问题。除了上述工作，我们还将介绍其他技术选择，包括对 NCCL，IBGDA（InfiniBand GPUDirect Async）和多后端（Backend）架构的思考与应用。

## AFD 流量特征

![comm-require](docs/figs/comm-requiremnt.png)

**<p align="center">图1： 通信开销约束，以 1A1F 3 级流水线为例</p>**

在实际讨论 AFD 流量特征前，我们先展示 AFD 的通信约束。为简化分析（但不损失通用性），我们采取图 1 所示的 1A1F 3 级场景做讨论。图 1 中， $A_{1,1}$ 和 $A_{1,2}$ 分别表示 Layer 1 的 Micorbatch 1 和 2 的计算时间。如果要满足 20Tokens/s 的 SLA 约束，则 Time Per Output Token （TPOT）需要小于 50ms，即每层的计算和通信开销小于 50ms /# Layers。Step-3 具有 61 层，因此每层开销需要小于 819us。则有 $A_{1,1} + A_{1,2} + A_{1,3} \le 819us$ 。因为 Micorbatch 大小相同，则 $A_{x,y} \le 273us$ 。上述过程同样可以应用于 FFN 侧。如果进一步考虑通信，因为 $A_{1,1}$ 和 $A_{2,1}$ 之间存在数据依赖，因此 $A_{1,1}$ 所有相关的 FFN、通信过程都需要在 $A_{2,1}$ 发生前完成，因此有如下关系。

$$A_{1,1} + F_{1,1} + A2F + F2A \le A_{1,1} + A_{1,2} + A_{1,3}  $$

在实际运行系统中，我们会调节 Attention Instance 和 FFN Instance 的数量，以保证 Attention 开销和 FFN 开销配平，即 $A_{x,y} \approx F_{x, y}$ 。基于这一假设，上述不等式则可进一步简化为 $A2F + F2A \le A_{x,y} \le 273us$ 。即 AFD 通信过程中，如果要满足 20Tokens/s 的 SLA 约束，完成一次双向通信时间开销应小于 $273us$ 。
上述计算过程对于不同 SLA 和不同级数的流水线同样生效。

![comm-pattern](docs/figs/comm-pattern.png)

**<p align="center">图 2：StepMesh 通信模式，以 2A2F 3 级流水线（FFN 并行策略为 EP）为例</p>**

下面我们介绍 AFD 通信模式。如图 2 所示，StepMesh 要求为不同的 Microbatch 预注册内存（StepMesh 中的注册是指 RDMA Register MemoryRegion，这是进行 RDMA 通信操作的前提）。上述设计会造成额外的显存开销，但是能够消除相同 Layer 不同 Microbatch 之间的数据依赖，提高 Overlap 程度。在通信过程，A2F Tensor（包括 Tokens，Expert Distribution 等）由 Attention Instance 广播给所有的 FFN Instance，同时告知用于接收当前 Microbatch FFN 计算结果的 F2A Tensor 地址。FFN 计算完成后则直接将计算结果（图 2 中 F2A Tensor，主要为 Activitions）直接 RDMA 发送到 Attention 对应的 Tensor 中。实际上 FFN Instance 还涉及到机内 AllGather 等通信操作，因为是由其余组件实现的，所以本文不讨论。

基于上述通信延迟约束以及通信模式，我们可以进一步分析 StepMesh 通信的预期吞吐。对于特定模型和芯片，FFN 所支持的 Batch Size 相对比较固定，并且不受 Context Length 变化影响，因此我们围绕 FFN GPU 做通信数据量和开销计算。表 1 中展示了技术报告中 2A2F（Batch Size=128）的通信量（Expert Distribution 通信量由于较小，本部分暂不考虑）以及满足 SLA 需求前提下的理想通信开销。例如，对于 A2F 方向，每个 FFN GPU 需要接收来自两个 Attention GPU 的数据，数据量为 Batch Size x Hidden Size，下表中为 128 x 7168，需要在 91us 内完成。需要注意的是表 1 中的通信开销不仅包括物理网络 RDMA 传输开销，还包括通信库收发的软件处理开销。

| 方向 | Scale | 单位 | Dtype  | 数据量 （Bytes Per-FFN-GPU）| 满足 SLA 的有效吞吐 | 满足 SLA 的通信开销 （us）|
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| A2F | 2A->1F | Per-Layer | FP8 | 2 x 128 x 7168 x 1 | 161.3Gbps | 91us |
| F2A | 2F -> 2A | Per-Layer | BF16 | 2 x 128 x 7168 x 2 | 161.3Gbps | 182us |
| Overall |  | Per-Layer |  | 2 x 128 x 7168 x 3 | 161.3Gbps | 273us |

**<p align="center">表 1：满足 SLA 条件下，2A2F（Batch Size=128）场景下，StepMesh 理想通信开销</p>**

本节讨论了 AFD 的流量特征。在这种流量特征下，当满足上述延迟约束时，StepMesh 可以实现通信开销完全被计算隐藏。

## StepMesh Timeline

为实现上述目标，我们为 StepMesh 设计了如下 Timeline（以 3 级流水线为例）。当第 1 层的的 Microbatch 1 Attention 完成计算后，StepMesh 的 Net-Send 线程启动对 A2F Tensor 的 RDMA 发送操作，发送完成后，FFN 侧将接收到相应的信号，并启动 FFN 计算操作。在实际计算前，FFN 还需要执行一次 AllGather 操作，用于将不同 GPU 收到的 Tokens 分发到所有 FFN GPU。AllGather 完成后将执行后续计算操作。完成前置操作后，FFN 侧将调用 StepMesh 的 Net-Send 线程将计算结果发送至 Attention 侧。

![StepMesh Timeline](docs/figs/timeline.png)

**<p align="center">图 3：3 级流水线下，StepMesh Timeline</p>**

图 3 所设计的 Timeline 有两个特点：第一，不同 Microbatch 使用不同的 GPU Stream，以实现不同 Microbatch 之间 GPU Op 的 Overlap，例如，当前 Microbatch 的 FFN 机内 AllGather 操作，可以与上一 Microbatch 的 FFN 计算操作 Overlap。第二，不同 Microbatch 之间不存在数据依赖，因此 StepMesh 不需要等待上一 Microbatch，计算完成后可立即执行 RDMA 发送操作。
当然，StepMesh 还包含更多与低延迟相关的工程设计，以确保图 3 中每个线程或 Stream 的完成时间均能满足 SLA 要求。本文重点讨论 Timeline，旨在展示在当前 AFD 系统中，通信与计算如何协同工作，同时也为后续社区设计更优的 AFD 系统提供参考。

## StepMesh Straggler

AFD 对于计算和通信延迟有着极高要求，当任意节点出现计算、通信过程的变慢都会明确反应在 TPOT 上。TPOT 最小数值代表了系统的性能上限，而对于实际运行的推理系统，我们则更关注 TPOT 平均值。偶发延迟抖动会使得 TPOT 的 P99 上升数十毫秒，进而影响其平均值和最终吞吐。节点的长期降速则会使的 TPOT 的最小值、均值、P99 同时上升。我们在实测中发现，抖动主要来源于以下几类接口的调用：ibv_poll_cq、cudaEventQuery 和 cudaLaunchKernel。因为根因涉及到硬件和驱动，超出我们的定位能力，所以暂未有明确结论。下面我们展示如何在运行的 AFD 系统“挖出”异常抖动节点和发生异常抖动的位点，以及当前我们所采取的抑制抖动的解决方案。

在 AFD 通信时，Attention 端和 FFN 端接收在接收数据时，都会在接收到了所有对端发送来的包后再提交给应用层。因此所有节点上统计 TPOT 的数据会同步，无法展示出具体的 Straggler 节点。

![Tracer](docs/figs/tracer.png)

**<p align="center">图 4： StepMesh 监控信息</p>**

为了有效发现 Straggler 节点，我们采集了以下信息：
上图的黑色时间戳为 Attention 和 FFN 实现中使用 get_nanosecond API 采集的时间戳。而红色为 StepMesh 中开启 ENABLE_TRACE 后采集的时间戳，并由数据包的 Meta 携带。在 Attention 端即可算出与不同 FFN 通信时的网络耗时、FFN 端的耗时总时长，以及 FFN 端的 GPU 计算时长。使用这些信息，可以分析出以下降速问题：

- 网络异常：异常 FFN 节点的 Network Cost 数值升高；
- FFN 端异常：
  - CPU 异常，如干扰或降速等：异常 FFN 节点的 FFN Process 的数值轻微上升，而（Server Overall - FFN Process）明显升高；
  - GPU 或 NVLink 异常，如降频或显存错误：异常 FFN 节点 Server Overall 和 FFN Process 数值均升高；
- Attention 端异常：
  - CPU 异常：request_post_send - push_pull_star 和 attn_start - response_recv_msg 上升；
  - GPU 或 NVLink 异常：push_pull_start - attn_start 数值上升，即 Attention 计算时长上升；
- 两侧收到包的时间戳（request/response recv_msg）与相应的 waiting 结束时间戳间的时间也可以反应对端的情况，当对端是慢节点时，这一等待时间会更短。

以上的分析均使用同一侧服务器的时间戳的差值进行分析，因此不需要多台服务器之间进行高精度时钟同步。

在实际部署时，仅需在 Attention 侧进行采集：直接采集 attn_start 和 push_pull_start 两个时间戳，使用 fetch_trace api 获取不同 FFN 节点在图 4 中红色的时间戳。按照上文计算出各个时间间隔后，在各个 Attention 节点侧即可看出 Straggler 的 FFN 节点；将这些时间间隔上传至分析系统后，通过对比可以找到 Straggler 的 Attention 节点。

在发现 Straggler 节点后，除了直接替换节点，我们还总结了常见的抖动原因和解决方案：

- CPU 占用问题：其他进程，如监控进程、推理框架的进程等，对 CPU 的抢占会干扰 StepMesh 中 Polling 线程的运行，使得延迟增高，因此我们选择为 StepMesh 的每个线程都仔细设置了 CPU Core Affinity， 为了尽可能减少影响，我们还配置了 Linux kernel 的 isolcpus 的参数，将 StepMesh 线程都绑定在这些隔离的核心上；
- GIL 问题： Python 调用 StepMesh 接口时，GIL 的释放与获取有概率造成较高延迟，因此我们选择让高频接口不再释放 gil，并将部分逻辑放在 C++中，如 get_batch 的多端同步和 repsond_vec 的批量返回；
- CudaEvent 同步问题：CudaEventSync 等接口会有偶发的高延迟，因此我们根据通信逻辑，降低了其调用逻辑，并提供了 cudaEventQuery Polling 和 memory sync 两种更高性能的同步方式（可以通过环境变量切换）。

本节讨论了 StepMesh 发现抖动节点的方式，以及抑制抖动的一些工程方法。目前在我们的测试中 TPOT 的最小数值和均值仍有约 2ms 的差距，需要更精细的监控和更深度的优化来进一步降低这一差值、提升性能。

## 其它设计选择

### NCCL vs StepMesh

NCCL 是大模型训练推理的 SOTA（State-of-the-Art）集合通信库。在 AFD 系统中，我们没有选择 NCCL 的原因主要有以下四点：

1. 性能保证问题：根据 MegaScale-Infer 论文以及我们自己的测试结果，NCCL 在支持 MxN 通信的低延迟场景下无法提供可靠的性能保证。
2. 通信模式不匹配：NCCL 原生不支持二分图通信模式。考虑到 NCCL 本身二次开发的难度，我们选择了相对更轻量级且通信模式更匹配的 BytePS 作为二次开发的基础库。
3. 资源抢占问题：NCCL 依赖于专用通信 SM（Streaming Multiprocessor）进行执行操作，这与计算过程会抢占资源，导致整体性能下降。对于必须进行计算通信 Overlap 的 AFD 系统来说，这一问题的影响尤为显著。
4. 异构推理支持问题：AFD 系统的一个演进方向是支持异构推理。然而，NCCL 本身作为 Nvidia GPU 专用的通信库，在支持异构推理方面存在较高的改造挑战。

基于 IBGDA 技术为核心构建的 DeepEP 也是 AFD 的备选通信库之一，下面我们将讨论 StepMesh 在使用 IBGDA 方面的技术选择。

### CPU-Only vs IBGDA

基于 NVSHMEM 的 IBGDA 使用 GPU 直接处理 RDMA 控制面信息，DeepEP、Triton-distributed 等开源库中为 low latency 模式配置了 IBGDA 技术。IBGDA 能够有效降低 RDMA 小包的传输延迟，并避免 CPU 的干扰，有着更好的延迟稳定性。在当前版本中，我们没有使用 IBGDA，而是使用了 CPU-only 的 IBRC 设计，原因如下：

- 数据包更大：在我们的使用场景中，以 Step 3 Tech Report 的 2A2F 场景为例，每次传输的 Tensor 大小为 A->F 896KB，F->A 1.75MB，在 400Gbps 网络下，其传输时间为 37us 和 18us，此时 CPU 和 GPU 控制面的延迟影响并不大；
- CPU 性能满足需求：在为 StepMesh 的所有线程配置的严格的 CPU Core Affinity 和 isolcpu 配置后，在 2A2F 的典型场景中，端到端的波动降低到了 5ms；
- SM 占用：在一些配置下，FFN 的设计为计算 Bound，此时没有额外的 SM 可以用于 IBGDA 通信。

如 Tech Report 中的数据所示，CPU-Only 模式已经满足当前 StepMesh 和 Step 3 的性能需求。当然我们会在未来尝试引入 IBGDA 的能力，以适配更多的场景。引入 IBGDA 不可避免会带来 SM 占用问题，如我们前文所述，存在计算性能下降的风险。上述说法并非抵制 IBGDA 或认为采取 IBGDA 的通信库不好，而是 SM 占用冲突这个问题在计算通信 Overlap 场景一直存在，而 AFD 则要求计算通信严格 Overlap，我们认为做好计算和通信的 SM 资源 Trade-Off 是解决问题合理方式，因此我们依旧认为 IBGDA 是 StepMesh 解决性能问题的一个可能的演进技术方向。

为了规避或缓解 SM 占用问题，我们会考虑实现一个 CPU-GPU 协同的方案，处理开销极小但位于关键路径的操作由 GPU 完成，开销极大的操作或不位于关键路径的操作由 CPU 完成。当然，当前采取 CPU-Only 的方案有支持异构芯片方面的需求，因为 IBGDA 强绑定 NVIDIA 芯片。下面我们将阐释当前对于异构芯片支持的考虑。

### 异构芯片通信

在 AF 分离后，通过配置不同的比例的 AF 实例数量，并调整 Batch Size，Step 3 可以在不同计算-存储吞吐比的计算芯片上进行高效的推理部署。对于最高效率、最低成本的部署，Attention 和 FFN 两侧可以使用不同芯片。因此适配不同品牌、不同型号的计算卡，能够有效增强部署的灵活性和推理系统性价比。
为了能方便适配不同的计算芯片，我们将与计算芯片交互抽象为 Backend 对象和统一的 API，其功能包括：
- 内存管理： Alloc 与 Free，用于预注册 Buffer 的管理；
- 事件管理：用于 CPU 管理线程与 GPU Stream 的状态同步；
数据的搬移使用 GPU Direct RDMA 技术卸载给 RNIC 去处理。作为参考，当前开源版本包含了 CpuBackend 和 GpuBackend（NVIDIA Cuda）两种实现。更多相关内容可以参考我们的 [接口实现文档](docs/backend.md)。