/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_
#define XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_

#include <atomic>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "absl/container/fixed_array.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/blocking_counter.h"
#include "absl/types/span.h"
#include "xla/service/cpu/runtime/thunk.h"

namespace xla::cpu {

// A dataflow-style (run when ready) executor for a ThunkSequence that depends
// on buffer uses to build a DAG defining execution order. At run time executes
// thunks concurrently in a given thread pool.
class ThunkExecutor {
 public:
  // It's up to the caller to provide the task runner that will execute tasks
  // produced by the executor. It can be a simple inline executor that runs
  // tasks on the same thread, or a runner backed by a thread pool.
  using Task = absl::AnyInvocable<void()>;
  using TaskRunner = absl::AnyInvocable<void(Task)>;

  // Nodes identified by their index in the captured ThunkSequence.
  using NodeId = int64_t;

  static constexpr NodeId kInvalidNodeId = std::numeric_limits<NodeId>::min();

  ThunkExecutor(ThunkExecutor&&) = default;
  ThunkExecutor& operator=(ThunkExecutor&&) = default;

  static absl::StatusOr<ThunkExecutor> Create(ThunkSequence thunk_sequence);

  // NodeDef defines an execution order for all thunks in a sequence.
  struct NodeDef {
    NodeId id = kInvalidNodeId;
    std::vector<NodeId> in_edges;
    std::vector<NodeId> out_edges;
  };

  // Executes the thunk sequence using the prepared dataflow graph. Executor
  // uses runner to execute ready tasks concurrently. If runner is not provided,
  // executes all tasks in the caller thread.
  absl::Status Execute(const Thunk::ExecuteParams& params,
                       TaskRunner runner = nullptr);

  absl::Span<const NodeDef> nodes_defs() const { return nodes_defs_; }
  const NodeDef& node_def(NodeId id) const { return nodes_defs_[id]; }

  absl::Span<const NodeId> source() const { return source_; }
  absl::Span<const NodeId> sink() const { return sink_; }

  std::string ToString() const;

 private:
  using ReadyQueue = absl::InlinedVector<NodeId, 8>;

  ThunkExecutor(ThunkSequence thunk_sequence, std::vector<NodeDef> nodes_defs);

  // At run time NodeDef instantiated as a Node with an atomic counter that
  // drops to zero when all in_edges are ready.
  struct Node {
    NodeId id = kInvalidNodeId;
    std::atomic<int64_t>* counter = nullptr;
    const std::vector<NodeId>* out_edges = nullptr;
  };

  // A struct to keep the state of a running executor.
  struct ExecuteState {
    ExecuteState(ThunkExecutor* executor, TaskRunner runner);

    ThunkExecutor* executor;
    TaskRunner runner;

    absl::FixedArray<std::atomic<int64_t>> counters;
    absl::InlinedVector<Node, 32> nodes;
    absl::BlockingCounter done;
  };

  // Executes nodes in the ready queue with given thunk parameters.
  absl::Status Execute(ExecuteState* state, const Thunk::ExecuteParams& params,
                       ReadyQueue ready_queue);

  // Processes out edges of a completed `node` and updates `ready_queue` with
  // nodes that are ready to execute.
  void ProcessOutEdges(ExecuteState* state, Node& node,
                       ReadyQueue& ready_queue);

  ThunkSequence thunk_sequence_;
  std::vector<NodeDef> nodes_defs_;

  std::vector<NodeId> source_;
  std::vector<NodeId> sink_;
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_THUNK_EXECUTOR_H_
