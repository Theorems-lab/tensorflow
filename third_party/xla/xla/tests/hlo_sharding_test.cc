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

#include "xla/hlo/ir/hlo_sharding.h"

#include <cstdint>
#include <memory>

#include "xla/array.h"
#include "xla/array3d.h"
#include "xla/test.h"

namespace xla {
namespace {

TEST(HloShardingTest, V1ProtoRoundTrip) {
  Array3D<int64_t> array({{{0, 2, 1, 3}}, {{4, 6, 5, 7}}});
  TileAssignment ta(std::make_shared<const Array<int64_t>>(array));
  auto tile_sharding = HloSharding::Tile(ta);
  auto proto = tile_sharding.ToProto();
  proto.set_replicate_on_last_tile_dim(true);
  ASSERT_OK_AND_ASSIGN(auto partial_tile_sharding,
                       HloSharding::FromProto(proto));
  auto round_trip_proto = partial_tile_sharding.ToProto();
  ASSERT_OK_AND_ASSIGN(auto round_trip_sharding,
                       HloSharding::FromProto(round_trip_proto));
  EXPECT_EQ(partial_tile_sharding, round_trip_sharding);
}

TEST(HloShardingTest, V1ProtoCanonicalizedReplicated) {
  OpSharding proto;
  proto.set_type(OpSharding::OTHER);
  proto.add_tile_assignment_dimensions(1);
  proto.add_tile_assignment_dimensions(3);
  proto.add_tile_assignment_devices(2);
  proto.add_tile_assignment_devices(1);
  proto.add_tile_assignment_devices(0);
  proto.set_replicate_on_last_tile_dim(true);
  ASSERT_OK_AND_ASSIGN(auto sharding, HloSharding::FromProto(proto));
  EXPECT_TRUE(sharding.IsReplicated());
}

TEST(HloShardingTest, V1ProtoCanonicalizedFullyTiled) {
  OpSharding proto;
  proto.set_type(OpSharding::OTHER);
  proto.add_tile_assignment_dimensions(3);
  proto.add_tile_assignment_dimensions(1);
  proto.add_tile_assignment_devices(2);
  proto.add_tile_assignment_devices(1);
  proto.add_tile_assignment_devices(0);
  proto.set_replicate_on_last_tile_dim(true);
  ASSERT_OK_AND_ASSIGN(auto sharding, HloSharding::FromProto(proto));
  EXPECT_TRUE(sharding.IsTiled());
  EXPECT_FALSE(sharding.ReplicateOnLastTileDim());
}

}  // namespace
}  // namespace xla
