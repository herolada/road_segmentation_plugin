#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "depthai_ros_driver/dai_nodes/base_node.hpp"
#include "depthai_ros_driver/pipeline/base_pipeline.hpp"


namespace dai {
class Pipeline;
class Device;
}  // namespace dai

namespace rclcpp {
class Node;
}

namespace depthai_ros_driver {
namespace pipeline_gen {


class RGBDRoadSegmentation : public BasePipeline {
   public:
    std::vector<std::unique_ptr<dai_nodes::BaseNode>> createPipeline(std::shared_ptr<rclcpp::Node> node,
                                                                     std::shared_ptr<dai::Device> device,
                                                                     std::shared_ptr<dai::Pipeline> pipeline,
                                                                     const std::string& nnType) override;
};

}  // namespace pipeline_gen
}  // namespace depthai_ros_driver
