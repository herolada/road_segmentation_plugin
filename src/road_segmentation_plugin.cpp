#include <memory>
#include <string>
#include <vector>

#include "depthai-shared/common/CameraBoardSocket.hpp"
#include "depthai_ros_driver/dai_nodes/base_node.hpp"

#define private public
#include "depthai_ros_driver/dai_nodes/nn/nn_wrapper.hpp"
#undef private

#include "depthai_ros_driver/pipeline/base_types.hpp"

#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai_ros_driver/dai_nodes/base_node.hpp"
#include "depthai_ros_driver/dai_nodes/nn/nn_helpers.hpp"
#include "depthai_ros_driver/dai_nodes/nn/spatial_nn_wrapper.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/imu.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/sensor_helpers.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/sensor_wrapper.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/stereo.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/tof.hpp"
#include "depthai_ros_driver/pipeline/base_pipeline.hpp"
#include "depthai_ros_driver/utils.hpp"
#include "rclcpp/node.hpp"

#include "road_segmentation_plugin/nn_wrapper_road_segmentation.hpp"
#include "road_segmentation_plugin/road_segmentation_plugin.hpp"
#include "road_segmentation_plugin/road_segmentation.hpp"



namespace depthai_ros_driver {
namespace pipeline_gen {
std::unique_ptr<dai_nodes::BaseNode> roadSegmentationNN(std::shared_ptr<rclcpp::Node> node, std::shared_ptr<dai::Pipeline> pipeline, dai_nodes::BaseNode& daiNode) {
    using namespace dai_nodes::sensor_helpers;
    auto nn = std::make_unique<dai_nodes::NNWrapperRoadSegmentation>(getNodeName(node, NodeNameEnum::NN), node, pipeline);
    daiNode.link(nn->getInput(), static_cast<int>(dai_nodes::link_types::RGBLinkType::preview));
    return nn;
}

std::vector<std::unique_ptr<dai_nodes::BaseNode>> RGBDRoadSegmentation::createPipeline(std::shared_ptr<rclcpp::Node> node,
                                                                                std::shared_ptr<dai::Device> device,
                                                                                std::shared_ptr<dai::Pipeline> pipeline,
                                                                                const std::string& /*nnType*/) {
    using namespace dai_nodes::sensor_helpers;

    std::vector<std::unique_ptr<dai_nodes::BaseNode>> daiNodes;
    auto rgb = std::make_unique<dai_nodes::SensorWrapper>(getNodeName(node, NodeNameEnum::RGB), node, pipeline, device, dai::CameraBoardSocket::CAM_A);
    auto stereo = std::make_unique<dai_nodes::Stereo>(getNodeName(node, NodeNameEnum::Stereo), node, pipeline, device);
    auto nn = roadSegmentationNN(node, pipeline, *rgb);

    daiNodes.push_back(std::move(nn));
    daiNodes.push_back(std::move(rgb));
    daiNodes.push_back(std::move(stereo));
    return daiNodes;
}

}  // namespace pipeline_gen
}  // namespace depthai_ros_driver

#include <pluginlib/class_list_macros.hpp>

PLUGINLIB_EXPORT_CLASS(depthai_ros_driver::pipeline_gen::RGBDRoadSegmentation, depthai_ros_driver::pipeline_gen::BasePipeline)
