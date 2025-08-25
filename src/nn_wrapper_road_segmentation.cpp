#include "road_segmentation_plugin/nn_wrapper_road_segmentation.hpp"
#include "road_segmentation_plugin/road_segmentation.hpp"

#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/node/DetectionNetwork.hpp"
#include "depthai_ros_driver/dai_nodes/nn/detection.hpp"
#include "depthai_ros_driver/dai_nodes/nn/segmentation.hpp"
// #include "depthai_ros_driver/dai_nodes/nn/road_segmentation.hpp"
#include "depthai_ros_driver/param_handlers/nn_param_handler.hpp"
#include "rclcpp/node.hpp"

namespace depthai_ros_driver {
namespace dai_nodes {
NNWrapperRoadSegmentation::NNWrapperRoadSegmentation(const std::string& daiNodeName,
                     std::shared_ptr<rclcpp::Node> node,
                     std::shared_ptr<dai::Pipeline> pipeline,
                     const dai::CameraBoardSocket& socket)
    : BaseNode(daiNodeName, node, pipeline) {
    RCLCPP_DEBUG(node->get_logger(), "Creating node %s base", daiNodeName.c_str());
    
    ph = std::make_unique<param_handlers::NNParamHandler>(node, daiNodeName, socket);
    auto family = ph->getNNFamily();
    nnNode = std::make_unique<dai_nodes::nn::RoadSegmentation>(getName(), getROSNode(), pipeline, socket);

    RCLCPP_DEBUG(node->get_logger(), "Base node %s created", daiNodeName.c_str());
}
NNWrapperRoadSegmentation::~NNWrapperRoadSegmentation() = default;

void NNWrapperRoadSegmentation::setNames() {}

void NNWrapperRoadSegmentation::setXinXout(std::shared_ptr<dai::Pipeline> /*pipeline*/) {}

void NNWrapperRoadSegmentation::setupQueues(std::shared_ptr<dai::Device> device) {
    nnNode->setupQueues(device);
}
void NNWrapperRoadSegmentation::closeQueues() {
    nnNode->closeQueues();
}

void NNWrapperRoadSegmentation::link(dai::Node::Input in, int linkType) {
    nnNode->link(in, linkType);
}

dai::Node::Input NNWrapperRoadSegmentation::getInput(int linkType) {
    return nnNode->getInput(linkType);
}

void NNWrapperRoadSegmentation::updateParams(const std::vector<rclcpp::Parameter>& params) {
    ph->setRuntimeParams(params);
    nnNode->updateParams(params);
}

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
