#pragma once

#include "depthai_ros_driver/dai_nodes/nn/nn_wrapper.hpp"
#include <depthai_ros_driver/param_handlers/nn_param_handler.hpp>

#include <memory>
#include <string>
#include <vector>

#include "depthai-shared/common/CameraBoardSocket.hpp"
#include "depthai_ros_driver/dai_nodes/base_node.hpp"
#include "road_segmentation_plugin/road_segmentation.hpp"

namespace dai {
class Pipeline;
class Device;
}  // namespace dai

namespace rclcpp {
class Node;
class Parameter;
}  // namespace rclcpp

namespace depthai_ros_driver {
namespace param_handlers {
class NNParamHandler;
}

namespace dai_nodes {

class NNWrapperRoadSegmentation : public NNWrapper {
    public:
        // Constructor: must call NNWrapper's constructor in the initializer list
        explicit NNWrapperRoadSegmentation(const std::string& daiNodeName,
                            std::shared_ptr<rclcpp::Node> node,
                            std::shared_ptr<dai::Pipeline> pipeline,
                            const dai::CameraBoardSocket& socket = dai::CameraBoardSocket::CAM_A)
            : NNWrapper(daiNodeName, node, pipeline, socket) {
            // nnNode->closeQueues();
            ph.reset();
            nnNode.reset();

            // RCLCPP_WARN(node->get_logger(),"a %s", daiNodeName.c_str());
            // RCLCPP_WARN(node->get_logger(),"b %s", getName().c_str());

            setNodeName("road");
            ph = std::make_unique<param_handlers::NNParamHandler>(node, getName(), socket);
            auto family = ph->getNNFamily();

            nnNode = std::make_unique<dai_nodes::nn::RoadSegmentation>(getName(), getROSNode(), pipeline, socket);
            std::cout << "NNWrapperRoadSegmentation constructed" << std::endl;
        }

        ~NNWrapperRoadSegmentation() {
            std::cout << "NNWrapperRoadSegmentation destructed" << std::endl;
        }
};

// class NNWrapper : public BaseNode {
//    public:
//     explicit NNWrapper(const std::string& daiNodeName,
//                        std::shared_ptr<rclcpp::Node> node,
//                        std::shared_ptr<dai::Pipeline> pipeline,
//                        const dai::CameraBoardSocket& socket = dai::CameraBoardSocket::CAM_A);
//     ~NNWrapper();
//     void updateParams(const std::vector<rclcpp::Parameter>& params) override;
//     void setupQueues(std::shared_ptr<dai::Device> device) override;
//     void link(dai::Node::Input in, int linkType = 0) override;
//     dai::Node::Input getInput(int linkType = 0) override;
//     virtual void setNames() override;
//     virtual void setXinXout(std::shared_ptr<dai::Pipeline> pipeline) override;
//     void closeQueues() override;

//    private:
//     std::unique_ptr<param_handlers::NNParamHandler> ph;
//     std::unique_ptr<BaseNode> nnNode;
// };

}  // namespace dai_nodes
}  // namespace depthai_ros_driver
