#include "road_segmentation_plugin/road_segmentation.hpp"

#include "camera_info_manager/camera_info_manager.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include "depthai/device/DataQueue.hpp"
#include "depthai/device/Device.hpp"
#include "depthai/pipeline/Pipeline.hpp"
#include "depthai/pipeline/datatype/NNData.hpp"
#include "depthai/pipeline/node/ImageManip.hpp"
#include "depthai/pipeline/node/NeuralNetwork.hpp"
#include "depthai/pipeline/node/XLinkOut.hpp"
#include "depthai_bridge/depthaiUtility.hpp"
#include "depthai_bridge/ImageConverter.hpp"
#include "depthai_ros_driver/dai_nodes/sensors/sensor_helpers.hpp"
#include "depthai_ros_driver/param_handlers/nn_param_handler.hpp"
#include "depthai_ros_driver/utils.hpp"
#include "image_transport/camera_publisher.hpp"
#include "image_transport/image_transport.hpp"
#include "rclcpp/node.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "depthai_bridge/BridgePublisher.hpp"



cv::Mat compute_normalized_entropy(const std::vector<cv::Mat> &logits, double epsilon = 1e-10) {
    int num_classes = logits.size();
    int height = logits[0].rows;
    int width  = logits[0].cols;

    // 1. Find max logit per pixel (for numerical stability)
    cv::Mat max_logits(height, width, CV_32F, cv::Scalar(-FLT_MAX));
    for (int c = 0; c < num_classes; ++c)
        cv::max(max_logits, logits[c], max_logits);

    // 2. Compute exp(logit - max_logit) for each class
    std::vector<cv::Mat> exp_logits(num_classes);
    for (int c = 0; c < num_classes; ++c) {
        exp_logits[c] = cv::Mat(height, width, CV_32F);
        cv::subtract(logits[c], max_logits, exp_logits[c]);
        cv::exp(exp_logits[c], exp_logits[c]);
    }

    // 3. Sum exp_logits over channels
    cv::Mat sum_exp(height, width, CV_32F, cv::Scalar(0));
    for (int c = 0; c < num_classes; ++c)
        cv::add(sum_exp, exp_logits[c], sum_exp);

    // 4. Compute probabilities: exp_logits / sum_exp
    std::vector<cv::Mat> prob(num_classes);
    for (int c = 0; c < num_classes; ++c)
        cv::divide(exp_logits[c], sum_exp, prob[c]);

    // 5. Compute entropy: -Σ p * log(p + ε)
    cv::Mat entropy(height, width, CV_32F, cv::Scalar(0));
    for (int c = 0; c < num_classes; ++c) {
        cv::Mat log_p;
        cv::log(prob[c] + epsilon, log_p);
        cv::Mat p_logp = prob[c].mul(log_p);
        cv::subtract(entropy, p_logp, entropy);  // entropy -= p * log(p)
    }

    // 6. Normalize entropy by log(num_classes)
    double max_entropy = std::log(static_cast<double>(num_classes));
    entropy /= static_cast<float>(max_entropy);

    return entropy;  // CV_32F, range [0, 1]
}

namespace depthai_ros_driver {
namespace dai_nodes {
namespace nn {

RoadSegmentation::RoadSegmentation(const std::string& daiNodeName,
                           std::shared_ptr<rclcpp::Node> node,
                           std::shared_ptr<dai::Pipeline> pipeline,
                           const dai::CameraBoardSocket& socket)
    : BaseNode(daiNodeName, node, pipeline) {
    RCLCPP_DEBUG(getLogger(), "Creating node %s", daiNodeName.c_str());
    setNames();
    segNode = pipeline->create<dai::node::NeuralNetwork>();
    imageManip = pipeline->create<dai::node::ImageManip>();
    ph = std::make_unique<param_handlers::NNParamHandler>(node, daiNodeName, socket);
    ph->declareParams(segNode, imageManip);

    classCosts[static_cast<size_t>(SegmentationClass::BACKGROUND)] = node->declare_parameter("background_cost", 1.0);
    classCosts[static_cast<size_t>(SegmentationClass::ROAD)] = node->declare_parameter("road_cost", 0.0);
    classCosts[static_cast<size_t>(SegmentationClass::SKY)] = node->declare_parameter("sky_cost", 1.0);

    normalizedEntropyThreshold = node->declare_parameter("normalized_entropy_threshold", 0.5);

    RCLCPP_DEBUG(getLogger(), "Node %s created", daiNodeName.c_str());
    RCLCPP_WARN(getLogger(), "ROAD SEGMENTATION blocking %i size %i wait %i",
        segNode->input.blocking.value_or(segNode->input.defaultBlocking),
        segNode->input.queueSize.value_or(segNode->input.defaultQueueSize),
        segNode->input.waitForMessage.value_or(segNode->input.defaultWaitForMessage));
    segNode->input.setBlocking(false);
    segNode->input.setQueueSize(1);
    segNode->input.setWaitForMessage(false);
    imageManip->out.link(segNode->input);
    setXinXout(pipeline);
}

RoadSegmentation::~RoadSegmentation() = default;

void RoadSegmentation::setNames() {
    nnQName = getName() + "_nn";
    ptQName = getName() + "_pt";
}

void RoadSegmentation::setXinXout(std::shared_ptr<dai::Pipeline> pipeline) {
    xoutNN = pipeline->create<dai::node::XLinkOut>();
    xoutNN->setStreamName(nnQName);
    segNode->out.link(xoutNN->input);
    if(ph->getParam<bool>("i_enable_passthrough")) {
        xoutPT = pipeline->create<dai::node::XLinkOut>();
        xoutPT->setStreamName(ptQName);
        segNode->passthrough.link(xoutPT->input);
    }
}

void RoadSegmentation::setupQueues(std::shared_ptr<dai::Device> device) {
    // CAMERA INFO
    int width = imageManip->initialConfig.getResizeWidth();
    int height = imageManip->initialConfig.getResizeHeight();

    frame = getOpticalTFPrefix(getSocketName(static_cast<dai::CameraBoardSocket>(ph->getParam<int>("i_board_socket_id"))));
    dai::rosBridge::ImageConverter converter(frame, true);
    auto calibrationHandler = device->readCalibration();
    nnInfo = converter.calibrationToCameraInfo(calibrationHandler, static_cast<dai::CameraBoardSocket>(ph->getParam<int>("i_board_socket_id")), width, height);

    nnQ = device->getOutputQueue(nnQName, ph->getParam<int>("i_max_q_size"), false);
    // nnPub = image_transport::create_camera_publisher(getROSNode().get(), "~/" + getName() + "/image_raw");
    nnPub_mask = image_transport::create_camera_publisher(getROSNode().get(), "~/" + getName() + "/mask/image_raw");
    nnPub_entropy = image_transport::create_camera_publisher(getROSNode().get(), "~/" + getName() + "/entropy/image_raw");
    nnPub_cost = image_transport::create_camera_publisher(getROSNode().get(), "~/" + getName() + "/cost/image_raw");
    nnQ->addCallback(std::bind(&RoadSegmentation::segmentationCB, this, std::placeholders::_1, std::placeholders::_2));

    if(ph->getParam<bool>("i_enable_passthrough")) {
        auto tfPrefix = getOpticalTFPrefix(getSocketName(static_cast<dai::CameraBoardSocket>(ph->getParam<int>("i_board_socket_id"))));
        ptQ = device->getOutputQueue(ptQName, ph->getParam<int>("i_max_q_size"), false);
        imageConverterPt = std::make_unique<dai::ros::ImageConverter>(tfPrefix, false);
        infoManager = std::make_shared<camera_info_manager::CameraInfoManager>(
            getROSNode()->create_sub_node(std::string(getROSNode()->get_name()) + "/" + getName()).get(), "/" + getName());
        infoManager->setCameraInfo(sensor_helpers::getCalibInfo(getROSNode()->get_logger(),
                                                                imageConverterPt,
                                                                device,
                                                                dai::CameraBoardSocket::CAM_A,
                                                                imageManip->initialConfig.getResizeWidth(),
                                                                imageManip->initialConfig.getResizeWidth()));

        ptPub = image_transport::create_camera_publisher(getROSNode().get(), "~/" + getName() + "/passthrough/image_raw");
        ptQ->addCallback(std::bind(sensor_helpers::basicCameraPub, std::placeholders::_1, std::placeholders::_2, *imageConverterPt, ptPub, infoManager));
    }
}

void RoadSegmentation::closeQueues() {
    nnQ->close();
    if(ph->getParam<bool>("i_enable_passthrough")) {
        ptQ->close();
    }
}

// void RoadSegmentation::process_frame(std::vector<float>& nn_output, cv::Mat &mask, cv::Mat &entropy, int img_width, int img_height)
//     {
//         // Create cv::Mat for each channel (reshape the flat vector)
//         std::vector<cv::Mat> channels(3);
//         for (int c = 0; c < 3; ++c) {
//             channels[c] = cv::Mat(img_height, img_width, CV_32F,
//                                 nn_output.data() + c * img_height * img_width);
//         }

//         // Prepare result matrix
//         cv::Mat argmax_mat(img_height, img_width, CV_8U, cv::Scalar(0));

//         // Do argmax manually
//         for (int y = 0; y < img_height; ++y) {
//             for (int x = 0; x < img_width; ++x) {
//                 float max_val = channels[0].at<float>(y, x);
//                 int max_idx = 0;
//                 for (int c = 1; c < 3; ++c) {
//                     float val = channels[c].at<float>(y, x);
//                     if (val > max_val) {
//                         max_val = val;
//                         max_idx = c;
//                     }
//                 }
//                 argmax_mat.at<uchar>(y, x) = static_cast<uchar>(max_idx);
//             }
//         }

//         // Define custom colors (BGR)
//         cv::Vec3b colors[3];
//         colors[0] = cv::Vec3b(0, 0, 0);          // Background
//         colors[1] = cv::Vec3b(128, 64, 128);     // Road or Sidewalk
//         colors[2] = cv::Vec3b(235, 206, 135);    // Sky (convert RGB->BGR)

//         // Create color output
//         for (int y = 0; y < img_height; ++y) {
//             for (int x = 0; x < img_width; ++x) {
//                 uchar cls = argmax_mat.at<uchar>(y, x);
//                 mask.at<cv::Vec3b>(y, x) = colors[cls];
//             }
//         }

//         cv::Mat entropy_map = compute_normalized_entropy(channels);
//         entropy_map.convertTo(entropy, CV_8U, 255.0);
//     }



void RoadSegmentation::process_frame(std::vector<float>& nn_output, cv::Mat &mask, cv::Mat &entropy, cv::Mat &cost, int img_width, int img_height)
{
    //RCLCPP_WARN(getLogger(), "process_frame called");
    //RCLCPP_WARN(getLogger(), "nn_output size: %zu", nn_output.size());
    //RCLCPP_WARN(getLogger(), "img_width: %d, img_height: %d", img_width, img_height);

    // Print first few elements of nn_output
    // std::string nn_preview = "";
    // for(int i = 0; i < std::min<size_t>(10, nn_output.size()); ++i) {
        // nn_preview += std::to_string(nn_output[i]) + " ";
    // }
    //RCLCPP_WARN(getLogger(), "First 10 elements of nn_output: %s", nn_preview.c_str());

    // Create cv::Mat for each channel
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(img_height, img_width, CV_32F, nn_output.data() + c * img_height * img_width);
        //RCLCPP_WARN(getLogger(), "Channel %d dimensions: %dx%d", c, channels[c].rows, channels[c].cols);

        // Print first row of each channel
        // std::string row_preview = "";
        // for (int x = 0; x < std::min(10, img_width); ++x) {
            // row_preview += std::to_string(channels[c].at<float>(0, x)) + " ";
        // }
        //RCLCPP_WARN(getLogger(), "Channel %d first row (first 10 elements): %s", c, row_preview.c_str());
    }

    // Prepare result matrix for argmax
    cv::Mat argmax_mat(img_height, img_width, CV_8U, cv::Scalar(0));
    //RCLCPP_WARN(getLogger(), "Argmax matrix initialized: %dx%d", argmax_mat.rows, argmax_mat.cols);

    // Do argmax manually
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            float max_val = channels[0].at<float>(y, x);
            int max_idx = 0;
            for (int c = 1; c < 3; ++c) {
                float val = channels[c].at<float>(y, x);
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            argmax_mat.at<uchar>(y, x) = static_cast<uchar>(max_idx);
        }
    }
    //RCLCPP_WARN(getLogger(), "Argmax computation done");

    // Print first row of argmax matrix
    // std::string argmax_row_preview = "";
    // for(int x = 0; x < std::min(10, img_width); ++x) {
        // argmax_row_preview += std::to_string(argmax_mat.at<uchar>(0, x)) + " ";
    // }
    //RCLCPP_WARN(getLogger(), "Argmax first row (first 10 elements): %s", argmax_row_preview.c_str());

    // Define colors (BGR)
    cv::Vec3b colors[3];
    colors[0] = cv::Vec3b(0, 0, 0);
    colors[1] = cv::Vec3b(128, 64, 128);
    colors[2] = cv::Vec3b(235, 206, 135);

    // Apply colors to mask
    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            uchar cls = argmax_mat.at<uchar>(y, x);
            mask.at<cv::Vec3b>(y, x) = colors[cls];
        }
    }
    //RCLCPP_WARN(getLogger(), "Mask creation done. Mask size: %dx%d, type: %d", mask.rows, mask.cols, mask.type());

    // Print first pixel of mask
    // cv::Vec3b first_pixel = mask.at<cv::Vec3b>(0,0);
    //RCLCPP_WARN(getLogger(), "First pixel of mask: B=%d G=%d R=%d", first_pixel[0], first_pixel[1], first_pixel[2]);

    // Compute entropy
    cv::Mat entropy_map = compute_normalized_entropy(channels);
    //RCLCPP_WARN(getLogger(), "Entropy map computed: %dx%d, type: %d", entropy_map.rows, entropy_map.cols, entropy_map.type());

    // Convert to 8-bit
    entropy_map.convertTo(entropy, CV_8U, 255.0);
    //RCLCPP_WARN(getLogger(), "Entropy conversion done. Entropy size: %dx%d, type: %d", entropy.rows, entropy.cols, entropy.type());

    for (int y = 0; y < img_height; ++y) {
        for (int x = 0; x < img_width; ++x) {
            uchar cls = argmax_mat.at<uchar>(y, x);
            if (entropy_map.at<float>(y, x) > normalizedEntropyThreshold)
                cls = static_cast<uchar>(SegmentationClass::BACKGROUND);

            cost.at<float>(y, x) = classCosts[static_cast<size_t>(cls)];
        }
    }

    // Print first row of entropy
    // std::string entropy_row_preview = "";
    // for(int x = 0; x < std::min(10, img_width); ++x) {
    //     entropy_row_preview += std::to_string(entropy.at<uchar>(0,x)) + " ";
    // }
    //RCLCPP_WARN(getLogger(), "Entropy first row (first 10 elements): %s", entropy_row_preview.c_str());
}





void RoadSegmentation::segmentationCB(const std::string& /*name*/, const std::shared_ptr<dai::ADatatype>& data) {

    //RCLCPP_WARN(getLogger(), "segmentationCB called");

    // Cast input data
    auto in_det = std::dynamic_pointer_cast<dai::NNData>(data);
    if (!in_det) {
        RCLCPP_WARN(getLogger(), "Failed to cast data to dai::NNData");
        return;
    } else {
        //RCLCPP_WARN(getLogger(), "Successfully casted data to dai::NNData");
    }

    // Get first layer FP16 data
    std::vector<float> nn_frame = in_det->getFirstLayerFp16();
    //RCLCPP_WARN(getLogger(), "nn_frame size: %zu", nn_frame.size());

    // Get image dimensions
    int img_width = imageManip->initialConfig.getResizeWidth();
    int img_height = imageManip->initialConfig.getResizeHeight();

    // Create mask and entropy matrices
    cv::Mat mask(img_height, img_width, CV_8UC3);
    cv::Mat entropy(img_height, img_width, CV_8UC1);
    cv::Mat cost(img_height, img_width, CV_32FC1);

    // Process frame
    process_frame(nn_frame, mask, entropy, cost, img_width, img_height);
    //RCLCPP_WARN(getLogger(), "Processed frame");

    // Prepare cv_bridge images
    cv_bridge::CvImage imgBridge_mask, imgBridge_entropy, imgBridge_cost;
    sensor_msgs::msg::Image img_msg_mask, img_msg_entropy, img_msg_cost;



    // // Prepare header
    // auto tfPrefix = getOpticalTFPrefix(getSocketName(static_cast<dai::CameraBoardSocket>(ph->getParam<int>("i_board_socket_id"))));
    
    // // CAMERA INFO
    // dai::rosBridge::ImageConverter converter(tfPrefix, true);
    // auto calibrationHandler = device.readCalibration();
    // auto cameraInfo = converter.calibrationToCameraInfo(calibrationHandler, ph->getParam<int>("i_board_socket_id"), ph->getParam<int>("i_width"), ph->getParam<int>("i_height"));

    dai::ros::updateBaseTime(steadyBaseTime, rosBaseTime, totalNsChange);

    std_msgs::msg::Header header;
    header.stamp = dai::ros::getFrameTime(rosBaseTime, steadyBaseTime, in_det->getTimestamp());
    header.frame_id = frame;

    nnInfo.header = header;

    // Convert to ROS Image messages
    imgBridge_mask = cv_bridge::CvImage(header, sensor_msgs::image_encodings::BGR8, mask);
    imgBridge_mask.toImageMsg(img_msg_mask);

    imgBridge_entropy = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO8, entropy);
    imgBridge_entropy.toImageMsg(img_msg_entropy);

    imgBridge_cost = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, cost);
    imgBridge_cost.toImageMsg(img_msg_cost);

    // Publish
    nnPub_mask.publish(img_msg_mask, nnInfo);
    nnPub_entropy.publish(img_msg_entropy, nnInfo);
    nnPub_cost.publish(img_msg_cost, nnInfo);
}

// cv::Mat RoadSegmentation::decodeDeeplab(cv::Mat mat) {
//     cv::Mat out = mat.mul(255 / 21);
//     out.convertTo(out, CV_8UC1);
//     cv::Mat colors = cv::Mat(256, 1, CV_8UC3);
//     cv::applyColorMap(out, colors, cv::COLORMAP_JET);
//     for(int row = 0; row < out.rows; ++row) {
//         uchar* p = out.ptr(row);
//         for(int col = 0; col < out.cols; ++col) {
//             if(*p++ == 0) {
//                 colors.at<cv::Vec3b>(row, col)[0] = 0;
//                 colors.at<cv::Vec3b>(row, col)[1] = 0;
//                 colors.at<cv::Vec3b>(row, col)[2] = 0;
//             }
//         }
//     }
//     return colors;
// }
void RoadSegmentation::link(dai::Node::Input in, int /*linkType*/) {
    segNode->out.link(in);
}

dai::Node::Input RoadSegmentation::getInput(int /*linkType*/) {
    if(ph->getParam<bool>("i_disable_resize")) {
        return segNode->input;
    }
    return imageManip->inputImage;
}

void RoadSegmentation::updateParams(const std::vector<rclcpp::Parameter>& params) {
    ph->setRuntimeParams(params);
    for (const auto& p : params)
    {
        if (p.get_name() == "normalized_entropy_threshold")
            normalizedEntropyThreshold = p.as_double();
    }
}
}  // namespace nn
}  // namespace dai_nodes
}  // namespace depthai_ros_driver
