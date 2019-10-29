#include "v9_ball_detector/v9_ball_detector.h"

#define FROM_VIDEO 0

const float BallDetector::MIN_CONTOUR_AREA = 100.0f;//100 -> Lomba ; 200 -> Bengkel
const float BallDetector::MIN_FIELD_CONTOUR_AREA = 1600.0f;

BallDetector::BallDetector():
    nh_(ros::this_node::getName()),
    it_(this->nh_),
    it_subs_(it_.subscribe("image_in", 1, &BallDetector::imageCallback, this)),
    it_pubs_(it_.advertise("image_out", 100)),
    cam_info_sub_(nh_.subscribe("camera_info_in", 100, &BallDetector::cameraInfoCallback, this)),
    cam_info_pub_(nh_.advertise<sensor_msgs::CameraInfo>("camera_info_out", 100)),
    frame_mode_subs_(nh_.subscribe("frame_mode", 1, &BallDetector::frameModeCallback, this)),
    save_param_subs_(nh_.subscribe("save_param", 1, &BallDetector::saveParamCallback, this)),
    LUT_sub_(nh_.subscribe("LUT_data", 1, &BallDetector::lutCallback, this)),
    it_bf_sub_(it_.subscribe("ball_ref", 1, &BallDetector::ballRefCallback, this)),
    it_sw_pub_(it_.advertise("segment_white", 10)),
    it_inv_sg_pub_(it_.advertise("inv_segment_green", 10)),
    line_tip_sub_(nh_.subscribe("line_tip",1,&BallDetector::lineTipCallback, this)),
    field_boundary_pub_(nh_.advertise<vision_utils::FieldBoundary > ("field_boundary", 10)),
    ball_pos_pub_(nh_.advertise<geometry_msgs::Point > ("ball_pos", 100)),
    update_params_pub_(nh_.advertise<std_msgs::Empty > ("update_params", 10)),
    pred_status_sub_(nh_.subscribe("/alfarobi/prediction_status", 1, &BallDetector::predStatusCb, this)),
    frame_mode_(0),
    pred_status_(false){

    nh_.param<std::string>("ball_config_path", ball_config_path,
                           ros::package::getPath("v9_ball_detector") + "/config/saved_config.yaml");

    param_cb_ = boost::bind(&BallDetector::paramCallback, this, _1, _2);
    param_server_.setCallback(param_cb_);

    LUT_dir = ros::package::getPath("v9_ball_detector") + "/config/tabel_warna.xml";

    loadParam();
}

void BallDetector::loadParam(){
    YAML::Node config_file;
    try{
        config_file = YAML::LoadFile(ball_config_path.c_str());
    }catch(const std::exception &e){
        ROS_ERROR("[v9_ball_detector] Unable to open config file: %s", e.what());
    }

    config_.score = config_file["score"].as<int>();
    config_.cost = config_file["cost"].as<int>();

    cv::FileStorage fs(LUT_dir.c_str(),cv::FileStorage::READ);
    fs["Tabel_Warna"] >> LUT_data;
    fs.release();

    ball_ref_ = cv::imread(ros::package::getPath("v9_ball_detector") + "/config/ball_ref.jpg");
    ball_ref_ = cvtMulti(ball_ref_);
    if(!ball_ref_.empty()){
        cv::calcHist(&ball_ref_, 1, hist_param_.channels, cv::Mat(), ball_ref_hist_, 2, hist_param_.hist_size, hist_param_.ranges);
        cv::normalize(ball_ref_hist_,ball_ref_hist_, .0, 1.0, cv::NORM_MINMAX);
    }
}

void BallDetector::saveParam(){
    YAML::Emitter yaml_out;
    yaml_out << YAML::BeginMap;
    yaml_out << YAML::Key << "score" << YAML::Value << config_.score;
    yaml_out << YAML::Key << "cost" << YAML::Value << config_.cost;
    yaml_out << YAML::EndMap;
    std::ofstream file_out(ball_config_path.c_str());
    file_out << yaml_out.c_str();
    file_out.close();
    cv::FileStorage fs(LUT_dir.c_str(), cv::FileStorage::WRITE);
    fs << "Tabel_Warna" << LUT_data;
    fs.release();

    cv::imwrite(ros::package::getPath("v9_ball_detector") + "/config/ball_ref.jpg", ball_ref_);
}

void BallDetector::frameModeCallback(const std_msgs::Int8::ConstPtr &_msg){
    frame_mode_ = _msg->data;
}

void BallDetector::saveParamCallback(const std_msgs::Empty::ConstPtr &_msg){
    (void)_msg;
    saveParam();
}

void BallDetector::ballRefCallback(const sensor_msgs::ImageConstPtr &_msg){
    cv_bf_ptr_sub_ = cv_bridge::toCvCopy(_msg,_msg->encoding);
    ball_ref_ = cv_bf_ptr_sub_->image;
    ball_ref_ = cvtMulti(ball_ref_);
    cv::calcHist(&ball_ref_, 1, hist_param_.channels, cv::Mat(), ball_ref_hist_, 2, hist_param_.hist_size, hist_param_.ranges);
    cv::normalize(ball_ref_hist_, ball_ref_hist_, .0, 1.0 , cv::NORM_MINMAX);
}

void BallDetector::lutCallback(const vision_utils::LUTConstPtr &_msg){
//    uchar* LUT_ptr = LUT_data.data;
    for(size_t i = 0; i < _msg->color.size(); i++){
        int h = (int)_msg->color[i].x;
        int s = (int)_msg->color[i].y;
        LUT_data.at<uchar>(h,s) = (int) _msg->color_class.data;
//        std::cout << h << " , " << s << " ; " << (int) _msg->color_class.data << std::endl;
//        LUT_ptr[s + h*256] = (int) _msg->color[i].z;
    }
//    cv::Mat diff = tempor != LUT_data;
//    if(cv::countNonZero(diff)==0)std::cout << "SAMA" << std::endl;
}

void BallDetector::lineTipCallback(const vision_utils::LineTipConstPtr &_msg){
    line_tip_.tip1 = _msg->tip1;
    line_tip_.tip2 = _msg->tip2;
}

void BallDetector::imageCallback(const sensor_msgs::ImageConstPtr &_msg){

    try{
        img_encoding_ = Alfarobi::GRAY8Bit;
        if(_msg->encoding.compare(sensor_msgs::image_encodings::MONO8))
            img_encoding_ = Alfarobi::GRAY8Bit;
#if FROM_VIDEO == 0
        if(_msg->encoding.compare(sensor_msgs::image_encodings::BGR8))
            img_encoding_ = Alfarobi::BGR8Bit;
#else
        if(_msg->encoding.compare(sensor_msgs::image_encodings::RGB8))
            img_encoding_ = Alfarobi::BGR8Bit;
#endif
    }catch(cv_bridge::Exception &e){
        ROS_ERROR("[v9_ball_detector] cv bridge exception: %s",e.what());
        return;
    }

    cv_img_ptr_subs_ = cv_bridge::toCvCopy(_msg,_msg->encoding);
    this->stamp_ = _msg->header.stamp;
    this->frame_id_ = _msg->header.frame_id;
}

void BallDetector::cameraInfoCallback(const sensor_msgs::CameraInfo &_msg){
    //cam_info_msg_ = *_msg;
    
//    ROS_INFO("CHECK...");
}

void BallDetector::paramCallback(v9_ball_detector::BallDetectorParamsConfig &_config, uint32_t level){
    (void)level;
    this->config_ = _config;
}

void BallDetector::publishImage(){
    cv_img_pubs_.image = out_img_.clone();

    //Stamp
    cv_img_pubs_.header.seq++;
    cv_img_pubs_.header.stamp = this->stamp_;
    cv_img_pubs_.header.frame_id = this->frame_id_;

    //microsoft lifecam brightness setting only work when the camera is capturing
    //setting first to zero brightness after first 2 frame then set to desired value
    //3 April 2019
    if(cv_img_pubs_.header.seq == 2){
        std_msgs::Empty empty_msg;
        update_params_pub_.publish(empty_msg);
    }else if(cv_img_pubs_.header.seq == 4){
        std_msgs::Empty empty_msg;
        update_params_pub_.publish(empty_msg);
    }

    switch(img_encoding_){
        case Alfarobi::GRAY8Bit:cv_img_pubs_.encoding = sensor_msgs::image_encodings::MONO8;break;
        case Alfarobi::BGR8Bit:cv_img_pubs_.encoding = sensor_msgs::image_encodings::RGB8;break;
        default:cv_img_pubs_.encoding = sensor_msgs::image_encodings::RGB8;break;
    }

    it_pubs_.publish(cv_img_pubs_.toImageMsg());
    cam_info_pub_.publish(cam_info_msg_);
}

void BallDetector::publishLocalizationUtils(const cv::Mat &_segmented_white,
                                         const cv::Mat &_inv_segmented_green,
                                         vision_utils::FieldBoundary _field_boundary){
    cv_sw_pub_.image = _segmented_white.clone();
    cv_inv_sg_pub_.image = _inv_segmented_green.clone();

    cv_sw_pub_.header.seq++;
    cv_inv_sg_pub_.header.seq++;
    _field_boundary.header.seq++;

    cv_sw_pub_.header.stamp = this->stamp_;
    cv_inv_sg_pub_.header.stamp = this->stamp_;
    _field_boundary.header.stamp = this->stamp_;

    cv_sw_pub_.header.frame_id = this->frame_id_;
    cv_inv_sg_pub_.header.frame_id = this->frame_id_;
    _field_boundary.header.frame_id = this->frame_id_;

    cv_sw_pub_.encoding = sensor_msgs::image_encodings::MONO8;
    cv_inv_sg_pub_.encoding = sensor_msgs::image_encodings::MONO8;

    it_sw_pub_.publish(cv_sw_pub_.toImageMsg());
    it_inv_sg_pub_.publish(cv_inv_sg_pub_.toImageMsg());
    field_boundary_pub_.publish(_field_boundary);
}

cv::Mat& BallDetector::setInputImage(){
    return in_img_;
}

void BallDetector::setOutputImage(const cv::Mat &_out_img){
    out_img_ = _out_img.clone();
}

std::pair<cv::Mat, vision_utils::FieldBoundary > BallDetector::getFieldImage(const cv::Mat &_segmented_green){
    cv::Mat _field_contour = cv::Mat::zeros(_segmented_green.size(), CV_8UC1);
    vision_utils::FieldBoundary field_boundary;
    Points contour_points;
    std::vector<Points > contours;
    std::vector<cv::Vec4i > hierarchy;

    cv::findContours(_segmented_green, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    for(size_t i = 0; i < contours.size(); i++){
        if(cv::contourArea(contours[i]) > MIN_FIELD_CONTOUR_AREA){
            contour_points.insert(contour_points.end(), contours[i].begin(), contours[i].end());
        }
    }

    if(contour_points.size()){
        std::vector<Points > contour(1);
        cv::convexHull(contour_points,contour.front());
        cv::Rect field_bound = cv::boundingRect(contour.front());
        drawContours(_field_contour, contour, 0, cv::Scalar(255), CV_FILLED);
        //[HW] Scan from dual direction
        for(int i = field_bound.tl().x;
            i < field_bound.br().x; i++){
            geometry_msgs::Vector3 temp;
            temp.x = i;
            temp.y = -1;
            temp.z = field_bound.br().y - 1;
            for(int j = field_bound.tl().y;
                j < field_bound.br().y; j++){
                if(_field_contour.at<uchar >(j, i) > 0 &&
                        temp.y == -1){
                    temp.y = j;
                }else if(_field_contour.at<uchar >(j, i) == 0 &&
                         temp.y != -1){
                    temp.z = j - 1;
                    break;
                }
            }
            field_boundary.bound1.push_back(temp);
        }

        for(int i = field_bound.tl().y;
            i < field_bound.br().y; i++){
            geometry_msgs::Vector3 temp;
            temp.x = i;
            temp.y = -1;
            temp.z = field_bound.br().x - 1;
            for(int j = field_bound.tl().x;
                j < field_bound.br().x; j++){
                if(_field_contour.at<uchar >(i, j) > 0 &&
                        temp.y == -1){
                    temp.y = j;
                }else if(_field_contour.at<uchar >(i, j) == 0 &&
                         temp.y != -1){
                    temp.z = j - 1;
                    break;
                }
            }
            field_boundary.bound2.push_back(temp);
        }
    }

    std::pair<cv::Mat, vision_utils::FieldBoundary > result;
    result.first = _field_contour;
    result.second = field_boundary;
    return result;
}

void BallDetector::localizationInputEnhance(cv::Mat &_input){
    cv::Mat result = _input.clone();
    cv::dilate(result, result ,cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)), cv::Point(), 1);

    std::vector<Points > contours;
    std::vector<cv::Vec4i > hierarchy;
    std::vector<double > contours_area;

    cv::findContours(result, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

    
    result = cv::Mat::zeros(result.size(),CV_8UC1);

    for(size_t i=0;i<contours.size();i++){
        contours_area.emplace_back(cv::contourArea(contours[i]));
        if(contours_area[i] > MIN_CONTOUR_AREA && hierarchy[i][3] == -1){
            Points approx_curve;
            cv::approxPolyDP(contours[i], approx_curve, 0.002*cv::arcLength(contours[i],true),true);
            std::vector<Points > target_contour;
            target_contour.push_back(approx_curve);
            drawContours(result, target_contour, 0, cv::Scalar(255), CV_FILLED);
        }
    }

    for(size_t i=0;i<contours.size();i++){
        if(contours_area[i] > MIN_CONTOUR_AREA && hierarchy[i][3] > -1){
            Points approx_curve;
            cv::approxPolyDP(contours[i], approx_curve, 0.002*cv::arcLength(contours[i],true),true);
            std::vector<Points > target_contour;
            target_contour.push_back(approx_curve);
            drawContours(result, target_contour, 0, cv::Scalar(0), CV_FILLED);
        }
    }
    _input = result.clone();
}

cv::Mat BallDetector::segmentColor(cv::Mat &_segmented_green, cv::Mat &_inv_segmented_green,
                                    cv::Mat &_segmented_ball_color, cv::Mat &_segmented_white){

    cv::Mat blank = cv::Mat::zeros(Alfarobi::FRAME_HEIGHT, Alfarobi::FRAME_WIDTH, CV_8UC1);
    cv::Mat out_segment = cv::Mat::zeros(Alfarobi::FRAME_HEIGHT, Alfarobi::FRAME_WIDTH, CV_8UC3);

    cv::Mat segmented_green = blank.clone();
    cv::Mat segmented_ball_color = blank.clone();
    cv::Mat segmented_white = blank.clone();

    cv::cvtColor(in_img_,in_hsv_,CV_BGR2HSV);

    /*cv::Mat gray;
    cv::cvtColor(in_img_,gray,CV_BGR2GRAY);
    cv::medianBlur(gray,gray,3);
    cv::Mat kernel = (cv::Mat_<double>(3,3) << 0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111,0.111111111);
    cv::filter2D(gray,gray,CV_8UC1,kernel);
    cv::Mat lutable = cv::Mat(1,256,CV_8U);
    uchar *lutable_ptr = lutable.ptr();
    for(int i=0;i<256;i++){
        lutable_ptr[i] = pow(i/255.0,0.2)*255.0;
    }
    cv::Mat resulttt = gray.clone();
    cv::LUT(gray,lutable,resulttt);
    cv::equalizeHist(gray,gray);
    cv::adaptiveThreshold(gray,gray,255,cv::ADAPTIVE_THRESH_MEAN_C,cv::THRESH_BINARY,11,1);
    cv::imshow("GRAY",gray);*/

    int num_cols = Alfarobi::FRAME_WIDTH;
    int num_rows = Alfarobi::FRAME_HEIGHT;

    // auto LUT_ptr = LUT_data.data;
    for(int i = 0; i < num_rows; i++){
        cv::Vec3b* in_hsv_ptr = in_hsv_.ptr<cv::Vec3b>(i);
        cv::Vec3b* out_segment_ptr = out_segment.ptr<cv::Vec3b>(i);
        uchar* sg_ptr = segmented_green.ptr<uchar>(i);
        uchar* sbc_ptr = segmented_ball_color.ptr<uchar>(i);
        uchar* sw_ptr = segmented_white.ptr<uchar>(i);
        for(int j = 0; j < num_cols; j++){
//            std::cout << i << " , " << j << " ; " << (int)in_hsv_ptr[j][0] << " , " << (int)in_hsv_ptr[j][1] << std::endl;
            // if(LUT_data.at<uchar>(in_hsv_ptr[j][0], in_hsv_ptr[j][1]) == 1){
            // uchar pres_class = LUT_ptr[in_hsv_ptr[j][1] + in_hsv_ptr[j][0]*num_cols];
            uchar pres_class = LUT_data.at<uchar>(in_hsv_ptr[j][0], in_hsv_ptr[j][1]);
            if(pres_class == 1){
                sg_ptr[j] = 255;
                out_segment_ptr[j][0] = 0;
                out_segment_ptr[j][1] = 200;
                out_segment_ptr[j][2] = 0;                
            }else if(pres_class == 2){
                sbc_ptr[j] = 255;
                out_segment_ptr[j][0] = 0;
                out_segment_ptr[j][1] = 140;
                out_segment_ptr[j][2] = 255;
            }else if(pres_class == 3){
                sw_ptr[j] = 255;
                out_segment_ptr[j][0] = 255;
                out_segment_ptr[j][1] = 255;
                out_segment_ptr[j][2] = 255;
            }
        }
    }

    cv::Mat inv_segmented_green;
    cv::bitwise_not(segmented_green,inv_segmented_green);

    localizationInputEnhance(segmented_white);
    localizationInputEnhance(inv_segmented_green);

    _segmented_green = segmented_green.clone();
    _inv_segmented_green = inv_segmented_green.clone();
    _segmented_ball_color = segmented_ball_color.clone();
    _segmented_white = segmented_white.clone();

    return out_segment;
}

void BallDetector::filterContourData(std::vector<cv::Mat> &divided_roi, cv::Point top_left_pt,
                       std::vector<Points > &selected_data, cv::Mat *debug_mat, int sub_mode = 0){
    int num_roi_cols = divided_roi.front().cols;
    int num_roi_rows = divided_roi.front().rows;
    bool horizon_scan = (float)num_roi_rows/(float)num_roi_cols < .75f;
    cv::Point map_origin[4];
    map_origin[0].x = top_left_pt.x;
    map_origin[0].y = top_left_pt.y;
    map_origin[1].x = (sub_mode == 2)?top_left_pt.x:top_left_pt.x + divided_roi.front().cols;
    map_origin[1].y = (sub_mode == 2)?top_left_pt.y + divided_roi.front().rows:top_left_pt.y;
    map_origin[2].x = top_left_pt.x;
    map_origin[2].y = top_left_pt.y + num_roi_rows;
    map_origin[3].x = top_left_pt.x + num_roi_cols;
    map_origin[3].y = top_left_pt.y + num_roi_rows;
    for(size_t idx = 0; idx < divided_roi.size(); idx++){

        int scan_mode = idx;

        switch(idx){
        case 0:scan_mode = (sub_mode == 1) ? 0 : (sub_mode == 2) ? 2 : horizon_scan ? 0 : 2;break;
        case 1:scan_mode = (sub_mode == 1) ? 1 : (sub_mode == 2) ? 3 : horizon_scan ? 1 : 2;break;
        case 2:scan_mode = horizon_scan ? 0 : 3;break;
        case 3:scan_mode = horizon_scan ? 1 : 3;break;
        }

        switch(scan_mode){
        case 0:{
            for(int i=0;i<num_roi_rows;i++){
                for(int j=0;j<num_roi_cols;j++){
                    if(divided_roi[idx].at<uchar>(i,j) == 255){
                        if(j==0)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + j;
                        selected_point.y = map_origin[idx].y + i;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }break;
        case 1:{
            for(int i=0;i<num_roi_rows;i++){
                for(int j=num_roi_cols-1;j>=0;j--){
                    if(divided_roi[idx].at<uchar>(i,j) == 255){
                        if(j==num_roi_cols-1)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + j;
                        selected_point.y = map_origin[idx].y + i;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }break;
        case 2:{
            for(int i=0;i<num_roi_cols;i++){
                for(int j=0;j<num_roi_rows;j++){
                    if(divided_roi[idx].at<uchar>(j,i) == 255){
                        if(j==0)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + i;
                        selected_point.y = map_origin[idx].y + j;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(j,i) = 255;
                        break;
                    }
                }
            }
        }break;
        case 3:{
            for(int i=0;i<num_roi_cols;i++){
                for(int j=num_roi_rows-1;j>=0;j--){
                    if(divided_roi[idx].at<uchar>(j,i) == 255){
                        if(j==num_roi_rows-1)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + i;
                        selected_point.y = map_origin[idx].y + j;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(j,i) = 255;
                        break;
                    }
                }
            }
        }break;

        }
    }
}

cv::Mat BallDetector::cvtMulti(const cv::Mat &_ball_ref){
    //cv::Mat hsv;
    cv::Mat yuv;
    //cv::Mat ycrcb;

//    cv::cvtColor(_ball_ref,hsv,CV_BGR2HSV);
    cv::cvtColor(_ball_ref,yuv,CV_BGR2YUV);
//    cv::cvtColor(_ball_ref,ycrcb,CV_BGR2YCrCb);
//    cv::Mat temp;
//    hsv.convertTo(hsv,CV_32F);
//    yuv.convertTo(yuv,CV_32F);
//    ycrcb.convertTo(ycrcb,CV_32F);
//    cv::multiply(hsv,yuv,temp);
//    cv::multiply(temp,ycrcb,temp);
//    cv::normalize(temp,temp,0,1,cv::NORM_MINMAX);
//    temp *=255;
//    temp.convertTo(temp,CV_8U);
    return yuv.clone();
}

float BallDetector::checkTargetHistogram(cv::Mat _target_roi){

    if(ball_ref_.empty()){
        ROS_ERROR("[v9_ball_detector] Ball reference not found !!!");
        return -1;
    }
    _target_roi = cvtMulti(_target_roi);
    cv::MatND target_hist;
    cv::calcHist(&_target_roi, 1, hist_param_.channels, cv::Mat(), target_hist, 2, hist_param_.hist_size, hist_param_.ranges);
    cv::normalize(target_hist,target_hist, .0, 1.0, cv::NORM_MINMAX);

    return cv::compareHist(ball_ref_hist_, target_hist, CV_COMP_KL_DIV);
}

std::vector<cv::Mat > BallDetector::getBallPosPrediction(const Points &_data){
    int total_smp = 0;
    int total_smp2 = 0;
    int total_smp3 = 0;
    int total_smp4 = 0;

    int total_x=0;
    int total_smpx=0;
    int total_smp2x=0;

    int total_y=0;
    int total_smpy=0;
    int total_smp2y=0;
    for(size_t i=0;i < _data.size();i++){

        total_smp += i;
        int smp2 = i*i;
        int smp3 = smp2*i;
        total_smp2 += smp2;
        total_smp3 += smp3;
        total_smp4 += smp3*i;

        total_x += _data[i].x;
        total_smpx += i*_data[i].x;
        total_smp2x += smp2*_data[i].x;

        total_y += _data[i].y;
        total_smpy += i*_data[i].y;
        total_smp2y += smp2*_data[i].y;

    }
    cv::Mat A = (cv::Mat_<double>(3,3) << total_smp4, total_smp3, total_smp2, total_smp3, total_smp2, total_smp, total_smp2, total_smp, _data.size());
    cv::Mat bx = (cv::Mat_<double>(3,1) << total_smp2x,total_smpx,total_x);
    cv::Mat by = (cv::Mat_<double>(3,1) << total_smp2y,total_smpy,total_y);
    cv::Mat A_inv = A.inv();
    cv::Mat xpoly_const = A_inv*bx;
    cv::Mat ypoly_const = A_inv*by;
    cv::Mat first_term = (cv::Mat_<double>(2,1) << xpoly_const.at<double>(0), ypoly_const.at<double>(0));
    cv::Mat second_term = (cv::Mat_<double>(2,1) << xpoly_const.at<double>(1), ypoly_const.at<double>(1));
    cv::Mat third_term = (cv::Mat_<double>(2,1) << xpoly_const.at<double>(2), ypoly_const.at<double>(2));
    std::vector<cv::Mat > result;
    result.push_back(first_term);
    result.push_back(second_term);
    result.push_back(third_term);

    return result;

}

void BallDetector::process(){
    if(cv_img_ptr_subs_ == nullptr)return;
    // auto t1 = boost::chrono::high_resolution_clock::now();
    static geometry_msgs::Point last_ball_pos_;
    setInputImage() = cv_img_ptr_subs_->image;
    // cv::medianBlur(in_img_,in_img_,3);
    // cv::GaussianBlur(in_img_,in_img_,cv::Size(3,3),0,0);
    // cv::GaussianBlur(in_img_,in_img_,cv::Size(3,3),0,0);
    // cv::GaussianBlur(in_img_,in_img_,cv::Size(3,3),0,0);
    cv::Mat output_view = in_img_.clone();
    cv::Mat segmented_green,segmented_ball_color,segmented_white, inv_segmented_green;
    cv::Mat thresh_image = segmentColor(segmented_green, inv_segmented_green, segmented_ball_color, segmented_white);

//    integralImage(in_hsv_);

//    cv::imshow("SBC",segmented_ball_color);
//    cv::imshow("SG",inv_segmented_green);
//    cv::imshow("SW",segmented_white);
//    cv::waitKey(1);

    cv::Mat field_contour;
    std::pair<cv::Mat, vision_utils::FieldBoundary > field_prop = getFieldImage(segmented_green);
    field_contour = field_prop.first;
    publishLocalizationUtils(segmented_white,inv_segmented_green,field_prop.second);

    cv::Mat ball_inside_field;
    cv::bitwise_and(segmented_ball_color,field_contour,ball_inside_field);

    std::vector<Points > contours;
    //Approx NONE to get the authentic contours
    cv::findContours(ball_inside_field, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    int count = 0;

    cv::Vec4f best_candidate(-1.0f, -1.0f , std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
    cv::Mat ROI;
    cv::Rect ball_roi;
    for(size_t i = 0; i < contours.size(); i++){
        float contour_area = cv::contourArea(contours[i]);
        if(contour_area > MIN_CONTOUR_AREA){
            cv::Rect rect_rough_roi = cv::boundingRect(contours[i]);

            cv::Mat roi_hsv_ = cv::Mat(in_hsv_,rect_rough_roi);

            float histogram_score = checkTargetHistogram(roi_hsv_);
#ifdef DEBUG
            std::cout << "=============================================" << std::endl;
            std::cout << count << ". HIST SCORE : " << histogram_score << std::endl;
            std::cout << "CONTOUR AREA : " << contour_area << std::endl;
#endif
            if(histogram_score < (float)config_.score / 10.0f)continue;

//            cv::rectangle(output_view,rect_rough_roi,cv::Scalar(255,255,255),2);

            cv::Point tl_pt = rect_rough_roi.tl();
//            cv::Point br_pt = rect_rough_roi.br();

            cv::Mat frame_rough_roi(ball_inside_field, rect_rough_roi);

            float roi_ratio = (rect_rough_roi.width < rect_rough_roi.height)?(float)rect_rough_roi.width/(float)rect_rough_roi.height:
                                                                             (float)rect_rough_roi.height/(float)rect_rough_roi.width;

            cv::Vec4f circle_param(-1.0f, -1.0f , std::numeric_limits<float>::max(), std::numeric_limits<float>::max());

            std::vector<cv::Mat> sub_frame;
#ifdef DEBUG
            std::cout << "ROI Ratio : " << roi_ratio << " ; Recip : " << 1.0f/roi_ratio << std::endl;
#endif
            if(roi_ratio >= 0.55f && 1.0f/roi_ratio <= 1.45f){
                if(contour_area < 5000){
                    Points outer_circle;
                    cv::convexHull(contours[i],outer_circle);
                    cv::Moments moment;
                    moment = cv::moments(outer_circle,true);
                    cv::Point ball_com(moment.m10/moment.m00, moment.m01/moment.m00);
                    cv::Point2f ctr;
                    float radius;
                    cv::minEnclosingCircle(outer_circle,ctr,radius);
                    cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(outer_circle, Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);
//                    sub_circle_param[3] /= (sub_circle_param[2]*sub_circle_param[2]);
                //    std::cout << sub_circle_param << std::endl;
                //     if(std::abs(ball_com.x - sub_circle_param[0]) >= 2)
                //         std::cout << "Param 1 : " << std::abs(ball_com.x - sub_circle_param[0]) << std::endl;
                //     if(std::abs(ball_com.y - sub_circle_param[1]) >= 2)
                //         std::cout << "Param 2 : " << std::abs(ball_com.y - sub_circle_param[1]) << std::endl;
                //     if(std::abs(sub_circle_param[0] - ctr.x) >= 2)
                //         std::cout << "Param 3 : " << std::abs(sub_circle_param[0] - ctr.x) << std::endl;
                //     if(std::abs(sub_circle_param[1] - ctr.y) >= 2)
                //         std::cout << "Param 4 : " << std::abs(sub_circle_param[1] - ctr.y) << std::endl;
                //     std::cout << "Contour2Circle : " << cv::contourArea(outer_circle)/(CV_PI*radius*radius) << std::endl;
                    if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < config_.cost*5 &&
                            std::fabs(ball_com.x - sub_circle_param[0]) <= 2 &&
                            std::fabs(ball_com.y - sub_circle_param[1]) <= 2 &&
                            std::fabs(sub_circle_param[0] - ctr.x) <= 2 &&
                            std::fabs(sub_circle_param[1] - ctr.y) <= 2 &&
                            cv::contourArea(outer_circle)/(Math::PI*radius*radius) > 0.75)
//                            sub_circle_param[0] > tl_pt.x && sub_circle_param[0] < br_pt.x &&
//                            sub_circle_param[1] > tl_pt.y && sub_circle_param[1] < br_pt.y)
                        circle_param=sub_circle_param;
                    
                }else{
                    sub_frame.resize(4);
                    sub_frame[0] = cv::Mat(frame_rough_roi,cv::Rect(0, 0, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1));
                    sub_frame[1] = cv::Mat(frame_rough_roi,cv::Rect(rect_rough_roi.width >> 1, 0, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1));
                    sub_frame[2] = cv::Mat(frame_rough_roi,cv::Rect(0, rect_rough_roi.height >> 1, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1));
                    sub_frame[3] = cv::Mat(frame_rough_roi,cv::Rect(rect_rough_roi.width >> 1, rect_rough_roi.height >> 1, rect_rough_roi.width >> 1, rect_rough_roi.height >> 1));

//                    cv::line(output_view,cv::Point(tl_pt.x + rect_rough_roi.width/2,tl_pt.y),cv::Point(tl_pt.x + rect_rough_roi.width/2,tl_pt.y + rect_rough_roi.height),cv::Scalar(255,0,0),2);
//                    cv::line(output_view,cv::Point(tl_pt.x,tl_pt.y+rect_rough_roi.height/2),cv::Point(tl_pt.x+rect_rough_roi.width,tl_pt.y+rect_rough_roi.height/2),cv::Scalar(255,0,0),2);

                    std::vector<Points > selected_data(4);

                    cv::Mat sub_sample[4];
                    sub_sample[0] = cv::Mat::zeros(sub_frame[0].size(), CV_8UC1);
                    sub_sample[1] = cv::Mat::zeros(sub_frame[1].size(), CV_8UC1);
                    sub_sample[2] = cv::Mat::zeros(sub_frame[2].size(), CV_8UC1);
                    sub_sample[3] = cv::Mat::zeros(sub_frame[3].size(), CV_8UC1);

                    filterContourData(sub_frame, tl_pt, selected_data, sub_sample, 0);

                    // cv::imshow("CEKK1",cek_aj[0]);
                    // cv::imshow("CEKK2",cek_aj[1]);
                    // cv::imshow("CEKK3",cek_aj[2]);
                    // cv::imshow("CEKK4",cek_aj[3]);

                    for(int j = 0; j < 4; j++){

                        cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(selected_data[j], Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);

//                        sub_circle_param[3] /= (sub_circle_param[2]*sub_circle_param[2]);
//                        int axis_dir = frame_rough_roi.cols > frame_rough_roi.rows?1:0;
//                        cv::Vec2f interval_ctr = axis_dir?cv::Vec2f(tl_pt.x, br_pt.x):cv::Vec2f(tl_pt.y,br_pt.y);
//                        std::cout << j << ". " << sub_circle_param << std::endl;
                        if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < config_.cost)
                                //sub_circle_param[axis_dir] > interval_ctr[0] && sub_circle_param[axis_dir] < interval_ctr[1])
//                                sub_circle_param[0] > 1.0*(float)tl_pt.x && sub_circle_param[0] < 1.0*(float)br_pt.x &&
//                                sub_circle_param[1] > 1.0*(float)tl_pt.y && sub_circle_param[1] < 1.0*(float)br_pt.y)
//                            sub_circle_param[2] > (float)std::min(sub_frame[0].cols,sub_frame[0].rows) &&
//                                    sub_circle_param[2] < 1.2*(float)std::max(sub_frame[0].cols,sub_frame[0].rows) )
                            circle_param=sub_circle_param;
                    }
                }

            }else if(roi_ratio > 0.2f){
                
                sub_frame.resize(2);
                int sub_mode=0;
                if(rect_rough_roi.width > rect_rough_roi.height){
                    sub_frame[0] = cv::Mat(frame_rough_roi, cv::Rect(0, 0, rect_rough_roi.width >> 1, rect_rough_roi.height));
                    sub_frame[1] = cv::Mat(frame_rough_roi, cv::Rect(rect_rough_roi.width >> 1, 0, rect_rough_roi.width >> 1, rect_rough_roi.height));
                    sub_mode=1;
//                    cv::line(output_view,cv::Point(tl_pt.x+rect_rough_roi.width/2,tl_pt.y),cv::Point(tl_pt.x+rect_rough_roi.width/2,tl_pt.y+rect_rough_roi.height),cv::Scalar(255,0,0),2);
                }else{
                    sub_frame[0] = cv::Mat(frame_rough_roi, cv::Rect(0, 0, rect_rough_roi.width, rect_rough_roi.height >> 1));
                    sub_frame[1] = cv::Mat(frame_rough_roi, cv::Rect(0, rect_rough_roi.height >> 1, rect_rough_roi.width, rect_rough_roi.height >> 1));
                    sub_mode=2;
//                    cv::line(output_view,cv::Point(tl_pt.x,tl_pt.y+rect_rough_roi.height/2),cv::Point(tl_pt.x+rect_rough_roi.width,tl_pt.y+rect_rough_roi.height/2),cv::Scalar(255,0,0),2);
                }
                cv::Mat sub_sample[4];
                sub_sample[0] = cv::Mat::zeros(sub_frame[0].size(), CV_8UC1);
                sub_sample[1] = cv::Mat::zeros(sub_frame[1].size(), CV_8UC1);
                std::vector<Points > selected_data(2);

                filterContourData(sub_frame,tl_pt, selected_data, sub_sample, sub_mode);
//                float area_ratio = contour_area/(rect_rough_roi.width*rect_rough_roi.height);
//                if(area_ratio > 0.7 && area_ratio < 0.8){

//                    selected_data[0].insert(selected_data[0].end(),selected_data[1].begin(), selected_data[1].end());
//                    selected_data.resize(1);
//                }

                for(size_t j = 0; j < selected_data.size(); j++){
                    cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(selected_data[j], Alfarobi::FIT_CIRCLE_MAX_STEPS, Alfarobi::FIT_CIRCLE_EPS);
//                    sub_circle_param[3] /= (sub_circle_param[2]*sub_circle_param[2]);
//                    int axis_dir = frame_rough_roi.cols > frame_rough_roi.rows?1:0;
//                    cv::Vec2f interval_ctr = axis_dir?cv::Vec2f(tl_pt.x, br_pt.x):cv::Vec2f(tl_pt.y,br_pt.y);
//                    std::cout << j << ". " << sub_circle_param << std::endl;
                    if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < config_.cost &&
                            sub_circle_param[2] > std::max(frame_rough_roi.cols,frame_rough_roi.rows) >> 2)
                            //sub_circle_param[axis_dir] > interval_ctr[0] && sub_circle_param[axis_dir] < interval_ctr[1])
//                            sub_circle_param[2] >= (float)std::max(sub_frame[0].cols,sub_frame[0].rows))
                        //                           2*sub_circle_param[2] <= 1.333*(float)std::max(sub_frame[0].cols,sub_frame[0].rows))
                        circle_param=sub_circle_param;
                }
            }

//            std::cout << " Cost : " << circle_param[3] << std::endl;
            float ball_percentage = (float)cv::countNonZero(frame_rough_roi)/(frame_rough_roi.cols*frame_rough_roi.rows);
#ifdef DEBUG
           std::cout << "Ball Percentage : " << ball_percentage << std::endl;
#endif
                                                        // Maximum Circle Radius
            // std::cout << circle_param << std::endl;
           constexpr float MAX_BALL_RAD = static_cast<float>(Alfarobi::FRAME_WIDTH >> 1);
        //    std::cout << "MAX BALL RAD : " << MAX_BALL_RAD << std::endl;
            if(circle_param[2] > .0f && circle_param[2] < MAX_BALL_RAD && ball_percentage > .35f){
                count++;

                float circle_radius = circle_param[2];
                int tl_roi_x = std::min(std::max(0,int(circle_param[0] - circle_radius)), output_view.cols-1);
                int tl_roi_y = std::min(std::max(0,int(circle_param[1] - circle_radius)), output_view.rows-1);
                int br_roi_x = std::min(std::max(0,int(circle_param[0] + circle_radius)), output_view.cols-1);
                int br_roi_y = std::min(std::max(0,int(circle_param[1] + circle_radius)), output_view.rows-1);

                cv::Rect roi_region(tl_roi_x,tl_roi_y, (br_roi_x - tl_roi_x), (br_roi_y - tl_roi_y));
                ball_roi = roi_region;
                cv::Mat green_percentage(segmented_green, rect_rough_roi);
                float green_percent = (float)cv::countNonZero(green_percentage)/(rect_rough_roi.area());
#ifdef DEBUG
                std::cout << "Green Percent : " << green_percent << std::endl;
#endif
                if(green_percent > .001f){ // Minimum 0.1% Green
                    if(circle_param[3] < best_candidate[3]){
                        best_candidate = circle_param;
                        ROI = cv::Mat(ball_inside_field,rect_rough_roi);
                    }
                }
            }
        }
    }
    static int next_idx=11;
    ball_pos_.x = best_candidate[0];
    ball_pos_.y = best_candidate[1];
    ball_pos_.z = best_candidate[2];
    if(best_candidate[0] > 0){
//        static int count_img = 0;

        next_idx=11;
        static int sample_count = 0;
        regression_data_.emplace_back(cv::Point(best_candidate[0],best_candidate[1]));
        sample_count++;
        if(sample_count >= 10){
            sample_count=0;
            est_trajectory_ = getBallPosPrediction(regression_data_);

            regression_data_.clear();
        }

//        std::stringstream file_name;
//        file_name << "/media/koseng/New Volume/temp4/" << frame_id_ << "_" << count_img << ".jpg";
//        cv::imwrite(file_name.str().c_str(),output_view);
//        count_img++;
    //    cv::imshow("ROI", ROI);
    //    cv::waitKey(0);
    //    cv::destroyAllWindows();
    }else if(next_idx < 16 && est_trajectory_.size() > 0){
        if(pred_status_){
            cv::Mat pred_pos = (next_idx*next_idx) * est_trajectory_[0] +
                                next_idx * est_trajectory_[1] +
                                est_trajectory_[2];
            ball_pos_.x = pred_pos.at<double>(0);
            ball_pos_.y = pred_pos.at<double>(1);            
        }else{
            ball_pos_ = last_ball_pos_;
        }
        next_idx++;
    }else{
        if(regression_data_.size() > 0)regression_data_.clear();
    }
    last_ball_pos_ = ball_pos_;
    ball_pos_pub_.publish(ball_pos_);

    //For purpose GUI only
//    cv::waitKey(1);
    switch(frame_mode_){// 144 - 408 microseconds
//    case 0:cvtColor(in_img_,in_img_,CV_BGR2RGB);setOutputImage(in_img_);break;
    case 1:setOutputImage(in_hsv_);break;
    case 2:setOutputImage(thresh_image);break;
    case 3:{
        cv::cvtColor(field_contour,field_contour,CV_GRAY2BGR);
        cv::Mat output_view_roi(output_view,ball_roi);
        cv::Mat ball_region(output_view_roi.size(),CV_8UC3,cv::Scalar(0,255,255));
        cv::addWeighted(output_view_roi, .5, ball_region, .5, .0,output_view_roi);
//        cv::circle(output_view,cv::Point(best_candidate[0],best_candidate[1]),best_candidate[2],cv::Scalar(0,255,255),2);

        for(int idx = 0; idx < 16 && est_trajectory_.size() > 0 &&
            regression_data_.size() > 0; idx++){
            cv::Mat pred_pos = (idx*idx) * est_trajectory_[0] +
                                idx * est_trajectory_[1] +
                                est_trajectory_[2];
            cv::Point pred_pos_pt(pred_pos.at<double>(0), pred_pos.at<double>(1));
            cv::circle(output_view, pred_pos_pt,
                       4, cv::Scalar(idx<10?255:0,idx<10?255:0,idx<10?255:0), CV_FILLED);
        }
        for(int i=0;i<line_tip_.tip1.size();i++){
            cv::Point tip1(line_tip_.tip1[i].x, line_tip_.tip1[i].y);
            cv::Point tip2(line_tip_.tip2[i].x, line_tip_.tip2[i].y);
            cv::line(output_view,cv::Point(line_tip_.tip1[i].x, line_tip_.tip1[i].y),
                     cv::Point(line_tip_.tip2[i].x, line_tip_.tip2[i].y), cv::Scalar(255,0,255), 3);
            cv::circle(output_view, tip1, 7, cv::Scalar(100,50,100), CV_FILLED);
            cv::circle(output_view, tip2, 7, cv::Scalar(100,50,100), CV_FILLED);
        }
        line_tip_.tip1.clear();
        line_tip_.tip2.clear();
        cv::bitwise_and(output_view,field_contour,output_view);
//        cvtColor(output_view,output_view,CV_BGR2RGB);
        setOutputImage(output_view);break;
    }
    default://cvtColor(in_img_,in_img_,CV_BGR2RGB);
        setOutputImage(in_img_);break;
    }
//    setOutputImage(in_img_);
    publishImage();
    // auto t2 = boost::chrono::high_resolution_clock::now();
    // auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();
    // std::cout << "Elapsed Time : " << elapsed_time << std::endl;
}

void BallDetector::integralImage(const cv::Mat &_input){
    cv::Mat ch1_ii = cv::Mat::zeros(Alfarobi::FRAME_HEIGHT, Alfarobi::FRAME_WIDTH, CV_32FC1);
    cv::Mat ch2_ii = ch1_ii.clone();
    cv::Mat ch3_ii = ch1_ii.clone();

    int bound_y{0}, bound_x{0};
    int prev_y{0}, prev_x{0};

    for(int i=0;i<_input.rows;i++){
        bound_y = (int)(i > 0);
        prev_y = bound_y ? i-1 : 0;
        for(int j=0;j<_input.cols;j++){
            bound_x = (int)(j > 0);
            prev_x = bound_x ? j-1 : 0;
            ch1_ii.at<float>(i,j) = _input.at<cv::Vec3b>(i,j)[0] +
                                    ch1_ii.at<float>(i,prev_x) * bound_x +
                                    ch1_ii.at<float>(prev_y,j) * bound_y;
            ch2_ii.at<float>(i,j) = _input.at<cv::Vec3b>(i,j)[1] +
                                    ch2_ii.at<float>(i,prev_x) * bound_x +
                                    ch2_ii.at<float>(prev_y,j) * bound_y;
            ch3_ii.at<float>(i,j) = _input.at<cv::Vec3b>(i,j)[2] +
                                    ch3_ii.at<float>(i,prev_x) * bound_x +
                                    ch3_ii.at<float>(prev_y,j) * bound_y;
            std::cout << j << " , " << i << " ; " << ch1_ii.at<float>(i,j) << " ; " << ch2_ii.at<float>(i,j) << " ; " << ch1_ii.at<float>(i,j) << std::endl;
        }
    }

//    cv::normalize(ch1_ii,ch1_ii, .0, 1.0, cv::NORM_MINMAX);
//    cv::normalize(ch2_ii,ch2_ii, .0, 1.0, cv::NORM_MINMAX);
//    cv::normalize(ch3_ii,ch3_ii, .0, 1.0, cv::NORM_MINMAX);

    cv::imshow("CH1_II", ch1_ii);
    cv::imshow("CH2_II", ch2_ii);
    cv::imshow("CH3_II", ch3_ii);
    cv::waitKey(1);
}

// Average ~ 22 ms ~~ 45 FPS
// Updated June 16th 2019 ~ Avg. 18 ms ~~ 55 FPS

// TODO : change findContour into segmenting colour simultaneously
