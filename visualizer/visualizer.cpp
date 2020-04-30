#include "visualizer.h"

#include <iostream>
//#include <algorithm>
#include <string>
//#include <numeric>

#define CERES_FOUND true

#include <opencv2/opencv.hpp>
//#include <opencv2/sfm.hpp>
#include <utility>
#include <opencv2/viz.hpp>
#include <opencv2/core/utils/logger.hpp>
//#include <opencv2/core/utils/filesystem.hpp>
//#include <opencv2/xfeatures2d.hpp>

#include <boost/filesystem.hpp>
//#include <boost/graph/graph_traits.hpp>
//#include <boost/graph/adjacency_list.hpp>
//#include <boost/graph/connected_components.hpp>
//#include <boost/graph/graphviz.hpp>
#include <boost/tokenizer.hpp>


#define _USE_OPENCV true
#include <MVS/Interface.h>

using namespace cv;
using namespace cv::utils::logging;
using namespace std;
using namespace boost;
namespace fs = boost::filesystem;

#define TYPE_CALIBRATION_MATRIX 0
#define TYPE_CAMERA 1
#define TYPE_POINT 2

class Visualizer {
private:
    const string dir;    // directory with file
    ifstream infile; // opened file
    string file_info;

    vector<Matx33f> Rs;
    vector<Vec3f> Ts;
    vector<int> camera_ids;

//    vector<Mat> points;
    vector<Vec3f> points;
    vector<Vec3b> point_colors;
    vector<int> point_ids;

    Matx33f K;

public:
    explicit Visualizer(string dir): dir(std::move(dir)) {

    }

    int run() {
        int ret;

        cout << "Loading file" << endl;
        ret = load_file();

        if (ret != 0) {
            cout << "[Error] Couldn't open file" << endl;
            return -1;
        }

        cout << "Parsing file" << endl;
        parse_file();


        cout << "Generating visualization" << endl;
        visualize3D();

        return 0;
    }

private:
    int load_file() {
//        infile = ifstream(dir);
//        string data(dir);

        cout << "Trying to open file " << dir << endl;
        infile = ifstream(dir);
        if (!infile.is_open()) return 1;

        return 0;
    }

    void parse_file() {
        int item_type;
        int item_id;
        string item_data;
        string line;

        typedef tokenizer< escaped_list_separator<char> > Tokenizer;

        tokenizer<escaped_list_separator<char>>::iterator beg;

        // Start by geting the file info
        getline(infile, file_info);
        cout << "File info: " << file_info << endl << endl;

        // Then proceed to getting the rest
        while (getline(infile, line))
        {
            Tokenizer tok(line);
            beg = tok.begin();

            item_type = stoi(*beg++);
            item_id = stoi(*beg++);

//            cout << endl;
//            cout << "item type: " << item_type << endl;
//            cout << "item id  : " << item_id   << endl;

            switch(item_type) {
                case TYPE_CALIBRATION_MATRIX: {
//                    cout << "Loading Camera Matrix... ";
                    K = Matx33f(
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++)
                            );
//                    cout << "DONE" << endl;
                    break;
                }

                case TYPE_CAMERA: {
                    Vec3f t;
                    Matx33f R;

//                    cout << "Loading R: ";
                    R = Matx33f(
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++)
                    );
//                    cout << "DONE" << endl;

//                    cout << "Loading T: ";
                    t = Vec3f(
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++)
                    );
//                    t = Mat(3, 3, CV_32FC3, t_values);
//                    cout << "DONE" << endl;

                    cout << "R: " << R << endl;
                    cout << "t: " << t << endl;

                    Rs.push_back(R);
                    Ts.push_back(t);
                    camera_ids.push_back(item_id);

                    break;
                }
                case TYPE_POINT: {
                    vector<float> point_color_values;
                    Vec3f point;
                    Vec3b point_color;

//                    cout << "Loading Point... ";
                    point = Vec3f(
                            stof(*beg++),
                            stof(*beg++),
                            stof(*beg++)
                    );
//                    cout << "DONE" << endl;

                    if (beg == tok.end()) {
//                        cout << "Point Color not found" << endl;
                        point_color = Vec3b(0, 0, 0);
                    } else {
                        point_color = Vec3f(
                                stoi(*beg++),
                                stoi(*beg++),
                                stoi(*beg++)
                        );
//                        cout << "DONE" << endl;
                    }

                    points.push_back(point);
                    point_colors.push_back(point_color);
                    point_ids.push_back(item_id);
                }
            }
//            cout << endl;
        }
    }

    void visualize3D() {
        cout << "Visualize reconstruction" << endl;

        // Create 3D windows
        viz::Viz3d window("Coordinate Frame");
        window.setWindowSize(Size(500, 500));
        window.setWindowPosition(Point(150, 150));
        window.setBackgroundColor(viz::Color::white());

        // Recovering cameras
        vector<Affine3d> path;
        for (size_t index = 0; index < Rs.size(); ++index) {
            cout << Rs[index] << endl;
            cout << Ts[index] << endl;
            path.push_back(Affine3d(Rs[index], Ts[index]));
        }

        // Add the pointcloud
        viz::WCloud cloud_widget(points, point_colors);
        cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 8 );
        window.showWidget("point_cloud", cloud_widget);
        // Add cameras
        window.showWidget("cameras_frames_and_lines", viz::WTrajectory(path, viz::WTrajectory::BOTH, 0.1, viz::Color::black()));
        window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, 1, viz::Color::navy()));
        window.setViewerPose(path[0]);

        /// Wait for key 'q' to close the window
        cout << "Press 'q' to close ... " << endl;

        window.spin();
    }
};




int main(int argc, char** argv) {
    utils::logging::setLogLevel(utils::logging::LOG_LEVEL_DEBUG);

    cv::CommandLineParser parser(argc, argv,
                                 "{help h ? |       | help message}"
                                 "{@dir     | .     | directory with image files for reconstruction }"
    );

    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }

    Visualizer visualizer(parser.get<string>("@dir"));
    visualizer.run();

    return 0;
}
