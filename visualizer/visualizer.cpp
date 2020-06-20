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

#include <unistd.h>

#define _USE_OPENCV true
//#include <MVS/Interface.h>

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

    vector<Matx33d> Rs;
    vector<Vec3d> Ts;
    vector<int> camera_ids;

//    vector<Mat> points;
    vector<Vec3d> points;
    vector<Vec3b> point_colors;
    vector<int> point_ids;

    Matx33d K;
    viz::Viz3d window;


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

        int counter = 0;

        // Then proceed to getting the rest
        while (getline(infile, line))
        {
            counter++;
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
                    K = Matx33d(
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++)
                            );
//                    cout << "DONE" << endl;
                    break;
                }

                case TYPE_CAMERA: {
                    Vec3d t;
                    Matx33d R;

//                    cout << "Loading R: ";
                    R = Matx33d(
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++)
                    );
//                    cout << "DONE" << endl;

//                    cout << "Loading T: ";
                    t = Vec3d(
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++)
                    );
//                    t = Mat(3, 3, CV_32FC3, t_values);
//                    cout << "DONE" << endl;

//                    cout << "R: " << R << endl;
//                    cout << "t: " << t << endl;

                    Rs.push_back(R);
                    Ts.push_back(t);
                    camera_ids.push_back(item_id);

                    break;
                }
                case TYPE_POINT: {
                    vector<float> point_color_values;
                    Vec3d point;
                    Vec3b point_color;

//                    cout << "Loading Point... ";
                    point = Vec3d(
                            stod(*beg++),
                            stod(*beg++),
                            stod(*beg++)
                    );
//                    cout << "DONE" << endl;

                    if (beg == tok.end()) {
//                        cout << "Point Color not found" << endl;
                        point_color = Vec3b(0, 0, 0);
                    } else {
                        point_color = Vec3d(
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

    static void mouseCallback(int event, int x, int y, int flags, void* userdata) {
        cout << "Mouse event" << endl;
    }

    void visualize3D() {
        cout << "Visualize reconstruction" << endl;

        // Create 3D windows
        window = viz::Viz3d("SfM");
        window.setWindowSize(Size(2500, 1500));
        window.setWindowPosition(Point(150, 150));
        window.setBackgroundColor(viz::Color::white());

        // Recovering cameras
        vector<Affine3d> path;
        for (size_t index = 0; index < Rs.size(); ++index) {
            path.push_back(Affine3d(Rs[index], Ts[index]));
        }

        // Add the pointcloud
        viz::WCloud cloud_widget(points, point_colors);
        cloud_widget.setRenderingProperty( cv::viz::POINT_SIZE, 8 );
        window.showWidget("point_cloud", cloud_widget);
        // Add cameras
        window.showWidget("camera_path", viz::WTrajectory(path, viz::WTrajectory::PATH, 0.1, viz::Color::black()));
        window.showWidget("camera_frames", viz::WTrajectory(path, viz::WTrajectory::FRAMES, 0.1, viz::Color::black()));
        window.showWidget("cameras_frustums", viz::WTrajectoryFrustums(path, K, .1, viz::Color::navy()));
        window.setViewerPose(path[0]);

        bool show_viewer_pose = true;
        bool use_viewer_pose = false;

        if (!show_viewer_pose) {
            window.spin();
        } else {
            Affine3d pose;

            const double pose_vals[] = {0.97492, 0.0146819, -0.222069, 4.80515, 0.020948, 0.987338, 0.157242, -2.38275, 0.221565, -0.157951, 0.962268, -12.7444, 0, 0, 0, 1, };
            Affine3d pose2 = Affine3d(pose_vals);

            if (use_viewer_pose) {
                window.setViewerPose(pose2);
            }

            int counter = 0;
            window.spinOnce(200, true);
            while (!window.wasStopped()) {
                pose = window.getViewerPose();
                for (int i = 0; i < 16; i++) {
                    cout << pose.matrix.val[i] << ", ";
                }
                cout << endl;
                window.spinOnce(1000, true);
            }
        }
        cout << "Visualisation closed" << endl;
    }

//    static void mouseCallback(const viz::MouseEvent::Type& type, const viz::MouseEvent::MouseButton& button, const Point& p, int modifiers) {
//        cout << "Mouse event" << endl;
//    }
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
