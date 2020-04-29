import os
import cv2
import glob
import copy
import random
import argparse
import itertools
import collections
import numpy as np
import open3d as o3d
import networkx as nt
import matplotlib.pyplot as plt

from os import path


class StructureFromMotion:

    MATCH_RATIO_THRESHOLD = 0.8

    def __init__(self,
                 dir,
                 match_survival_rate,
                 visualize,
                 save_mvs,
                 save_cloud,
                 save_debug_visualization):
        self.dir = dir
        self.match_survival_rate = match_survival_rate
        self.visualize = visualize
        self.save_mvs = save_mvs
        self.save_cloud = save_cloud
        self.save_debug_visualization = save_debug_visualization

    def run(self):

        # Start by finding the images
        files, filenames = self._find_images(self.dir)

        # Then get the features from images
        keypoints, descriptors = self._extract_features(files, filenames)

        # TODO: change above functions to video

        # Match them with one another
        # TODO: videos won't match features between every 2 frame combination
        matches = self.match_features(files, filenames, keypoints, descriptors)

        # TODO: what is this?
        # TODO: implement
        tracks = self._build_tracks(files, filenames, keypoints, descriptors, matches)

        # TODO: what is this?
        # TODO: implement
        rs, ts, k_, points_3d, points_3d_colors = self.reconstruct_from_tracks(files, tracks)

        # Show results
        if self.visualize:
            # TODO: implement
            self._visualize_3d(rs, ts, points_3d, points_3d_colors)

        # TODO: Save what to file?
        if self.save_mvs is not None:
            # TODO: implement
            self.save_to_mvs_file()

        # TODO: Save what to file?
        if self.save_cloud is not None:
            # TODO: implement
            # CV_LOG_INFO( & TAG, 'Save point cloud to: ' + saveCloud);
            # viz::writeCloud(saveCloud, pointCloud, pointCloudColor);
            pass

    # =================== INTERNAL FUNCTIONS ===========================================================================

    def _find_images(self, dir):
        print('Looking for files in {}'.format(dir))

        types = ['jpg', 'JPG', 'jpeg', 'JPEG', 'png', 'PNG']
        filenames = []

        assert path.isdir(dir), 'Invalid directory location'

        for filetype in types:
            filenames += glob.glob(path.join(dir, '*.' + filetype))
        filenames.sort()
        files = []

        print('Found {} files'.format(len(filenames)))
        for filename in filenames:
            print('\t{}'.format(filename))
            files += [cv2.imread(filename)]

        return files, filenames

    def _extract_features(self, files, filenames):
        print("Extracting features")

        detector = cv2.AKAZE.create()
        extractor = cv2.AKAZE.create()

        keypoints = []
        descriptors = []

        for file_index, (file, filename) in enumerate(zip(files, filenames)):
            grayscale = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

            file_keypoints, file_descriptors = detector.detectAndCompute(grayscale, None)

            print("Found {} keypoints in {}".format(len(file_keypoints), filename))

            keypoints += [file_keypoints]
            descriptors += [file_descriptors]

            if self.save_debug_visualization:
                out_image = cv2.drawKeypoints(file, file_keypoints, None, color=(0, 0, 255))
                cv2.imwrite(path.join(os.getcwd(),
                                      'out',
                                      os.path.splitext(os.path.basename(filename))[0] + '_featuers.jpg'
                                      ), out_image)

        return keypoints, descriptors

    def _filter_with_ratio_test(self, raw_match):
        ratio_matched = []

        for pair in raw_match:
            if len(pair) < 2:
                continue

            if pair[0].distance < self.MATCH_RATIO_THRESHOLD * pair[1].distance:
                ratio_matched += [pair[0]]

        return ratio_matched

    def _merge_with_reciprocity(self, match_ij, match_ji, size_descriptors):
        match_merged_partial = [None] * size_descriptors
        match_merged = []

        for item in match_ji:
            match_merged_partial[item.queryIdx] = item.trainIdx

        for item in match_ij:
            if match_merged_partial[item.trainIdx] != item.queryIdx:
                continue
            match_merged += [item]

        return match_merged

    def match_features(self, files, filenames, keypoints, descriptors):
        print("Matching features")

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = {}

        # for (index_i, file_i, filename_i, descriptors_i, keypoints_i), (index_j, file_j, filename_j, descriptors_j, keypoints_j) \
        #         in itertools.combinations(zip(range(len(files)), files, filenames, descriptors, keypoints), r=2):

        base_zip = list(zip(range(len(files)), files, filenames, descriptors, keypoints))
        for (index_i, file_i, filename_i, descriptors_i, keypoints_i), (index_j, file_j, filename_j, descriptors_j, keypoints_j) \
                in zip(base_zip[0:-2], base_zip[1:-1]):

            raw_match_ij = matcher.knnMatch(descriptors_i, descriptors_j, 2)
            raw_match_ji = matcher.knnMatch(descriptors_j, descriptors_i, 2)

            random.shuffle(raw_match_ij)

            match_ij = self._filter_with_ratio_test(raw_match_ij)
            match_ji = self._filter_with_ratio_test(raw_match_ji)

            match_merged = self._merge_with_reciprocity(match_ij, match_ji, max(len(descriptors_i), len(descriptors_j)))

            selected_points_i = []
            selected_points_j = []

            for match in match_merged:
                selected_points_i += [keypoints_i[match.queryIdx].pt]
                selected_points_j += [keypoints_j[match.trainIdx].pt]

            fundamental_mat, inliers_mask = cv2.findFundamentalMat(np.float_(selected_points_i), np.float_(selected_points_j))

            final_selection = list(itertools.compress(match_merged, inliers_mask))

            print("Matching {} and {} : {} / {}".format(
                filename_i, filename_j, len(final_selection), len(match_ij)))

            if len(final_selection) / len(match_ij) < self.match_survival_rate:
                print('Final match {} -> {} has less than {} inliers from original. Skip'.format(filename_i, filename_j, self.match_survival_rate))
                continue


            matches[(index_i, index_j)] = final_selection

            if self.save_debug_visualization:
                debug_info = [
                    {
                        'title': 'Raw Match',
                        'key': 'raw_match',
                        'data': matcher.match(descriptors_i, descriptors_j),
                    }, {
                        'title': 'Ratio Test Filtered',
                        'key': 'ratio_test_filtered',
                        'data': match_ij,
                    }, {
                        'title': 'Reciprocal Filtered',
                        'key': 'reciprocal_filtered',
                        'data': match_merged,
                    }, {
                        'title': 'Epipolar Filtered',
                        'key': 'epipolar_filtered',
                        'data': final_selection,
                    }
                ]

                for item in debug_info:
                    out = cv2.drawMatches(file_i, keypoints_i, file_j, keypoints_j, item['data'], None, (255, 0, 0))
                    cv2.putText(out, item['title'], (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (255, 255, 255), 2)
                    cv2.putText(out, "# Matches: {}".format(len(item['data'])), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))
                    cv2.imwrite(path.join(os.getcwd(),
                                          'out',
                                          '{}_{}_{}.jpg'.format(
                                              os.path.splitext(os.path.basename(filename_i))[0],
                                              os.path.splitext(os.path.basename(filename_j))[0],
                                              item['key']
                                          )), out)

        return matches

    def _build_tracks(self, files, filenames, keypoints, descriptors, matches):
        print("Building tracks")

        Feature = collections.namedtuple('Feature', 'file_index, feature_index, pt')

        graph = nt.Graph()

        # Simultaneously add nodes and edges to the graph
        for (image_index_src, image_index_dst), pair_matches in matches.items():
            for match in pair_matches:
                src_feature_index = match.queryIdx
                src_feature_point = keypoints[image_index_src][src_feature_index].pt
                src_feature = Feature(image_index_src, src_feature_index, src_feature_point)
                graph.add_node(src_feature)

                dst_feature_index = match.trainIdx
                dst_feature_point = keypoints[image_index_dst][dst_feature_index].pt
                dst_feature = Feature(image_index_dst, dst_feature_index, dst_feature_point)
                graph.add_node(dst_feature)

                graph.add_edge(src_feature, dst_feature)

        components = list(nt.connected_components(graph))
        tracks = np.array([
            np.full((2, len(components)), -1, dtype=np.float_)
            for _ in filenames
        ])

        components_mask = [0]*len(components)

        for component_index, component in enumerate(components):
            file_indexes = [item.file_index for item in component]
            if len(file_indexes) > len(set(file_indexes)):
                # There's a duplicate file in this component. Ignore it
                continue

            components_mask[component_index] = 1

            for node in component:
                tracks[node.file_index][0][component_index] = node.pt[0]
                tracks[node.file_index][1][component_index] = node.pt[1]

        # Remove tracks for bad components
        tracks_cleaned = np.array([
            np.zeros((2, sum(components_mask)), dtype=np.float_)
            for _ in filenames
        ])

        for file_index in range(len(filenames)):
            for line_index in range(2):
                tracks_cleaned[file_index][line_index] = np.array(
                    list(
                        itertools.compress(tracks[file_index][line_index], components_mask)
                    )
                )
        tracks = tracks_cleaned

        print("Total number of components found: {}".format(len(components)))
        print("Number of good components: {}".format(sum(components_mask)))
        if sum(components_mask) > 0:
            print('Average component size: {}'.format(sum([len(c) for c in itertools.compress(components, components_mask)])/sum(components_mask)))

    #     if (saveDebugVisualizations) {
    #         struct my_node_writer {
    #             my_node_writer(Graph& g_, const map<string,int>& iid_) : g (g_), iid(iid_) {};
    #             void operator()(std::ostream& out, Vertex v) {
    #                 const int imgId = iid[g[v].image];
    #                 out << " [label=\"" << imgId << "\" colorscheme=\"accent8\" fillcolor="<<(imgId+1)<<" style=filled]";
    #             };
    #             Graph g;
    #             map<string,int> iid;
    #         };
    #         std::ofstream ofs("match_graph_good_components.dot");
    #         write_graphviz(ofs, gGoodComponents, my_node_writer(g, imageIDs));
    #         std::ofstream ofsf("match_graph_filtered.dot");
    #         write_graphviz(ofsf, gFiltered, my_node_writer(g, imageIDs));
    #     }

        if self.save_debug_visualization:
            img_size_x = np.shape(files[0])[1]
            img_size_y = np.shape(files[0])[0]

            colors = [(240, 248, 255),
                      (250, 235, 215),
                      (0, 255, 255),
                      (127, 255, 212),
                      (240, 255, 255),
                      (245, 245, 220),
                      (255, 228, 196),
                      (255, 235, 205),
                      (0, 0, 255),
                      (138, 43, 226),
                      (165, 42, 42),
                      (222, 184, 135)
            ]

            files_unified = cv2.hconcat(files)
            for i in range(20):
                track_id = random.randint(0, np.shape(tracks_cleaned)[2]-1)
                out = files_unified

                for file_pair_index in range(len(files) - 1):
                    pt_1 = copy.copy(tracks_cleaned[file_pair_index, :, track_id])
                    pt_2 = copy.copy(tracks_cleaned[file_pair_index + 1, :, track_id])

                    if pt_1[0] < 0 or pt_1[1] < 0 or pt_2[0] < 0 or pt_2[1] < 0:
                        continue

                    color = colors[i % len(colors)]

                    pt_1[0] += file_pair_index * img_size_x
                    pt_2[0] += (file_pair_index + 1) * img_size_x

                    pt_1 = tuple(pt_1.astype(int))
                    pt_2 = tuple(pt_2.astype(int))

                    out = cv2.circle(out, pt_1, 7, color, cv2.FILLED)
                    out = cv2.circle(out, pt_2, 7, color, cv2.FILLED)
                    out = cv2.line(out, pt_1, pt_2, color, 3)

                patch_size = 20
                for file_index in range(len(files)):
                    pt = copy.copy(tracks_cleaned[file_index, :, track_id])
                    pt = tuple(pt.astype(int))

                    # if it's too close to a border or it's not part of a track, skip it
                    if (pt[0] < patch_size or pt[1] < patch_size or
                        pt[0] > img_size_x - patch_size or pt[0] > img_size_y - patch_size):
                        continue

                    cv2.imwrite("out/track_{}_{}.png".format(track_id, i),
                                files[file_index][pt[0]-patch_size:pt[0]+patch_size, pt[1]-patch_size:pt[1]+patch_size,:])

            cv2.imwrite('out/tracks.jpg', out)

    #
    #     // Show
    #     track
    #     patches
    #     const
    #     int
    #     patchSize = 20;
    #     const
    #     Point2f
    #     patch(patchSize, patchSize);
    #     for (int i = 0; i < tracks.size(); i++) {
    #     Point2f a = Point2f(tracks[i].col(trackId));
    #     if (a.x < patchSize or a.y < patchSize or
    #     a.x > imgS.width - patchSize or a.y > imgS.height - patchSize) {
    #     continue;
    #
    # }
    #
    # imwrite("track_" + to_string(trackId) + "_" + to_string(i) + ".png",
    # imagesM[i](Rect(a - patch, a + patch)));
    # }
    # }
    # }


        return tracks_cleaned

    def reconstruct_from_tracks(self, files, tracks):
        print('Reconstructing from {} tracks'.format(np.shape(tracks)[2]))

        image_size = np.shape(files[0])[0:2]
        f = max(image_size)
        k = np.array([[f, 0, image_size[0]/2],
                      [0, f, image_size[1]/2],
                      [0, 0, 1]])

        (rs, ts, k_, points_3d) = cv2.sfm.reconstruct(points2d=tracks, K=k, Rs=None, Ts=None, points3d=None, is_projective=True)

        print("Reconstruction: ")
        print("Estimated 3D points: {}".format(len(points_3d)))
        print("Estimated cameras: {}".format(len(rs)))
        print("Refined intrinsics: ")
        print(k_)

        if len(rs) != len(files):
            print("Unable to reconstruct all camera views ({})".format(len(files)))
            # return

        if np.shape(tracks)[2] > len(points_3d):
            print("Unable to reconstruct all tracks ({})".format(np.shape(tracks)[2]))

        points_3d_colors = np.zeros([np.shape(points_3d)[0], 3])
        for index, point in enumerate(points_3d):
            for file, view_r, view_t in zip(files, rs, ts):
                point_2d, _ = cv2.projectPoints(point, view_r, view_t, k_, distCoeffs=None)
                point_2d = point_2d[0][0]
                point_2d = point_2d.astype(int)

                if 0 <= point_2d[0] < np.shape(file)[0] and \
                        0 <= point_2d[1] < np.shape(file)[1]:
                    points_3d_colors[index] = file[point_2d[0]][point_2d[1]]
                    break

        return rs, ts, k_, points_3d, points_3d_colors

    def _visualize_3d(self, rs, ts, points_3d, points_3d_colors):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.squeeze(points_3d))
        o3d.visualization.draw_geometries([pcd])


    def save_to_mvs_file(self):
        # TODO: implement
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir',
                        help='Directory with image files for reconstructions')
    parser.add_argument('-mrate', '--match_survival_rate', type=float,
                        help='Survival rate of matches to consider image pair success', default=0.5)
    parser.add_argument('-viz', '--visualize',
                        help='Visualize the sparse point cloud reconstruction?', action='store_true', default=False)
    parser.add_argument('-sd', '--save_debug_visualization',
                        help='Save debug visualizations to files?', action='store_true', default=False)
    parser.add_argument('-mvs', '--save_mvs',
                        help='Save reconstruction to an .mvs file? Provide filename')
    parser.add_argument('-sc', '--save_cloud',
                        help='Save reconstruction to a point cloud file (PLY, XYZ and OBJ). Provide filename')

    args = parser.parse_args()

    sfm = StructureFromMotion(**vars(args))
    sfm.run()
