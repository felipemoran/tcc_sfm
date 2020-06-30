# TCC - Structure from Motion

#### [PT-BR]
Este repositório comtém o trabalho de conclusão de curso realizado no ano de 2020 por Felipe Morán e Caio Garcia Cancian para a obtenção do diploma de Engenheiro da Escopa Politécnica da Universidade de São Paulo. O repositório contém o código fonte desenvolvido, assim como alguns _datasets_ e o relatório final entregue (na seção de _releases_).

O presente trabalho visa abordar o problema de visão computacional conhecido como _Structure from Motion_ em que, essencialmente, busca-se reconstruir, a partir de imagens de uma cena estática adquiridas em posições distintas, tanto o modelo tridimensional dessa cena, quanto a trajetória da câmera responsável por registrar as imagens. Para o caso específico desse trabalho, as imagens adquiridas são provenientes de uma sequência de vídeo e portanto os algoritmos estudados e implementados são direcionados ao tratamento dessse tipo de mídia. 

 
#### [EN]
This repo contains the senior project developed by Felipe Morán and Caio Garcia Cancian in 2020 as a requirement for the Engineering degree of Polytechnic School of the University of São Paulo, Brazil. It containt the source code developed as well as some datasets and the resulting report (in Portuguese in the releases section). 

The present work aims to address the computer vision problem known as Structure from Motion in which, essentially, it is necessary to reconstruct, from images of a static scene acquired in different positions, both the three-dimensional model of this scene and the trajectory of the camera responsible for registering the images. In this specific project the image source if a video file and therefore all studied and implemented algorithms are tirected towards this type of media.

## File Structure
The code is organized as follows:

- run_scripts/ : folder containing main scripts for reconstruction
    - main.py : main execution script
    - plot_stats.py : accessory script for plotting some error stats
- pieline/ : main module
    - video_pipeline.py : contains class VideoPipeline, responsible for orchestrating the reconstruction from extraction of feature to final optimizations.
    - synthetic_pipeline.py : contains class SyntheticPipeline which inherits from VideoPipeline. This class allows testing the reconstruction phase with synthetic data without having to rely on a video file and the difficulties associated with feature detection and tracking 
    - config.py : contain the definition of all config modules as well as functions for loading and parsing config files
    - video_algorithms.py : contains all algorithms that manipulate the video file/frames directly, namely KLT
    - reconstruction_algorithms.py : contains all algorithms that reconstruct camera poses and 3D points from frame features
    - init_algorithms.py : contains reconstruction initialization logic
    - bundle_adjuestment.py : contains bundle adjustment code which optimizes the reconstruction and improves robustness 
    - utils.py : misc functions that aid all other modules 
- config.yaml : example of config file with recommended parmeter set
- visualizar/
    - visualizer.h : visualizer module header file
    - visualizer.cpp : visualizer implementation
    - CMakeLists.txt
    
## Dependencies:
- Python 3.8
- OpenCV (tested with version 4.2.0)
- dacite
- ruamel.yaml
- numpy
- scipy
- pandas (for plots only)
- seaborn (for plots only)

## Configuration File

Below is the structure of the config.yaml file with some notes on each item. It is recomended that you read this section with the example config file side-by-side to better understand the structure and file syntax.

- pipeline_type: "synthetic" or "videoe"
- file_path: "/Users/userenamee/path/to/video/file/video.m4v" or "1", "2" or "3" if using synthetic data
- synthetic_config:
    - noise_covariance: noise to be added to synthetic points
    - number_of_cameras: maximum number of synthetic cameras reconstructed
    - case_3: creates an empty cube with a circle of cameras pointing to it
        - radius: radius of camera circle
        - number_of_cameras: number or cameras in the circle
        - step_size: distance along axis X, Y and Z between each point
        - x_points: points along X axis
        - y_points: points along Y axis
        - z_points: points along Z axis
- camera_matrix: calibration parameters for the camera used during data acquisition
- use_five_pt_algorithm: bool indicating if 5 point algorithm is to be used during reconstruction 
- use_solve_pnp: bool indicating if PnP algorithm should be used during reconstruction
- use_reconstruct_tracks: bool indicating if triangulation step should be performed after rotation matrix and translation vector are calculated
- klt:
    - calculate_every_frame: bool indicating if KLT feature tracking should be performed at every step even if some frames should be skipped
    - display_klt_debug_frames: bool indicating if debug visualization frames should be displayed 
    - klt_debug_frames_delay: delay time after displaying debug frame
    - frames_to_skip: number of frames to be skipped between two frames used for reconstruction
    - reset_period: period between two KLT frame detections
    - closeness_threshold: threshold (in pixels) to determine if two features are close
    - max_features: maximum number of KLT features to be tracked
    - corner_selection:
        - quality_level: value from 0 to 1 indicating quality of detected features
        - min_distance: minimum distance between detected points
        - block_size: block size for feature detection
    - optical_flow:
        - window_size: pyramid window size
            - width: 
            - height: 
        - max_level: maximum pyramid level
        - criteria:
            - criteria_sum: stopping criteria value (default is 3)
            - max_iter: maximum number of iterations
            - eps: moviment epsilon
- error_calculation:
    - period: how often should the error be calculated
    - window_length: how many past frames should be used
    - online_calculation: bool indicating if online calculation should be performed 
    - post_calculation: bool if post-reconstruction error calculation should be performed
- bundle_adjustment:
    - tol: tolerance for final solution, stopping criteria
    - method: method to be used (default it trf)
    - verbose: verbosity level of optimization step
    - camera_matrix: calibration parameters for the camera used during data acquisition
    - use_with_rolling_window: bool indicating if BA should be done online with rolling window
    - rolling_window:
        - method: type of window to be used "growing_step" or "constant_step"
        - period: period to execute BA in number of processed frame
        - length: number of frames to include in the rolling window
        - step: 1 
    - use_at_end: bool indicating if BA should be performed after reconstruction is done
- five_pt_algorithm:
    - min_number_of_points: required minimum number of points for reconstruction 
    - essential_mat_threshold: the maximum distance from a point to an epipolar line in pixels, beyond which the point is considered an outlier
    - ransac_probability: It specifies a desirable level of confidence (probability) that the estimated matrix is correct
    - refine_matches_repetitions: number of refinement steps to be performed
    - save_optimized_projections: bool indicating if refined tracks should be used on further steps 
    - camera_matrix: calibration parameters for the camera used during data acquisition
    - distance_threshold: threshold distance which is used to filter out far away points (i.e. infinite points)
- solve_pnp:
    - min_number_of_points: required minimum number of points for reconstruction
    - x: calibration parameters for the camera used during data acquisition
    - use_epnp: bocamera_matriol indicating if EPnP algorithm should be used. If yes, previous guess for R and T (such as from 5 point algorithm) is discarted
    - use_iterative_pnp: bool indicating if iterative PnP algorithm should be used to further refine rotation and translation matrix
- init:
    - error_threshold: projection error threshold that determines if reconstruction attempt is good enough  
    - num_reconstruction_frames: number of frames to be used during reconstruction. First 2 will be used with 5 point algorithm
    - num_error_calculation_frames: number of frames to be used for out-of-sample error calculation. These frames are reconstructed using EPnP and iterative PnP algorithms 

    