""" Demo for ROS2 to show prediction results from real-time camera input.
    Based on demo.py by chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import rclpy
from rclpy.node import Node
import cv_bridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String, Float32, Header

# Import custom message type (must be a ROS2 type). Fail fast if unavailable.
try:
    from graspnet_msgs.msg import GraspPose
except Exception as import_err:
    print("[!] Missing ROS2 message type 'graspnet_msgs/GraspPose'.")
    print("    Please build and source the message package before running:")
    print("    1) cd /root/src && colcon build")
    print("    2) source /root/src/install/setup.bash")
    raise import_err
from cv_bridge import CvBridge
import threading
import time
import json
import queue
from scipy.spatial.transform import Rotation as R

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo as GrCameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--no_collision', action='store_true', help='Disable collision detection for debugging')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
parser.add_argument('--headless', action='store_true', help='Run in headless mode (no interactive visualization)')
parser.add_argument('--color_topic', type=str, default='/camera/color/image_raw', help='Color image topic')
parser.add_argument('--depth_topic', type=str, default='/camera/depth/image_raw', help='Depth image topic')
parser.add_argument('--camera_info_topic', type=str, default='/camera/color/camera_info', help='Camera info topic')
parser.add_argument('--grasp_topic', type=str, default='/graspnet/grasps', help='Topic to publish grasp results')
parser.add_argument('--processing_interval', type=float, default=2.0, help='Interval between processing frames in seconds [default: 2.0]')
cfgs = parser.parse_args()


class GraspNetRos2Node(Node):
    """ROS2 node for GraspNet real-time prediction."""
    
    def __init__(self, net):
        super().__init__('graspnet_ros2_node')
        
        self.net = net
        self.bridge = CvBridge()
        
        # Data storage
        self.color_image = None
        self.depth_image = None
        self.camera_info = None
        self.data_lock = threading.Lock()
        
        # Queue for passing grasp results from processing thread to publishing thread
        self.grasp_result_queue = queue.Queue(maxsize=5)
        
        # Thread control
        self.running = True
        self.processing_thread = None
        self.publishing_thread = None
        
        # Create subscribers
        self.sub_color = self.create_subscription(
            Image,
            cfgs.color_topic,
            self.color_callback,
            10)
        
        self.sub_depth = self.create_subscription(
            Image,
            cfgs.depth_topic,
            self.depth_callback,
            10)
        
        self.sub_camera_info = self.create_subscription(
            CameraInfo,
            cfgs.camera_info_topic,
            self.camera_info_callback,
            10)
        
        # Create publisher for grasp results
        # Using custom GraspPose message type
        self.grasp_pub = self.create_publisher(
            GraspPose,
            cfgs.grasp_topic,
            10)
        
        # Start processing and publishing threads
        self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
        self.publishing_thread = threading.Thread(target=self.publishing_loop, daemon=True)
        self.processing_thread.start()
        self.publishing_thread.start()
        
        self.get_logger().info('[*] GraspNet ROS2 node started')
        self.get_logger().info(f'[*] Subscribing to: {cfgs.color_topic}, {cfgs.depth_topic}, {cfgs.camera_info_topic}')
        self.get_logger().info(f'[*] Publishing grasps to: {cfgs.grasp_topic}')
        self.get_logger().info(f'[*] Processing interval: {cfgs.processing_interval}s')
    
    def color_callback(self, msg):
        """Callback for color image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            with self.data_lock:
                self.color_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')
    
    def depth_callback(self, msg):
        """Callback for depth image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.data_lock:
                self.depth_image = cv_image
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
    
    def camera_info_callback(self, msg):
        """Callback for camera info."""
        with self.data_lock:
            self.camera_info = msg
    
    def processing_loop(self):
        """Processing thread: continuously collect data and compute grasps."""
        while self.running:
            try:
                # Get latest data snapshot
                with self.data_lock:
                    if self.color_image is None or self.depth_image is None or self.camera_info is None:
                        time.sleep(0.1)
                        continue
                    
                    # Make copies
                    color_img = self.color_image.copy()
                    depth_img = self.depth_image.copy()
                    cam_info = self.camera_info
                
                # Process the frame
                grasp_result = self.process_frame(color_img, depth_img, cam_info)
                
                # Put result in queue for publishing thread (non-blocking, drop if queue full)
                if grasp_result is not None:
                    try:
                        self.grasp_result_queue.put_nowait(grasp_result)
                    except queue.Full:
                        self.get_logger().warn('Grasp result queue full, dropping oldest result')
                        try:
                            self.grasp_result_queue.get_nowait()  # Remove oldest
                            self.grasp_result_queue.put_nowait(grasp_result)  # Add new
                        except queue.Empty:
                            pass
                
                # Wait for next processing cycle
                time.sleep(cfgs.processing_interval)
                
            except Exception as e:
                self.get_logger().error(f'Error in processing loop: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(1.0)
    
    def grasp_to_pose_stamped(self, grasp):
        """Convert a Grasp object to PoseStamped message."""
        msg = PoseStamped()
        
        # Set header
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_color_optical_frame"
        
        # Set position (translation)
        trans = grasp.translation.astype(float)
        msg.pose.position.x = float(trans[0])
        msg.pose.position.y = float(trans[1])
        msg.pose.position.z = float(trans[2])
        
        # Set orientation (rotation matrix to quaternion)
        rotation_matrix = grasp.rotation_matrix.reshape(3, 3)
        quat = R.from_matrix(rotation_matrix).as_quat()
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        
        return msg
    
    def publishing_loop(self):
        """Publishing thread: continuously publish grasp results from queue."""
        while self.running:
            try:
                # Get grasp result from queue (blocking with timeout)
                try:
                    grasp_data = self.grasp_result_queue.get(timeout=0.5)
                    
                    # Convert grasp to GraspPose message (ROS2 message type)
                    best_grasp = grasp_data['grasp']
                    pose_msg = self.grasp_to_pose_stamped(best_grasp)
                    gripper_width = float(best_grasp.width)
                    
                    # Create GraspPose message
                    # Message structure: geometry_msgs/PoseStamped target_pose, float32 gripper_width
                    grasp_pose_msg = GraspPose()
                    grasp_pose_msg.target_pose = pose_msg
                    grasp_pose_msg.gripper_width = gripper_width
                    
                    # Publish GraspPose message
                    self.grasp_pub.publish(grasp_pose_msg)
                    
                    self.get_logger().info(
                        f'Published grasp - processing_time: {grasp_data["processing_time"]:.4f}s, '
                        f'gripper_width: {gripper_width:.4f}m, '
                        f'pose: [{pose_msg.pose.position.x:.4f}, {pose_msg.pose.position.y:.4f}, {pose_msg.pose.position.z:.4f}]'
                    )
                except queue.Empty:
                    continue
                    
            except Exception as e:
                self.get_logger().error(f'Error in publishing loop: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(0.1)
    
    def process_frame(self, color_img, depth_img, cam_info):
        """Process one frame and return grasp results."""
        
        # Record start time for performance measurement
        start_time = time.time()
        
        # Process the frame
        try:
            print('[*] Starting frame processing...')
            # Convert images to numpy arrays
            color = color_img.astype(np.float32) / 255.0
            depth = depth_img.astype(np.float32)
            print('[*] Converted images to numpy arrays')
            
            # Extract camera intrinsics
            print('[*] Extracting camera intrinsics...')
            K = cam_info.k
            fx = K[0]
            fy = K[4]
            cx = K[2]
            cy = K[5]
            print(f'[*] Camera intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}')
            
            # Create camera info object
            height, width = depth.shape
            print(f'[*] Creating camera info: {width}x{height}')
            
            # Check if depth values need scaling
            # Depth images from RealSense cameras are typically in millimeters
            # but the model expects values in meters, so we need to scale by 1000
            depth_min, depth_max = depth.min(), depth.max()
            depth_mean = depth.mean()
            print(f'[*] Depth range: min={depth_min:.4f}, max={depth_max:.4f}, mean={depth_mean:.4f}')
            
            # RealSense cameras typically output depth in millimeters (0-65535 range)
            # We need to convert to meters for the model
            # Check if depth is in millimeters (typical range 0-7000 for RealSense)
            if depth_max > 10:
                print('[*] Detected depth likely in millimeters, using scale_factor=1000.0')
                scale_factor = 1000.0
            else:
                print('[*] Detected depth already in meters, using scale_factor=1.0')
                scale_factor = 1.0
            print(f'[*] Using depth scale factor: {scale_factor}')
            
            # Create workspace mask AFTER determining scale factor
            print('[*] Creating workspace mask...')
            # Scale depth to meters for filtering
            depth_meters = depth / scale_factor
            # Filter out points that are too close (noise) or too far (not useful)
            valid_depth = (depth_meters > 0.3) & (depth_meters < 3.0)  # Filter depth range: 30cm to 3m
            workspace_mask = valid_depth
            print(f'[*] Workspace mask: {np.sum(workspace_mask)} valid pixels (depth range: 0.3-3.0m)')
            
            camera = GrCameraInfo(width, height, fx, fy, cx, cy, scale=scale_factor)
            
            # Generate point cloud
            print('[*] Generating point cloud...')
            cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
            print(f'[*] Point cloud created: {cloud.shape}')
            
            # Get valid points
            print('[*] Getting valid points...')
            mask = (workspace_mask & (depth > 0))
            cloud_masked = cloud[mask]
            color_masked = color[mask]
            print(f'[*] Masked cloud: {len(cloud_masked)} points')
            
            if len(cloud_masked) == 0:
                print('[*] ERROR: No valid points in point cloud')
                self.get_logger().warning('No valid points in point cloud')
                return None
            
            # Sample points
            print(f'[*] Sampling {cfgs.num_point} points from {len(cloud_masked)}...')
            if len(cloud_masked) >= cfgs.num_point:
                idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
            else:
                idxs1 = np.arange(len(cloud_masked))
                idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
                idxs = np.concatenate([idxs1, idxs2], axis=0)
            cloud_sampled = cloud_masked[idxs]
            color_sampled = color_masked[idxs]
            print(f'[*] Sampled cloud: {cloud_sampled.shape}')
            
            # Convert data for network
            print('[*] Converting data for network...')
            end_points = dict()
            cloud_sampled_tensor = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f'[*] Using device: {device}')
            cloud_sampled_tensor = cloud_sampled_tensor.to(device)
            print(f'[*] Cloud tensor shape: {cloud_sampled_tensor.shape}')
            print(f'[*] Cloud tensor stats: min={cloud_sampled_tensor.min():.4f}, max={cloud_sampled_tensor.max():.4f}, mean={cloud_sampled_tensor.mean():.4f}')
            
            # Check coordinate ranges per axis
            x_min, x_max = cloud_sampled_tensor[0, :, 0].min(), cloud_sampled_tensor[0, :, 0].max()
            y_min, y_max = cloud_sampled_tensor[0, :, 1].min(), cloud_sampled_tensor[0, :, 1].max()
            z_min, z_max = cloud_sampled_tensor[0, :, 2].min(), cloud_sampled_tensor[0, :, 2].max()
            print(f'[*] Coordinate ranges - X: [{x_min:.4f}, {x_max:.4f}], Y: [{y_min:.4f}, {y_max:.4f}], Z: [{z_min:.4f}, {z_max:.4f}]')
            
            end_points['point_clouds'] = cloud_sampled_tensor
            end_points['cloud_colors'] = color_sampled
            
            # Create visualization point cloud
            print('[*] Creating visualization point cloud...')
            print(f'[*] Cloud masked shape: {cloud_masked.shape}, dtype: {cloud_masked.dtype}')
            print(f'[*] Color masked shape: {color_masked.shape}, dtype: {color_masked.dtype}, min: {color_masked.min()}, max: {color_masked.max()}')
            try:
                vis_cloud = o3d.geometry.PointCloud()
                vis_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
                vis_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
                print('[*] Visualization point cloud created')
                
                # Save input point cloud to PLY file
                # Create output directory (relative to current working directory)
                output_dir = '../output'
                os.makedirs(output_dir, exist_ok=True)
                timestamp = int(time.time())
                ply_filename = f'{output_dir}/input_pointcloud_{timestamp}.ply'
                o3d.io.write_point_cloud(ply_filename, vis_cloud)
                print(f'[*] Saved input point cloud to: {ply_filename}')
            except Exception as e:
                print(f'[*] ERROR creating visualization point cloud: {e}')
                raise
            
            # Get grasps
            print('[*] Running network inference...')
            with torch.no_grad():
                end_points = self.net(end_points)
                
                # Debug: check objectness scores
                objectness_score = end_points['objectness_score'][0]
                # objectness_score shape is (2, num_seed) where 2 classes: [background, graspable]
                objectness_pred = torch.argmax(objectness_score, 0)
                
                # Check raw scores too
                positive_scores = objectness_score[1, :]  # scores for "graspable" class
                max_positive_score = positive_scores.max().item()
                mean_positive_score = positive_scores.mean().item()
                num_positive = (objectness_pred == 1).sum().item()
                print(f'[*] Objectness prediction: {num_positive} points classified as graspable out of {len(objectness_pred)}')
                print(f'[*] Objectness positive scores: max={max_positive_score:.4f}, mean={mean_positive_score:.4f}')
                
                # Try with a threshold instead of argmax
                threshold_positive = (positive_scores > 0.0).sum().item()
                print(f'[*] Points with positive score > 0: {threshold_positive}')
                
                # Debug: check grasp scores
                if 'grasp_score_pred' in end_points:
                    grasp_scores = end_points['grasp_score_pred'][0]
                    print(f'[*] Grasp score range: min={grasp_scores.min():.4f}, max={grasp_scores.max():.4f}, mean={grasp_scores.mean():.4f}')
                
                grasp_preds = pred_decode(end_points)
            print('[*] Network inference complete')
            
            gg_array = grasp_preds[0].detach().cpu().numpy()
            gg = GraspGroup(gg_array)
            
            print(f'[*] Generated {len(gg)} grasps')
            print(f'[*] Point cloud size: {len(cloud_masked)} points')
            
            # Print grasp details before collision detection
            if len(gg) > 0:
                print(f'[*] Best grasp before collision detection: score={gg[0].score:.4f}, width={gg[0].width:.4f}, depth={gg[0].depth:.4f}')
            
            # Collision detection
            if cfgs.no_collision:
                print('[*] Collision detection DISABLED (--no_collision flag)')
            elif cfgs.collision_thresh > 0:
                print(f'[*] Running collision detection with thresh={cfgs.collision_thresh}...')
                mfcdetector = ModelFreeCollisionDetector(cloud_masked, voxel_size=cfgs.voxel_size)
                collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
                gg_filtered = gg[~collision_mask]
                print(f'[*] {len(gg_filtered)} grasps after collision detection (original: {len(gg)})')
                gg = gg_filtered
            
            # Visualize (optional, can be done in headless_viz mode)
            if not cfgs.headless:
                print(f'[*] Starting visualization...')
                self.visualize_grasps(gg, vis_cloud)
            
            # Prepare grasp results for publishing
            gg.nms()
            gg.sort_by_score()
            gg = gg[:50]  # Keep top 50
            
            # Calculate processing time
            processing_time = time.time() - start_time
            print(f'[*] GraspNet processing time: {processing_time:.4f} seconds')
            
            # Series best grasp (highest score)
            if len(gg) == 0:
                print('[*] No grasps found after filtering')
                return None
            
            best_grasp = gg[0]
            
            # Prepare data structure for publishing thread
            grasp_data = {
                'grasp': best_grasp,
                'processing_time': processing_time,
                'timestamp': time.time()
            }
            
            print('[*] Processing complete successfully!')
            print(f'[*] Best grasp - score: {best_grasp.score:.4f}, width: {best_grasp.width:.4f}')
            return grasp_data
            
        except Exception as e:
            print(f'\n[*] ERROR processing frame: {e}')
            import traceback
            print(traceback.format_exc())
            self.get_logger().error(f'Error processing frame: {e}')
            self.get_logger().error(traceback.format_exc())
            return None
    
    def visualize_grasps(self, gg, cloud):
        """Visualize predicted grasps."""
        print(f'[*] Before NMS: {len(gg)} grasps')
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        
        print(f'[*] Top grasps after NMS: {len(gg)}')
        if len(gg) > 0:
            print(f'[*] Best grasp score: {gg[0].score:.4f}')
            print(f'[*] Best grasp translation: {gg[0].translation}')
            print(f'[*] Best grasp rotation matrix shape: {gg[0].rotation_matrix.shape}')
        
        grippers = gg.to_open3d_geometry_list()
        
        # Check for headless mode
        headless = cfgs.headless or os.environ.get('OPEN3D_HEADLESS', '').lower() in ('1', 'true', 'yes')
        
        if headless:
            print('[*] Running in headless mode - skipping visualization')
            print(f'[*] Final: {len(gg)} grasps (top 50 after NMS)')
        else:
            print(f'[*] Visualizing {len(gg)} grasps (this may take a moment)')
            try:
                o3d.visualization.draw_geometries([cloud, *grippers])
            except Exception as e:
                print(f'[*] Visualization error: {e}')
                print('[*] Switching to headless mode')
        print('[*] Visualization complete')


def get_net():
    """Initialize the GraspNet model."""
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def main():
    # Initialize ROS2
    rclpy.init()
    
    node = None
    try:
        # Load the model
        print("[*] Loading GraspNet model...")
        net = get_net()
        
        # Create ROS2 node
        node = GraspNetRos2Node(net)
        
        # Continuous processing with multithreading
        print("[*] Starting continuous processing (press Ctrl+C to stop)...")
        print(f"[*] Processing frame every {cfgs.processing_interval} seconds")
        print("[*] Grasp results will be published to:", cfgs.grasp_topic)
        
        # Spin ROS2 node (handles callbacks)
        rclpy.spin(node)
    
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    except Exception as e:
        print(f"\n[*] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            # Signal threads to stop
            node.running = False
            # Wait for threads to finish
            if node.processing_thread is not None:
                node.processing_thread.join(timeout=2.0)
            if node.publishing_thread is not None:
                node.publishing_thread.join(timeout=2.0)
            node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    # Import cv2
    import cv2
    
    main()

