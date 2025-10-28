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
from cv_bridge import CvBridge
import threading
import time

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
parser.add_argument('--process_once', action='store_true', help='Process only one frame then exit')
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
        self.lock = threading.Lock()
        self.processed = False
        self.processing = False
        
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
        
        # Create timer for continuous processing
        if not cfgs.process_once:
            self.timer = self.create_timer(2.0, self.timer_callback)
        
        self.get_logger().info('[*] GraspNet ROS2 node started')
        self.get_logger().info(f'[*] Subscribing to: {cfgs.color_topic}, {cfgs.depth_topic}, {cfgs.camera_info_topic}')
    
    def timer_callback(self):
        """Timer callback for continuous processing."""
        if not self.processing:
            self.process_frame()
    
    def color_callback(self, msg):
        """Callback for color image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.color_image = cv_image
                self.get_logger().info(f'Received color image: {cv_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error converting color image: {e}')
        
        # If process_once mode, try to process frame when all data is ready
        if cfgs.process_once and not self.processed and not self.processing:
            with self.lock:
                if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
                    # Trigger processing in background to avoid blocking callbacks
                    thread = threading.Thread(target=self.process_frame)
                    thread.daemon = True
                    thread.start()
    
    def depth_callback(self, msg):
        """Callback for depth image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            with self.lock:
                self.depth_image = cv_image
                self.get_logger().info(f'Received depth image: {cv_image.shape}')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
        
        # If process_once mode, try to process frame when all data is ready
        if cfgs.process_once and not self.processed and not self.processing:
            with self.lock:
                if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
                    # Trigger processing in background to avoid blocking callbacks
                    thread = threading.Thread(target=self.process_frame)
                    thread.daemon = True
                    thread.start()
    
    def camera_info_callback(self, msg):
        """Callback for camera info."""
        with self.lock:
            self.camera_info = msg
            self.get_logger().info(f'Received camera info: {msg.width}x{msg.height}')
        
        # If process_once mode, try to process frame when all data is ready
        if cfgs.process_once and not self.processed and not self.processing:
            with self.lock:
                if self.color_image is not None and self.depth_image is not None and self.camera_info is not None:
                    # Trigger processing in background to avoid blocking callbacks
                    thread = threading.Thread(target=self.process_frame)
                    thread.daemon = True
                    thread.start()
    
    def process_frame(self):
        """Process one frame if all data is ready."""
        with self.lock:
            if self.color_image is None or self.depth_image is None or self.camera_info is None:
                return False
            
            if self.processed and cfgs.process_once:
                return True
            
            if self.processing:
                return False  # Already processing
            
            # Make copies
            color_img = self.color_image.copy()
            depth_img = self.depth_image.copy()
            cam_info = self.camera_info
            # Don't set self.processed = True here! Wait until processing is complete
            self.processing = True
        
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
                with self.lock:
                    self.processing = False
                    self.processed = True  # Mark as processed even if failed
                return False
            
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
                import os
                import time
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
            
            # Visualize
            print(f'[*] Starting visualization...')
            self.visualize_grasps(gg, vis_cloud)
            
            print('[*] Processing complete successfully!')
            with self.lock:
                self.processing = False
                self.processed = True  # Mark as processed after successful completion
            return True
            
        except Exception as e:
            print(f'\n[*] ERROR processing frame: {e}')
            import traceback
            print(traceback.format_exc())
            self.get_logger().error(f'Error processing frame: {e}')
            self.get_logger().error(traceback.format_exc())
            with self.lock:
                self.processing = False
                self.processed = True  # Mark as processed even on error
            return False
    
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
        
        if cfgs.process_once:
            # Process one frame
            print("[*] Waiting for camera data...")
            import time
            start_time = time.time()
            timeout = 30.0  # 30 seconds timeout
            
            while not node.processed:
                rclpy.spin_once(node, timeout_sec=0.1)
                
                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    print(f"\n[*] Timeout after {timeout}s waiting for camera data")
                    print("[*] Please check if camera topics are publishing:")
                    print("    - /camera/color/image_raw")
                    print("    - /camera/depth/image_raw")
                    print("    - /camera/color/camera_info")
                    break
                
                # Print status every 2 seconds
                if int(elapsed) % 2 == 0 and int(elapsed) > 0:
                    if node.color_image is None:
                        print(f"[*] Waiting for color image... ({int(elapsed)}s)")
                    elif node.depth_image is None:
                        print(f"[*] Waiting for depth image... ({int(elapsed)}s)")
                    elif node.camera_info is None:
                        print(f"[*] Waiting for camera info... ({int(elapsed)}s)")
            
            if node.processed:
                print("[*] Frame processed, exiting...")
        else:
            # Continuous processing
            print("[*] Starting continuous processing (press Ctrl+C to stop)...")
            print("[*] Processing frame every 2 seconds")
            rclpy.spin(node)
    
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
    except Exception as e:
        print(f"\n[*] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__=='__main__':
    # Import cv2
    import cv2
    
    main()

