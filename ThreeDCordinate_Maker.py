import pandas as pd
import numpy as np
import scipy.io
import cv2
from scipy.io import loadmat

def refract_points(pts_pixel, K, D, n_air=1.0, n_water=1.333):
    """
    Converts Pixel Coordinates -> Normalized Water Coordinates.
    Corrects for lens distortion AND flat port refraction.
    """
    if len(pts_pixel) == 0:
        return np.array([])

    # Undistort to Normalized Air Coordinates
    # resulting shape is (N, 2)
    pts_norm_air = cv2.undistortPoints(np.expand_dims(pts_pixel, 1), K, D).squeeze(1)

    # Apply Snell's Law (Flat Port Refraction)
    r_air = np.linalg.norm(pts_norm_air, axis=1)
    mask = r_air > 1e-8
    
    pts_norm_water = pts_norm_air.copy()
    
    if np.any(mask):
        theta_air = np.arctan(r_air[mask])
        
        # Snell's Law: n_air * sin(theta_a) = n_water * sin(theta_w)
        sin_theta_water = (n_air / n_water) * np.sin(theta_air)
        
        valid = np.abs(sin_theta_water) <= 1.0
        
        theta_water = np.arcsin(sin_theta_water[valid])
        r_water = np.tan(theta_water)
        
        # Scale factor = r_new / r_old
        scale = r_water / r_air[mask][valid]
        
        # Apply scaling to the masked points
        pts_norm_water[mask] = pts_norm_air[mask] * scale[:, np.newaxis]

    return pts_norm_water

def cor_maker_3d(df1_path, df2_path, camera_data_file, n_water=1.333):
    """
    Generates 3D coordinates from matched stereo files.
    SAFE MODE: Enforces strict 1-to-1 matching per frame to prevent crashes.
    """
    print(f"Processing 3D reconstruction for {df1_path} and {df2_path}...")
    
    # 1. Load Data
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    
    # 2. Safety: Remove duplicate IDs in the same frame (tracker glitches)
    df1 = df1.drop_duplicates(subset=['frame', 'id'])
    df2 = df2.drop_duplicates(subset=['frame', 'id'])
    
    try:
        dd = loadmat(camera_data_file)
    except Exception as e:
        print(f"Error loading MAT file: {e}")
        return pd.DataFrame()
    
    # 3. Extract Camera Parameters
    K1 = dd["intrinsicMatrix1"]
    K2 = dd["intrinsicMatrix2"]
    D1 = dd["distortionCoefficients1"]
    D2 = dd["distortionCoefficients2"]
    R = dd["rotationOfCamera2"]
    T = dd["translationOfCamera2"]
    
    # Setup Projection Matrices for Normalized Space
    # Since refract_points returns normalized coordinates (unit plane), 
    # we treat K as Identity for the triangulation step.
    P1 = np.hstack((np.eye(3), np.zeros((3, 1))))      # [I | 0]
    P2 = np.hstack((R, T.reshape(3, 1)))               # [R | t]
    
    # 4. Iterate Common Frames
    common_frames = sorted(list(set(df1['frame']) & set(df2['frame'])))
    results = []

    for frame in common_frames:
        f1 = df1[df1['frame'] == frame]
        f2 = df2[df2['frame'] == frame]
        
        # --- CRITICAL FIX: Merge on ID ---
        # This performs an INNER JOIN. 
        # Only IDs present in BOTH frames are kept.
        # This guarantees len(pts1) == len(pts2) automatically.
        merged = pd.merge(f1, f2, on='id', suffixes=('_1', '_2'))
        
        if merged.empty:
            continue
            
        # Extract aligned coordinates
        pts1_pix = merged[['x_1', 'y_1']].values.astype(np.float32)
        pts2_pix = merged[['x_2', 'y_2']].values.astype(np.float32)
        ids = merged['id'].values
        
        # Apply Refraction Correction
        pts1_norm = refract_points(pts1_pix, K1, D1, n_air=1.0, n_water=n_water)
        pts2_norm = refract_points(pts2_pix, K2, D2, n_air=1.0, n_water=n_water)
        
        # Transpose for OpenCV (2, N)
        pts1_norm_T = pts1_norm.T
        pts2_norm_T = pts2_norm.T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1_norm_T, pts2_norm_T)
        
        # Convert Homogeneous -> Euclidean (X/W, Y/W, Z/W)
        # Avoid division by zero
        w = points_4d[3]
        w[w == 0] = 1e-10
        points_3d = points_4d[:3] / w
        
        # Store results
        points_3d = points_3d.T # (N, 3)
        
        for i, obj_id in enumerate(ids):
            results.append({
                'frame': frame,
                'id': int(obj_id),
                'x': points_3d[i, 0],
                'y': points_3d[i, 1],
                'z': points_3d[i, 2]
            })

    # Create final DataFrame
    df_3d = pd.DataFrame(results)
    return df_3d