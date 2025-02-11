#!/usr/bin/env python3
"""
PCA Visualization with Pygame, Interactive Camera Rotation/Zoom, and SLERP Interpolation
This script loads a PCA mapping JSON file generated from RAVE latent space analysis.
If the JSON contains 2D coordinates (only "x" and "y"), it displays the data in 2D.
If the JSON contains 3D coordinates ("x", "y", and "z"), it displays a full 3D perspective
with interactive camera rotation (right-click drag), zoom (mouse scroll), and smooth SLERP
interpolation of latent vectors (sent via OSC).

Author: Moises Horta Valenzuela (modified for 2D/3D auto-detection)
"""

import pygame
import json
import math
import threading
import numpy as np
from pythonosc import dispatcher, osc_server, udp_client
import os
import argparse

# ---------------------
# Global Variables
# ---------------------
bubbles = []           # List to hold all Bubble objects.
DATA_IS_3D = False     # Will be determined from the JSON (True if "z" is present).

# Global cursor position (updated by the mouse)
cursor_x = 0.0
cursor_y = 0.0

# OSC client for sending OSC messages.
osc_client = None

# Screen dimensions.
WIDTH, HEIGHT = 1500, 1500

# Camera parameters for perspective projection.
CAMERA_DISTANCE = 300    # Camera's distance from the projection plane.
SCREEN_CENTER = (WIDTH / 2, HEIGHT / 2)

# Scaling factors for PCA coordinates.
SCALE_FACTOR = 100     # Scale for x and y.
Z_SCALE_FACTOR = 100   # Scale for z (only used for 3D data).
Z_OFFSET = 300         # Offset added to z to ensure bubbles appear in front of the camera.

# Global camera rotation parameters (in radians)
camera_yaw = 0.0    # Rotation around y-axis (left/right).
camera_pitch = 0.0  # Rotation around x-axis (up/down).

# Variables to handle right-click dragging for camera rotation.
rotating = False
prev_mouse_x = None
prev_mouse_y = None
ROTATION_SENSITIVITY = 0.01  # Controls how fast the view rotates.

# Variables for zooming using the mouse scroll wheel.
ZOOM_STEP = 50         # Amount to change CAMERA_DISTANCE per scroll event.
MIN_CAMERA_DISTANCE = 1  # Minimum allowed CAMERA_DISTANCE.

# Variables for spherical interpolation between latent vectors.
current_latent = None  # The current latent vector (list of floats).
target_latent = None   # The target latent vector (list of floats).
INTERP_SPEED = 0.999   # Interpolation factor (0.0 to 1.0 per frame).

# ---------------------
# Helper Function: SLERP
# ---------------------
def slerp(v0, v1, t):
    """
    Perform spherical linear interpolation between two vectors v0 and v1.
    t is the interpolation factor between 0 and 1.
    Returns the interpolated vector as a list of floats.
    """
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)
    # If either vector is near zero, fall back to linear interpolation.
    if norm0 < 1e-6 or norm1 < 1e-6:
        return ((1-t) * v0 + t * v1).tolist()
    # Normalize the vectors.
    v0_unit = v0 / norm0
    v1_unit = v1 / norm1
    dot = np.clip(np.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-6:
        interp = (1-t)*v0 + t*v1
        return interp.tolist()
    factor0 = math.sin((1-t)*theta) / sin_theta
    factor1 = math.sin(t*theta) / sin_theta
    interp_unit = factor0 * v0_unit + factor1 * v1_unit
    # Optionally interpolate the magnitudes linearly.
    interp_norm = (1-t)*norm0 + t*norm1
    return (interp_unit * interp_norm).tolist()

# ---------------------
# Helper Function for 3D Rotation
# ---------------------
def rotate_point(x, y, z, yaw, pitch):
    """
    Rotate a 3D point (x, y, z) by yaw (around y-axis) and pitch (around x-axis).
    Returns the rotated (x, y, z).
    """
    # Yaw rotation (around y-axis)
    x1 = math.cos(yaw)*x + math.sin(yaw)*z
    y1 = y
    z1 = -math.sin(yaw)*x + math.cos(yaw)*z
    # Pitch rotation (around x-axis)
    y2 = math.cos(pitch)*y1 - math.sin(pitch)*z1
    z2 = math.sin(pitch)*y1 + math.cos(pitch)*z1
    return x1, y2, z2

# ---------------------
# Bubble Class Definition
# ---------------------
class Bubble:
    def __init__(self, x, y, z, diameter, latent):
        """
        x, y, z: 3D coordinates (after scaling/offset) of the bubble.
                For 2D data, z should be 0.
        diameter: Base diameter in pixels (before perspective scaling).
        latent: The latent coordinate list.
        """
        self.x = x
        self.y = y
        self.z = z
        self.base_diameter = diameter
        self.latent = list(latent)
        self.over = False

    def project(self, camera_distance, screen_center, yaw, pitch):
        """
        Rotate the bubble's coordinates and apply perspective projection.
        If the data is 2D (z==0 and no "z" key was provided), yaw and pitch are ignored.
        Returns the projected (x, y) and scaled diameter.
        """
        if not DATA_IS_3D:
            # For 2D data, simply shift by the screen center.
            proj_x = screen_center[0] + self.x
            proj_y = screen_center[1] - self.y
            proj_diameter = self.base_diameter
            return proj_x, proj_y, proj_diameter

        # For 3D data, apply rotation and perspective.
        x_rot, y_rot, z_rot = rotate_point(self.x, self.y, self.z, yaw, pitch)
        factor = camera_distance / (camera_distance + z_rot)
        proj_x = screen_center[0] + x_rot * factor
        proj_y = screen_center[1] - y_rot * factor  # Invert y for screen coordinates.
        proj_diameter = self.base_diameter * factor
        return proj_x, proj_y, proj_diameter

    def rollover(self, px, py, camera_distance, screen_center, yaw, pitch):
        """
        Check if the point (px, py) is over the projected bubble.
        """
        proj_x, proj_y, proj_diameter = self.project(camera_distance, screen_center, yaw, pitch)
        d = math.hypot(proj_x - px, proj_y - py)
        self.over = d < (proj_diameter / 2)
        return self.over

    def display(self, surface, camera_distance, screen_center, yaw, pitch):
        """
        Draw the bubble on the given surface using perspective projection (or 2D if applicable).
        """
        proj_x, proj_y, proj_diameter = self.project(camera_distance, screen_center, yaw, pitch)
        if self.over:
            color = (0, 102, 0)
            pygame.draw.circle(surface, color, (int(proj_x), int(proj_y)), int(proj_diameter / 2))
        else:
            color = (255, 255, 255)
            pygame.draw.circle(surface, color, (int(proj_x), int(proj_y)), int(proj_diameter / 2), 1)

# ---------------------
# OSC Callback Functions
# ---------------------
def sensor_gx_handler(unused_addr, *args):
    global cursor_x
    try:
        first_value = float(args[0])
        print("Received /sensor/gx:", first_value)
        cursor_x += first_value
    except Exception as e:
        print("Error in sensor_gx_handler:", e)

def sensor_gy_handler(unused_addr, *args):
    global cursor_y
    try:
        first_value = float(args[0])
        print("Received /sensor/gy:", first_value)
        cursor_y += first_value
    except Exception as e:
        print("Error in sensor_gy_handler:", e)

def default_handler(addr, *args):
    print("### Received an OSC message with address pattern", addr, "and args", args)

def start_osc_server():
    """
    Set up and run the OSC server on port 12000 in a separate thread.
    """
    disp = dispatcher.Dispatcher()
    disp.map("/sensor/gx", sensor_gx_handler)
    disp.map("/sensor/gy", sensor_gy_handler)
    disp.set_default_handler(default_handler)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 12000), disp)
    print("OSC Server is running on {}".format(server.server_address))
    server.serve_forever()

# ---------------------
# Initialization and JSON Loading
# ---------------------
def load_bubbles(json_filename):
    """
    Loads the JSON file and creates Bubble objects.
    The JSON is expected to contain a key "pca", which is a list of objects.
    Each object should have a "PCA coordinate" dictionary.
    For 3D data, the dictionary should include keys "x", "y", and "z".
    For 2D data, only "x" and "y" are expected.
    """
    global bubbles, DATA_IS_3D
    if not os.path.exists(json_filename):
        print(f"JSON file not found: {json_filename}")
        return

    try:
        with open(json_filename, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print("Error loading JSON file:", e)
        return

    pca_values = json_data.get("pca", [])
    print("Number of objects in JSON:", len(pca_values))
    bubbles = []  # Clear any existing bubbles.

    # Auto-detect whether the data is 3D (if "z" is present in the first object's PCA coordinate)
    if pca_values:
        coord = pca_values[0].get("PCA coordinate", {})
        DATA_IS_3D = ("z" in coord)
    else:
        DATA_IS_3D = False

    for pca in pca_values:
        position = pca.get("PCA coordinate", {})
        x_val = float(position.get("x", 0)) * SCALE_FACTOR
        y_val = float(position.get("y", 0)) * SCALE_FACTOR
        if DATA_IS_3D:
            z_val = float(position.get("z", 0)) * Z_SCALE_FACTOR + Z_OFFSET
        else:
            z_val = 0  # For 2D data, set z to 0.
        latent = pca.get("Latent coordinate", [])
        try:
            latent_floats = [float(v) for v in latent]
        except Exception as e:
            print("Error parsing latent coordinates:", e)
            latent_floats = []
        bubble = Bubble(x_val, y_val, z_val, 4, latent_floats)
        bubbles.append(bubble)
    print("Done loading bubbles. Loaded {} bubbles as {}D data.".format(len(bubbles), "3D" if DATA_IS_3D else "2D"))

# ---------------------
# Main Loop using Pygame
# ---------------------
def main():
    global cursor_x, cursor_y, osc_client, rotating, prev_mouse_x, prev_mouse_y
    global camera_yaw, camera_pitch, CAMERA_DISTANCE, current_latent, target_latent

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="PCA Visualization with Rotation, Zoom, and SLERP. Loads a PCA JSON mapping (2D or 3D)."
    )
    parser.add_argument(
        "--pca_json",
        type=str,
        default="pca_latent_mapping_reza_stereo.json",
        help="Path to the PCA map JSON file."
    )
    args = parser.parse_args()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("PCA Visualization with Rotation, Zoom, and SLERP")
    clock = pygame.time.Clock()

    # Set initial cursor position.
    cursor_x = WIDTH / 2
    cursor_y = HEIGHT / 2

    # Load bubbles from the specified JSON file.
    load_bubbles(args.pca_json)

    # Initialize OSC client.
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9999)

    # Start the OSC server in a separate daemon thread.
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    # Initialize current_latent and target_latent with the first bubble's latent (if available).
    if bubbles:
        current_latent = bubbles[0].latent
        target_latent = bubbles[0].latent

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # Handle mouse button events.
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Right mouse button starts camera rotation.
                if event.button == 3:
                    rotating = True
                    prev_mouse_x, prev_mouse_y = event.pos
                # Scroll up (button 4) to zoom in.
                elif event.button == 4:
                    CAMERA_DISTANCE = max(MIN_CAMERA_DISTANCE, CAMERA_DISTANCE - ZOOM_STEP)
                # Scroll down (button 5) to zoom out.
                elif event.button == 5:
                    CAMERA_DISTANCE += ZOOM_STEP

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    rotating = False
                    prev_mouse_x, prev_mouse_y = None, None

            elif event.type == pygame.MOUSEMOTION:
                cursor_x, cursor_y = event.pos
                if rotating and prev_mouse_x is not None and prev_mouse_y is not None:
                    dx = event.pos[0] - prev_mouse_x
                    dy = event.pos[1] - prev_mouse_y
                    camera_yaw += dx * ROTATION_SENSITIVITY
                    camera_pitch += dy * ROTATION_SENSITIVITY
                    prev_mouse_x, prev_mouse_y = event.pos

        # --- SLERP Interpolation for latent vectors ---
        hovered_latent = None
        for b in bubbles:
            # For 2D data, use no rotation.
            if b.rollover(cursor_x, cursor_y, CAMERA_DISTANCE, SCREEN_CENTER,
                          camera_yaw if DATA_IS_3D else 0,
                          camera_pitch if DATA_IS_3D else 0):
                hovered_latent = b.latent
                break

        if hovered_latent is not None:
            target_latent = hovered_latent

        if current_latent is not None and target_latent is not None:
            current_latent = slerp(current_latent, target_latent, INTERP_SPEED)
            osc_client.send_message("/latent", current_latent)

        # Clear the screen.
        screen.fill((0, 0, 0))

        # Display each bubble.
        for b in bubbles:
            b.display(screen, CAMERA_DISTANCE, SCREEN_CENTER,
                      camera_yaw if DATA_IS_3D else 0,
                      camera_pitch if DATA_IS_3D else 0)

        # Draw an indicator at the mouse position.
        indicator_color = (0, 202, 0)
        pygame.draw.circle(screen, indicator_color, (int(cursor_x), int(cursor_y)), 10)

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == '__main__':
    main()
