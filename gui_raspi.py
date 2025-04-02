#!/usr/bin/env python3
"""
PCA Visualization with Pygame, Interactive Camera Rotation/Zoom, SLERP Interpolation,
Display of Selected Coordinates and Latent Vector, and an Orbit Feature with a SLERP Orbit Speed Slider

This script loads a PCA mapping JSON file generated from RAVE latent space analysis.
If the JSON contains 2D coordinates (only "x" and "y"), it displays the data in 2D.
If the JSON contains 3D coordinates ("x", "y", and "z"), it displays a full 3D perspective
with interactive camera rotation, zoom, and smooth SLERP interpolation of latent vectors (sent via OSC).

Additionally, the user can double-click on bubbles to select them for an "orbit". When 2 or more
bubbles are selected, the program continuously interpolates between the selected points, drawing a
purple trajectory and a moving marker along the orbit. Two sets of sliders are provided:
  • The original orbit speed (and smoothness) slider, now repositioned inside the visible circle.
  • Three new vertical sliders (for yaw, pitch, and distance) are placed along the center–left of the screen.
  
A new record button is drawn as a simple circle at the top center. When the user clicks the record button,
recording mode toggles; while active, as the user drags the mouse, any hovered bubble is recorded.
When recording stops, the recorded trajectory is used to create the orbit.

Finally, the latent vector of the last–selected bubble is displayed as a vertical column on the right side,
starting at the vertical center of the 3D control sliders (y = 400) and 90 pixels from the right edge.

Press F11 to toggle fullscreen mode. In fullscreen the mouse cursor is hidden.

Author: Moisés Horta Valenzuela
"""

import os
import pygame
import orjson
import math
import threading
import numpy as np
from pythonosc import dispatcher, osc_server, udp_client
import argparse

# ---------------------
# Screen Dimensions and Virtual Resolution
# ---------------------
screen_width = 800
screen_height = 800
DEFAULT_WIDTH, DEFAULT_HEIGHT = 800, 800
VIRTUAL_WIDTH, VIRTUAL_HEIGHT = DEFAULT_WIDTH, DEFAULT_HEIGHT
WIDTH, HEIGHT = VIRTUAL_WIDTH, VIRTUAL_HEIGHT

# ---------------------
# Global Variables
# ---------------------
bubbles = []           # List to hold all Bubble objects.
DATA_IS_3D = False     # Determined from the JSON

cursor_x = 0.0         # Global cursor position.
cursor_y = 0.0

osc_client = None      # OSC client.

# Camera parameters.
CAMERA_DISTANCE = 300
SCREEN_CENTER = (WIDTH // 2, HEIGHT // 2)
MIN_CAMERA_DISTANCE = 1
MAX_CAMERA_DISTANCE = 1000

# Scaling factors.
SCALE_FACTOR = 100
Z_SCALE_FACTOR = 100
Z_OFFSET = 300

# Camera rotation.
camera_yaw = 0.0    # In radians.
camera_pitch = 0.0  # In radians.

# Drag variables.
rotating = False
prev_mouse_x = None
prev_mouse_y = None
ROTATION_SENSITIVITY = 0.01

# SLERP interpolation.
current_latent = None
target_latent = None
INTERP_SPEED = 0.999

# Orbit feature.
orbit_points = []      # Orbit-selected bubbles.
orbit_index = 0
orbit_t = 0.0
last_click_time = 0
double_click_threshold = 300  # ms

# ---------------------
# Orbit Speed and Smoothness Sliders
# ---------------------
ORBIT_SLIDER_WIDTH = 200
ORBIT_SLIDER_HEIGHT = 20
ORBIT_SLIDER_X = (VIRTUAL_WIDTH - ORBIT_SLIDER_WIDTH) // 2  # 300
ORBIT_SLIDER_Y = 700  # Inside the round area.
orbit_slider_active = False
orbit_slider_value = 0.5
ORBIT_SPEED_MIN = 0.001
ORBIT_SPEED_MAX = 25.0
orbit_speed = orbit_slider_value * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN) + ORBIT_SPEED_MIN

NORMAL_SLIDER_WIDTH = 200
NORMAL_SLIDER_HEIGHT = 20
NORMAL_SLIDER_X = (VIRTUAL_WIDTH - NORMAL_SLIDER_WIDTH) // 2  # 300
NORMAL_SLIDER_Y = 700
normal_slider_active = False
normal_slider_value = 0.99

# ---------------------
# Camera Control Sliders (vertical, center-left)
# ---------------------
CAMERA_YAW_SLIDER_X = 90
CAMERA_YAW_SLIDER_Y = 155
CAMERA_YAW_SLIDER_WIDTH = 20
CAMERA_YAW_SLIDER_HEIGHT = 150

CAMERA_PITCH_SLIDER_X = 90
CAMERA_PITCH_SLIDER_Y = 325
CAMERA_PITCH_SLIDER_WIDTH = 20
CAMERA_PITCH_SLIDER_HEIGHT = 150

CAMERA_DIST_SLIDER_X = 90
CAMERA_DIST_SLIDER_Y = 495
CAMERA_DIST_SLIDER_WIDTH = 20
CAMERA_DIST_SLIDER_HEIGHT = 150

yaw_slider_active = False
pitch_slider_active = False
dist_slider_active = False

yaw_slider_value = 0.5
pitch_slider_value = 0.5
dist_slider_value = (300 - MIN_CAMERA_DISTANCE) / (MAX_CAMERA_DISTANCE - MIN_CAMERA_DISTANCE)

# ---------------------
# Record Button (New Feature)
# ---------------------
# Drawn as a circle at the top center. It should be green when not recording and purple when recording.
RECORD_BTN_RADIUS = 50
RECORD_BTN_CENTER = (VIRTUAL_WIDTH // 2, 50)

recording = False
recorded_points = []  # List to store bubbles recorded while dragging.

# ---------------------
# Fullscreen and Scaling Variables
# ---------------------
fullscreen = False
FS_SCALE_FACTOR = 1.0
FS_OFFSET_X = 0
FS_OFFSET_Y = 0

# ---------------------
# Helper Functions
# ---------------------
def slerp(v0, v1, t):
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    norm0 = np.linalg.norm(v0)
    norm1 = np.linalg.norm(v1)
    if norm0 < 1e-6 or norm1 < 1e-6:
        return ((1-t) * v0 + t * v1).tolist()
    v0_unit = v0 / norm0
    v1_unit = v1 / norm1
    dot = np.clip(np.dot(v0_unit, v1_unit), -1.0, 1.0)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if sin_theta < 1e-6:
        return ((1-t)*v0 + t*v1).tolist()
    factor0 = math.sin((1-t)*theta) / sin_theta
    factor1 = math.sin(t*theta) / sin_theta
    interp_unit = factor0 * v0_unit + factor1 * v1_unit
    interp_norm = (1-t)*norm0 + t*norm1
    return (interp_unit * interp_norm).tolist()

def slerp_nd(v0, v1, t):
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-6:
        return ((1-t)*v0 + t*v1).tolist()
    return ((math.sin((1-t)*theta)/sin_theta)*v0 + (math.sin(t*theta)/sin_theta)*v1).tolist()

def rotate_point(x, y, z, yaw, pitch):
    x1 = math.cos(yaw)*x + math.sin(yaw)*z
    y1 = y
    z1 = -math.sin(yaw)*x + math.cos(yaw)*z
    y2 = math.cos(pitch)*y1 - math.sin(pitch)*z1
    z2 = math.sin(pitch)*y1 + math.cos(pitch)*z1
    return x1, y2, z2

# ---------------------
# Bubble Class
# ---------------------
class Bubble:
    def __init__(self, x, y, z, diameter, latent):
        self.x = x
        self.y = y
        self.z = z
        self.base_diameter = diameter
        self.latent = list(latent)
        self.over = False
        self.orbit_selected = False

    def project(self, camera_distance, screen_center, cos_yaw, sin_yaw, cos_pitch, sin_pitch):
        if not DATA_IS_3D:
            proj_x = screen_center[0] + self.x
            proj_y = screen_center[1] - self.y
            proj_diameter = self.base_diameter
            return proj_x, proj_y, proj_diameter
        # Use precomputed cosine and sine values for performance.
        x_rot = cos_yaw * self.x + sin_yaw * self.z
        y_rot = self.y
        z_rot = -sin_yaw * self.x + cos_yaw * self.z
        # Apply pitch rotation:
        y_rot2 = cos_pitch * y_rot - sin_pitch * z_rot
        z_rot2 = sin_pitch * y_rot + cos_pitch * z_rot
        denom = camera_distance + z_rot2
        factor = camera_distance / denom if denom != 0 else 1
        proj_x = screen_center[0] + x_rot * factor
        proj_y = screen_center[1] - y_rot2 * factor
        proj_diameter = self.base_diameter * factor
        return proj_x, proj_y, proj_diameter

    def rollover(self, px, py, camera_distance, screen_center, cos_yaw, sin_yaw, cos_pitch, sin_pitch):
        proj_x, proj_y, proj_diameter = self.project(camera_distance, screen_center, cos_yaw, sin_yaw, cos_pitch, sin_pitch)
        d = math.hypot(proj_x - px, proj_y - py)
        self.over = d < (proj_diameter / 2)
        return self.over

    def display(self, surface, camera_distance, screen_center, cos_yaw, sin_yaw, cos_pitch, sin_pitch):
        proj_x, proj_y, proj_diameter = self.project(camera_distance, screen_center, cos_yaw, sin_yaw, cos_pitch, sin_pitch)
        # Offscreen culling: skip drawing if completely outside a margin.
        margin = 50
        if (proj_x + proj_diameter/2 < -margin or proj_x - proj_diameter/2 > VIRTUAL_WIDTH + margin or
            proj_y + proj_diameter/2 < -margin or proj_y - proj_diameter/2 > VIRTUAL_HEIGHT + margin):
            return
        try:
            cx = int(proj_x)
            cy = int(proj_y)
            radius = int(proj_diameter / 2)
        except Exception as e:
            print("Conversion error in display:", e)
            return
        if radius <= 0:
            return
        if self.orbit_selected:
            color = (128, 0, 128)
            pygame.draw.circle(surface, color, (cx, cy), radius)
        elif self.over:
            color = (0, 102, 0)
            pygame.draw.circle(surface, color, (cx, cy), radius)
        else:
            color = (255, 255, 255)
            pygame.draw.circle(surface, color, (cx, cy), radius, 1)

# ---------------------
# OSC Callbacks
# ---------------------
def sensor_gx_handler(unused_addr, *args):
    global cursor_x
    try:
        cursor_x += float(args[0])
    except Exception as e:
        print("Error in sensor_gx_handler:", e)

def sensor_gy_handler(unused_addr, *args):
    global cursor_y
    try:
        cursor_y += float(args[0])
    except Exception as e:
        print("Error in sensor_gy_handler:", e)

def default_handler(addr, *args):
    print("### Received an OSC message with address pattern", addr, "and args", args)

def start_osc_server():
    disp = dispatcher.Dispatcher()
    disp.map("/sensor/gx", sensor_gx_handler)
    disp.map("/sensor/gy", sensor_gy_handler)
    disp.set_default_handler(default_handler)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 12000), disp)
    print("OSC Server is running on {}".format(server.server_address))
    server.serve_forever()

# ---------------------
# JSON Loading using orjson (done only once at startup)
# ---------------------
def load_bubbles(json_filename):
    global bubbles, DATA_IS_3D
    if not os.path.exists(json_filename):
        print(f"JSON file not found: {json_filename}")
        return
    try:
        with open(json_filename, "rb") as f:
            json_bytes = f.read()
            json_data = orjson.loads(json_bytes)
    except Exception as e:
        print("Error loading JSON file:", e)
        return
    pca_values = json_data.get("mapping", [])
    print("Number of objects in JSON:", len(pca_values))
    bubbles.clear()
    if pca_values:
        coord = pca_values[0].get("Reduced coordinate", {})
        DATA_IS_3D = ("z" in coord)
    else:
        DATA_IS_3D = False
    for pca in pca_values:
        position = pca.get("Reduced coordinate", {})
        x_val = float(position.get("x", 0)) * SCALE_FACTOR
        y_val = float(position.get("y", 0)) * SCALE_FACTOR
        z_val = float(position.get("z", 0)) * Z_SCALE_FACTOR + Z_OFFSET if DATA_IS_3D else 0
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
# Fullscreen Mouse Coordinate Conversion
# ---------------------
def convert_mouse_coords(physical_pos):
    phys_x, phys_y = physical_pos
    virt_x = (phys_x - FS_OFFSET_X) / FS_SCALE_FACTOR
    virt_y = (phys_y - FS_OFFSET_Y) / FS_SCALE_FACTOR
    return virt_x, virt_y

# ---------------------
# Main Loop
# ---------------------
def main():
    global cursor_x, cursor_y, osc_client, rotating, prev_mouse_x, prev_mouse_y
    global camera_yaw, camera_pitch, CAMERA_DISTANCE, current_latent, target_latent
    global last_click_time, orbit_points, orbit_index, orbit_t, last_selected_bubble
    global fullscreen, WIDTH, HEIGHT, SCREEN_CENTER, FS_SCALE_FACTOR, FS_OFFSET_X, FS_OFFSET_Y
    global yaw_slider_active, pitch_slider_active, dist_slider_active, orbit_slider_active
    global yaw_slider_value, pitch_slider_value, dist_slider_value, orbit_slider_value, orbit_speed
    global recording, recorded_points

    # Lower the process priority to give audio engine more CPU time.
    try:
        os.nice(10)
    except Exception as e:
        print("Error lowering process priority:", e)

    parser = argparse.ArgumentParser(
        description="PCA Visualization with Rotation, Zoom, SLERP, Orbit Mode, and UI."
    )
    parser.add_argument(
        "--pca_json",
        type=str,
        default="pca_latent_mapping_reza_stereo.json",
        help="Path to the PCA map JSON file."
    )
    args = parser.parse_args()

    pygame.init()
    virtual_surface = pygame.Surface((VIRTUAL_WIDTH, VIRTUAL_HEIGHT))
    
    # --- Start fullscreen by default ---
    info = pygame.display.Info()
    full_width, full_height = info.current_w, info.current_h
    screen = pygame.display.set_mode((full_width, full_height), pygame.FULLSCREEN)
    pygame.display.set_caption("RAVE Latent PCA Viz")
    clock = pygame.time.Clock()

    fullscreen = True
    FS_SCALE_FACTOR = min(full_width / VIRTUAL_WIDTH, full_height / VIRTUAL_HEIGHT)
    scaled_width = int(VIRTUAL_WIDTH * FS_SCALE_FACTOR)
    scaled_height = int(VIRTUAL_HEIGHT * FS_SCALE_FACTOR)
    FS_OFFSET_X = (full_width - scaled_width) // 2
    FS_OFFSET_Y = (full_height - scaled_height) // 2
    pygame.mouse.set_visible(False)
    # --- End fullscreen modifications ---

    cursor_x = VIRTUAL_WIDTH / 2
    cursor_y = VIRTUAL_HEIGHT / 2

    # Load the JSON file once (outside the render loop)
    load_bubbles(args.pca_json)

    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9999)
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    if bubbles:
        current_latent = bubbles[0].latent
        target_latent = bubbles[0].latent
        last_selected_bubble = bubbles[0]

    font = pygame.font.SysFont(None, 20)

    orbit_points = []
    orbit_index = 0
    orbit_t = 0.0
    last_click_time = 0

    recording = False
    recorded_points = []

    # Record button drawn as a simple circle at the top center.
    RECORD_BTN_CENTER = (VIRTUAL_WIDTH // 2, 50)
    RECORD_BTN_RADIUS = 35

    running = True
    hovered_bubble = None
    while running:
        dt = clock.get_time() / 1000.0
        current_time = pygame.time.get_ticks()

        # Precompute trigonometric values once per frame (if 3D).
        if DATA_IS_3D:
            cos_yaw = math.cos(camera_yaw)
            sin_yaw = math.sin(camera_yaw)
            cos_pitch = math.cos(camera_pitch)
            sin_pitch = math.sin(camera_pitch)
        else:
            cos_yaw, sin_yaw, cos_pitch, sin_pitch = 1, 0, 1, 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F11:
                    fullscreen = not fullscreen
                    if fullscreen:
                        info = pygame.display.Info()
                        full_width, full_height = info.current_w, info.current_h
                        screen = pygame.display.set_mode((full_width, full_height), pygame.FULLSCREEN)
                        FS_SCALE_FACTOR = min(full_width / VIRTUAL_WIDTH, full_height / VIRTUAL_HEIGHT)
                        scaled_width = int(VIRTUAL_WIDTH * FS_SCALE_FACTOR)
                        scaled_height = int(VIRTUAL_HEIGHT * FS_SCALE_FACTOR)
                        FS_OFFSET_X = (full_width - scaled_width) // 2
                        FS_OFFSET_Y = (full_height - scaled_height) // 2
                        pygame.mouse.set_visible(False)
                    else:
                        screen = pygame.display.set_mode((VIRTUAL_WIDTH, VIRTUAL_HEIGHT))
                        FS_SCALE_FACTOR = 1.0
                        FS_OFFSET_X, FS_OFFSET_Y = 0, 0
                        pygame.mouse.set_visible(True)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if fullscreen:
                    pos = convert_mouse_coords(event.pos)
                if event.button == 1:
                    mx, my = pos
                    # Check if click is within the Clear Trajectories button.
                    clear_rect = pygame.Rect((VIRTUAL_WIDTH - ORBIT_SLIDER_WIDTH) // 2,
                                               ORBIT_SLIDER_Y + ORBIT_SLIDER_HEIGHT + 10,
                                               ORBIT_SLIDER_WIDTH, 25)
                    if clear_rect.collidepoint(mx, my):
                        orbit_points = []
                        orbit_index = 0
                        orbit_t = 0.0
                        for b in bubbles:
                            b.orbit_selected = False
                        print("Cleared orbit trajectories.")
                    # Check camera control sliders.
                    elif (CAMERA_YAW_SLIDER_X <= mx <= CAMERA_YAW_SLIDER_X + CAMERA_YAW_SLIDER_WIDTH and
                          CAMERA_YAW_SLIDER_Y <= my <= CAMERA_YAW_SLIDER_Y + CAMERA_YAW_SLIDER_HEIGHT):
                        yaw_slider_active = True
                    elif (CAMERA_PITCH_SLIDER_X <= mx <= CAMERA_PITCH_SLIDER_X + CAMERA_PITCH_SLIDER_WIDTH and
                          CAMERA_PITCH_SLIDER_Y <= my <= CAMERA_PITCH_SLIDER_Y + CAMERA_PITCH_SLIDER_HEIGHT):
                        pitch_slider_active = True
                    elif (CAMERA_DIST_SLIDER_X <= mx <= CAMERA_DIST_SLIDER_X + CAMERA_DIST_SLIDER_WIDTH and
                          CAMERA_DIST_SLIDER_Y <= my <= CAMERA_DIST_SLIDER_Y + CAMERA_DIST_SLIDER_HEIGHT):
                        dist_slider_active = True
                    # Check orbit speed slider.
                    elif (ORBIT_SLIDER_X <= mx <= ORBIT_SLIDER_X + ORBIT_SLIDER_WIDTH and
                          ORBIT_SLIDER_Y <= my <= ORBIT_SLIDER_Y + ORBIT_SLIDER_HEIGHT):
                        orbit_slider_active = True
                    # Check record button area.
                    elif math.hypot(mx - RECORD_BTN_CENTER[0], my - RECORD_BTN_CENTER[1]) <= RECORD_BTN_RADIUS:
                        recording = not recording
                        if recording:
                            recorded_points = []
                            print("Recording started.")
                        else:
                            if recorded_points:
                                orbit_points = recorded_points.copy()
                                print("Recording stopped. Orbit created with", len(orbit_points), "points.")
                            else:
                                print("Recording stopped; no points recorded.")
                    else:
                        # Otherwise, use for bubble selection (double-click detection).
                        if current_time - last_click_time < double_click_threshold:
                            for b in bubbles:
                                if b.rollover(mx, my, CAMERA_DISTANCE, SCREEN_CENTER,
                                              cos_yaw, sin_yaw, cos_pitch, sin_pitch):
                                    if b in orbit_points:
                                        orbit_points.remove(b)
                                        b.orbit_selected = False
                                        print("Removed bubble from orbit.")
                                    else:
                                        orbit_points.append(b)
                                        b.orbit_selected = True
                                        print("Added bubble to orbit.")
                                    break
                        last_click_time = current_time
                elif event.button == 3:
                    rotating = True
                    prev_mouse_x, prev_mouse_y = pos
                elif event.button == 4:
                    CAMERA_DISTANCE = max(MIN_CAMERA_DISTANCE, CAMERA_DISTANCE - 50)
                elif event.button == 5:
                    CAMERA_DISTANCE += 50

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    yaw_slider_active = False
                    pitch_slider_active = False
                    dist_slider_active = False
                    orbit_slider_active = False
                elif event.button == 3:
                    rotating = False
                    prev_mouse_x, prev_mouse_y = None, None

            elif event.type == pygame.MOUSEMOTION:
                pos = event.pos
                if fullscreen:
                    pos = convert_mouse_coords(event.pos)
                cursor_x, cursor_y = pos
                if yaw_slider_active:
                    yaw_slider_value = (cursor_y - CAMERA_YAW_SLIDER_Y) / CAMERA_YAW_SLIDER_HEIGHT
                    yaw_slider_value = max(0, min(1, yaw_slider_value))
                if pitch_slider_active:
                    pitch_slider_value = (cursor_y - CAMERA_PITCH_SLIDER_Y) / CAMERA_PITCH_SLIDER_HEIGHT
                    pitch_slider_value = max(0, min(1, pitch_slider_value))
                if dist_slider_active:
                    dist_slider_value = (cursor_y - CAMERA_DIST_SLIDER_Y) / CAMERA_DIST_SLIDER_HEIGHT
                    dist_slider_value = max(0, min(1, dist_slider_value))
                if orbit_slider_active:
                    orbit_slider_value = (cursor_x - ORBIT_SLIDER_X) / ORBIT_SLIDER_WIDTH
                    orbit_slider_value = max(0, min(1, orbit_slider_value))
                    orbit_speed = orbit_slider_value * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN) + ORBIT_SPEED_MIN
                if rotating and prev_mouse_x is not None and prev_mouse_y is not None:
                    dx = pos[0] - prev_mouse_x
                    dy = pos[1] - prev_mouse_y
                    camera_yaw += dx * ROTATION_SENSITIVITY
                    camera_pitch += dy * ROTATION_SENSITIVITY
                    prev_mouse_x, prev_mouse_y = pos

        # --- If recording is active, record the hovered bubble ---
        if recording:
            hovered_bubble = None
            for b in bubbles:
                if b.rollover(cursor_x, cursor_y, CAMERA_DISTANCE, SCREEN_CENTER,
                              cos_yaw, sin_yaw, cos_pitch, sin_pitch):
                    hovered_bubble = b
                    break
            if hovered_bubble is not None and hovered_bubble not in recorded_points:
                recorded_points.append(hovered_bubble)
                print("Recorded a point. Total recorded:", len(recorded_points))

        # --- Update Camera Parameters from slider values ---
        camera_yaw = -math.pi + 2 * math.pi * yaw_slider_value
        camera_pitch = - (math.pi/4) + (math.pi/2) * pitch_slider_value
        CAMERA_DISTANCE = MIN_CAMERA_DISTANCE + (MAX_CAMERA_DISTANCE - MIN_CAMERA_DISTANCE) * dist_slider_value

        # --- Orbit Interpolation ---
        if len(orbit_points) >= 2:
            orbit_t += orbit_speed * dt
            if orbit_t >= 1.0:
                steps = int(orbit_t)
                orbit_t %= 1.0
                orbit_index = (orbit_index + steps) % len(orbit_points)
            next_index = (orbit_index + 1) % len(orbit_points)
            current_orbit_latent = slerp_nd(orbit_points[orbit_index].latent,
                                            orbit_points[next_index].latent,
                                            orbit_t)
            osc_client.send_message("/latent", current_orbit_latent)
        else:
            orbit_index = 0
            orbit_t = 0.0
            hovered_bubble = None
            for b in bubbles:
                if b.rollover(cursor_x, cursor_y, CAMERA_DISTANCE, SCREEN_CENTER,
                              cos_yaw, sin_yaw, cos_pitch, sin_pitch):
                    hovered_bubble = b
                    break
            if hovered_bubble is not None:
                target_latent = hovered_bubble.latent
                last_selected_bubble = hovered_bubble
            if current_latent is not None and target_latent is not None:
                current_latent = slerp(current_latent, target_latent, INTERP_SPEED)
                osc_client.send_message("/latent", current_latent)

        # --- Drawing ---
        virtual_surface.fill((0, 0, 0))
        for b in bubbles:
            b.display(virtual_surface, CAMERA_DISTANCE, SCREEN_CENTER,
                      cos_yaw, sin_yaw, cos_pitch, sin_pitch)
        pygame.draw.circle(virtual_surface, (0, 202, 0), (int(cursor_x), int(cursor_y)), 5)

        # Draw vertical camera control sliders on the left.
        pygame.draw.rect(virtual_surface, (100, 100, 100),
                         (CAMERA_YAW_SLIDER_X, CAMERA_YAW_SLIDER_Y, CAMERA_YAW_SLIDER_WIDTH, CAMERA_YAW_SLIDER_HEIGHT))
        knob_y = CAMERA_YAW_SLIDER_Y + int(yaw_slider_value * CAMERA_YAW_SLIDER_HEIGHT) - 5
        pygame.draw.rect(virtual_surface, (200, 200, 200),
                         (CAMERA_YAW_SLIDER_X - 2, knob_y, CAMERA_YAW_SLIDER_WIDTH + 4, 10))
        yaw_label = font.render(f"Yaw: {math.degrees(camera_yaw):.1f}°", True, (255, 255, 255))
        virtual_surface.blit(yaw_label, (CAMERA_YAW_SLIDER_X + CAMERA_YAW_SLIDER_WIDTH + 5, CAMERA_YAW_SLIDER_Y))

        pygame.draw.rect(virtual_surface, (100, 100, 100),
                         (CAMERA_PITCH_SLIDER_X, CAMERA_PITCH_SLIDER_Y, CAMERA_PITCH_SLIDER_WIDTH, CAMERA_PITCH_SLIDER_HEIGHT))
        knob_y = CAMERA_PITCH_SLIDER_Y + int(pitch_slider_value * CAMERA_PITCH_SLIDER_HEIGHT) - 5
        pygame.draw.rect(virtual_surface, (200, 200, 200),
                         (CAMERA_PITCH_SLIDER_X - 2, knob_y, CAMERA_PITCH_SLIDER_WIDTH + 4, 10))
        pitch_label = font.render(f"Pitch: {math.degrees(camera_pitch):.1f}°", True, (255, 255, 255))
        virtual_surface.blit(pitch_label, (CAMERA_PITCH_SLIDER_X + CAMERA_PITCH_SLIDER_WIDTH + 5, CAMERA_PITCH_SLIDER_Y))

        pygame.draw.rect(virtual_surface, (100, 100, 100),
                         (CAMERA_DIST_SLIDER_X, CAMERA_DIST_SLIDER_Y, CAMERA_DIST_SLIDER_WIDTH, CAMERA_DIST_SLIDER_HEIGHT))
        knob_y = CAMERA_DIST_SLIDER_Y + int(dist_slider_value * CAMERA_DIST_SLIDER_HEIGHT) - 5
        pygame.draw.rect(virtual_surface, (200, 200, 200),
                         (CAMERA_DIST_SLIDER_X - 2, knob_y, CAMERA_DIST_SLIDER_WIDTH + 4, 10))
        dist_label = font.render(f"Dist: {CAMERA_DISTANCE:.1f}", True, (255, 255, 255))
        virtual_surface.blit(dist_label, (CAMERA_DIST_SLIDER_X + CAMERA_DIST_SLIDER_WIDTH + 5, CAMERA_DIST_SLIDER_Y))

        # Draw orbit speed slider and clear button.
        if len(orbit_points) >= 2:
            pygame.draw.rect(virtual_surface, (100, 100, 100),
                             (ORBIT_SLIDER_X, ORBIT_SLIDER_Y, ORBIT_SLIDER_WIDTH, ORBIT_SLIDER_HEIGHT))
            knob_x = ORBIT_SLIDER_X + int(orbit_slider_value * ORBIT_SLIDER_WIDTH) - 5
            pygame.draw.rect(virtual_surface, (200, 200, 200),
                             (knob_x, ORBIT_SLIDER_Y - 2, 10, ORBIT_SLIDER_HEIGHT + 4))
            orbit_label = font.render(f"Orbit Speed: {orbit_speed:.3f} Hz", True, (255, 255, 255))
            virtual_surface.blit(orbit_label, (ORBIT_SLIDER_X, ORBIT_SLIDER_Y - 25))
            clear_rect = pygame.Rect((VIRTUAL_WIDTH - ORBIT_SLIDER_WIDTH) // 2,
                                     ORBIT_SLIDER_Y + ORBIT_SLIDER_HEIGHT + 10,
                                     ORBIT_SLIDER_WIDTH, 25)
            pygame.draw.rect(virtual_surface, (150, 0, 0), clear_rect)
            clear_label = font.render("Clear Trajectories", True, (255, 255, 255))
            clear_text_rect = clear_label.get_rect(center=clear_rect.center)
            virtual_surface.blit(clear_label, clear_text_rect)
            proj_points = []
            for b in orbit_points:
                pt = b.project(CAMERA_DISTANCE, SCREEN_CENTER,
                               cos_yaw, sin_yaw, cos_pitch, sin_pitch)
                proj_points.append((int(pt[0]), int(pt[1])))
            if len(proj_points) >= 2:
                pygame.draw.lines(virtual_surface, (128, 0, 128), True, proj_points, 4)
            p1 = orbit_points[orbit_index].project(CAMERA_DISTANCE, SCREEN_CENTER,
                                                   cos_yaw, sin_yaw, cos_pitch, sin_pitch)
            p2 = orbit_points[(orbit_index+1) % len(orbit_points)].project(CAMERA_DISTANCE, SCREEN_CENTER,
                                                                           cos_yaw, sin_yaw, cos_pitch, sin_pitch)
            marker_x = p1[0] + orbit_t * (p2[0] - p1[0])
            marker_y = p1[1] + orbit_t * (p2[1] - p1[1])
            pygame.draw.circle(virtual_surface, (128, 0, 128), (int(marker_x), int(marker_y)), 5)
        # --- Draw Record Button ---
        record_color = (128, 0, 128) if recording else (0, 255, 0)
        pygame.draw.circle(virtual_surface, record_color, RECORD_BTN_CENTER, RECORD_BTN_RADIUS)

        # --- Display Latent Vector as a Column on the Right Side ---
        if last_selected_bubble is not None:
            x_text = VIRTUAL_WIDTH - 90  # 90 pixels from the right edge.
            y_text = VIRTUAL_WIDTH - 550 
            for val in last_selected_bubble.latent:
                text_line = f"{val:.3f}"
                text_surface = font.render(text_line, True, (0, 255, 0))
                virtual_surface.blit(text_surface, (x_text, y_text))
                y_text += text_surface.get_height() + 5

        if fullscreen:
            info = pygame.display.Info()
            full_width, full_height = info.current_w, info.current_h
            scaled_width = int(VIRTUAL_WIDTH * FS_SCALE_FACTOR)
            scaled_height = int(VIRTUAL_HEIGHT * FS_SCALE_FACTOR)
            scaled_surface = pygame.transform.scale(virtual_surface, (scaled_width, scaled_height))
            screen.fill((0, 0, 0))
            screen.blit(scaled_surface, (FS_OFFSET_X, FS_OFFSET_Y))
        else:
            screen.blit(virtual_surface, (0, 0))
        pygame.display.flip()
        clock.tick(23)  # You may try a lower FPS (e.g., 15) if dropouts persist.
    pygame.quit()

if __name__ == '__main__':
    main()
