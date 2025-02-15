#!/usr/bin/env python3
"""
PCA Visualization with Pygame, Interactive Camera Rotation/Zoom, SLERP Interpolation,
Display of Selected Coordinates and Latent Vector, and an Orbit Feature with a SLERP Orbit Speed Slider

This script loads a PCA mapping JSON file generated from RAVE latent space analysis.
If the JSON contains 2D coordinates (only "x" and "y"), it displays the data in 2D.
If the JSON contains 3D coordinates ("x", "y", and "z"), it displays a full 3D perspective
with interactive camera rotation (right-click drag), zoom (mouse scroll), and smooth SLERP
interpolation of latent vectors (sent via OSC).

Additionally, the user can double-click on bubbles to select them for an "orbit". When 2 or more
bubbles are selected, the program continuously interpolates between the selected points, drawing a
purple trajectory and a moving marker along the orbit. Double-clicking on an already selected bubble
removes it from the orbit trajectory. A dedicated slider allows adjustment of the orbit interpolation
speed. A "Clear Trajectories" button is provided under the orbit speed slider to clear the orbit
selections. Press F11 to toggle fullscreen mode (using the computer's native resolution).

Author: Moisés Horta Valenzuela
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
# Get Screen Dimensions Dynamically
# ---------------------
try:
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()
except ImportError:
    screen_width, screen_height = 1920, 1080

# ---------------------
# Global Variables
# ---------------------
bubbles = []           # List to hold all Bubble objects.
DATA_IS_3D = False     # Will be determined from the JSON.

# Global cursor position.
cursor_x = 0.0
cursor_y = 0.0

# OSC client.
osc_client = None

# Set your virtual (logical) resolution.
DEFAULT_WIDTH, DEFAULT_HEIGHT = int(screen_width / 1.25), int(screen_height / 1.25)
VIRTUAL_WIDTH, VIRTUAL_HEIGHT = DEFAULT_WIDTH, DEFAULT_HEIGHT
WIDTH, HEIGHT = VIRTUAL_WIDTH, VIRTUAL_HEIGHT

# Camera parameters.
CAMERA_DISTANCE = 300
SCREEN_CENTER = (WIDTH // 2, HEIGHT // 2)

# Scaling factors.
SCALE_FACTOR = 100
Z_SCALE_FACTOR = 100
Z_OFFSET = 300

# Camera rotation.
camera_yaw = 0.0
camera_pitch = 0.0

# Drag variables.
rotating = False
prev_mouse_x = None
prev_mouse_y = None
ROTATION_SENSITIVITY = 0.01

# Zoom.
ZOOM_STEP = 50
MIN_CAMERA_DISTANCE = 1

# Normal (hover-based) SLERP interpolation.
current_latent = None
target_latent = None
INTERP_SPEED = 0.999

# ---------------------
# Orbit Feature Variables
# ---------------------
orbit_points = []      # Orbit-selected bubbles.
orbit_index = 0
orbit_t = 0.0
last_click_time = 0
double_click_threshold = 300  # ms

# Orbit speed slider variables.
ORBIT_SLIDER_X = 10
ORBIT_SLIDER_Y = VIRTUAL_HEIGHT - 70
ORBIT_SLIDER_WIDTH = 200
ORBIT_SLIDER_HEIGHT = 20
orbit_slider_active = False
orbit_slider_value = 0.5
ORBIT_SPEED_MIN = 0.001
ORBIT_SPEED_MAX = 25.0  # Maximum orbit speed in Hz.
orbit_speed = orbit_slider_value * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN) + ORBIT_SPEED_MIN

# Clear Trajectories button.
ORBIT_CLEAR_BTN_X = ORBIT_SLIDER_X
ORBIT_CLEAR_BTN_Y = ORBIT_SLIDER_Y + ORBIT_SLIDER_HEIGHT + 10
ORBIT_CLEAR_BTN_WIDTH = ORBIT_SLIDER_WIDTH
ORBIT_CLEAR_BTN_HEIGHT = 25

# Normal slider variables.
NORMAL_SLIDER_X = 10
NORMAL_SLIDER_Y = VIRTUAL_HEIGHT - 40
NORMAL_SLIDER_WIDTH = 200
NORMAL_SLIDER_HEIGHT = 20
normal_slider_active = False
normal_slider_value = 0.99

# Persistent selection.
last_selected_bubble = None

# ---------------------
# Fullscreen Toggle and Scaling Variables
# ---------------------
fullscreen = False
FS_SCALE_FACTOR = 1.0
FS_OFFSET_X = 0
FS_OFFSET_Y = 0

# ---------------------
# Helper Function: SLERP
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

# ---------------------
# Helper Function: SLERP on N-Dimensional Vectors
# ---------------------
def slerp_nd(v0, v1, t):
    v0 = np.array(v0, dtype=float)
    v1 = np.array(v1, dtype=float)
    dot = np.clip(np.dot(v0, v1), -1.0, 1.0)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-6:
        return ((1-t)*v0 + t*v1).tolist()
    return ((math.sin((1-t)*theta)/sin_theta)*v0 + (math.sin(t*theta)/sin_theta)*v1).tolist()

# ---------------------
# Helper Function for 3D Rotation
# ---------------------
def rotate_point(x, y, z, yaw, pitch):
    x1 = math.cos(yaw)*x + math.sin(yaw)*z
    y1 = y
    z1 = -math.sin(yaw)*x + math.cos(yaw)*z
    y2 = math.cos(pitch)*y1 - math.sin(pitch)*z1
    z2 = math.sin(pitch)*y1 + math.cos(pitch)*z1
    return x1, y2, z2

# ---------------------
# Bubble Class Definition
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

    def project(self, camera_distance, screen_center, yaw, pitch):
        if not DATA_IS_3D:
            proj_x = screen_center[0] + self.x
            proj_y = screen_center[1] - self.y
            proj_diameter = self.base_diameter
            return proj_x, proj_y, proj_diameter
        x_rot, y_rot, z_rot = rotate_point(self.x, self.y, self.z, yaw, pitch)
        denom = camera_distance + z_rot
        factor = camera_distance / denom if denom != 0 else 1
        proj_x = screen_center[0] + x_rot * factor
        proj_y = screen_center[1] - y_rot * factor
        proj_diameter = self.base_diameter * factor
        return proj_x, proj_y, proj_diameter

    def rollover(self, px, py, camera_distance, screen_center, yaw, pitch):
        proj_x, proj_y, proj_diameter = self.project(camera_distance, screen_center, yaw, pitch)
        d = math.hypot(proj_x - px, proj_y - py)
        self.over = d < (proj_diameter / 2)
        return self.over

    def display(self, surface, camera_distance, screen_center, yaw, pitch):
        proj_x, proj_y, proj_diameter = self.project(camera_distance, screen_center, yaw, pitch)
        # Check that the projection values are finite numbers.
        if not (math.isfinite(proj_x) and math.isfinite(proj_y) and math.isfinite(proj_diameter)):
            return
        try:
            cx = int(float(proj_x))
            cy = int(float(proj_y))
            radius = int(float(proj_diameter) / 2)
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
# OSC Callback Functions
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
# Initialization and JSON Loading
# ---------------------
def load_bubbles(json_filename):
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
    pca_values = json_data.get("mapping", [])
    print("Number of objects in JSON:", len(pca_values))
    bubbles = []
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
# Helper to convert fullscreen mouse coordinates to virtual coordinates
# ---------------------
def convert_mouse_coords(physical_pos):
    phys_x, phys_y = physical_pos
    virt_x = (phys_x - FS_OFFSET_X) / FS_SCALE_FACTOR
    virt_y = (phys_y - FS_OFFSET_Y) / FS_SCALE_FACTOR
    return virt_x, virt_y

# ---------------------
# Main Loop using Pygame
# ---------------------
def main():
    global cursor_x, cursor_y, osc_client, rotating, prev_mouse_x, prev_mouse_y
    global camera_yaw, camera_pitch, CAMERA_DISTANCE, current_latent, target_latent
    global normal_slider_active, normal_slider_value, INTERP_SPEED, last_click_time
    global orbit_points, orbit_index, orbit_t, orbit_slider_active, orbit_slider_value, orbit_speed, last_selected_bubble
    global fullscreen, WIDTH, HEIGHT, SCREEN_CENTER, FS_SCALE_FACTOR, FS_OFFSET_X, FS_OFFSET_Y

    parser = argparse.ArgumentParser(
        description="PCA Visualization with Rotation, Zoom, SLERP, Orbit Mode, and UI. Loads a PCA JSON mapping (2D or 3D)."
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
    screen = pygame.display.set_mode((VIRTUAL_WIDTH, VIRTUAL_HEIGHT))
    pygame.display.set_caption("RAVE Latent PCA Viz")
    clock = pygame.time.Clock()

    fullscreen = False

    cursor_x = VIRTUAL_WIDTH / 2
    cursor_y = VIRTUAL_HEIGHT / 2

    load_bubbles(args.pca_json)

    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9999)

    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    if bubbles:
        current_latent = bubbles[0].latent
        target_latent = bubbles[0].latent
        last_selected_bubble = bubbles[0]

    font = pygame.font.SysFont(None, 20)

    # Initialize orbit-related globals.
    orbit_points = []
    orbit_index = 0
    orbit_t = 0.0
    orbit_slider_active = False
    orbit_slider_value = 0.5
    ORBIT_SPEED_MIN = 0.001
    ORBIT_SPEED_MAX = 25.0
    orbit_speed = orbit_slider_value * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN) + ORBIT_SPEED_MIN
    last_click_time = 0

    # Initialize normal slider.
    normal_slider_active = False
    normal_slider_value = 0.99
    INTERP_SPEED = normal_slider_value * (1.0 - 0.90) + 0.90

    WIDTH, HEIGHT = VIRTUAL_WIDTH, VIRTUAL_HEIGHT
    SCREEN_CENTER = (WIDTH // 2, HEIGHT // 2)
    FS_SCALE_FACTOR = 1.0
    FS_OFFSET_X, FS_OFFSET_Y = 0, 0

    running = True
    hovered_bubble = None
    while running:
        dt = clock.get_time() / 1000.0
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # Toggle fullscreen on F11.
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
                    else:
                        screen = pygame.display.set_mode((VIRTUAL_WIDTH, VIRTUAL_HEIGHT))
                        FS_SCALE_FACTOR = 1.0
                        FS_OFFSET_X, FS_OFFSET_Y = 0, 0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = event.pos
                if fullscreen:
                    pos = convert_mouse_coords(event.pos)
                if event.button == 1:
                    mx, my = pos
                    if len(orbit_points) >= 2:
                        if (ORBIT_CLEAR_BTN_X <= mx <= ORBIT_CLEAR_BTN_X + ORBIT_CLEAR_BTN_WIDTH and
                            ORBIT_CLEAR_BTN_Y <= my <= ORBIT_CLEAR_BTN_Y + ORBIT_CLEAR_BTN_HEIGHT):
                            orbit_points = []
                            orbit_index = 0
                            orbit_t = 0.0
                            for b in bubbles:
                                b.orbit_selected = False
                            print("Cleared orbit trajectories.")
                            continue
                        if (ORBIT_SLIDER_X <= mx <= ORBIT_SLIDER_X + ORBIT_SLIDER_WIDTH and
                            ORBIT_SLIDER_Y <= my <= ORBIT_SLIDER_Y + ORBIT_SLIDER_HEIGHT):
                            orbit_slider_active = True
                            orbit_slider_value = (mx - ORBIT_SLIDER_X) / ORBIT_SLIDER_WIDTH
                            orbit_speed = orbit_slider_value * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN) + ORBIT_SPEED_MIN
                            continue
                    else:
                        if (NORMAL_SLIDER_X <= mx <= NORMAL_SLIDER_X + NORMAL_SLIDER_WIDTH and
                            NORMAL_SLIDER_Y <= my <= NORMAL_SLIDER_Y + NORMAL_SLIDER_HEIGHT):
                            normal_slider_active = True
                            normal_slider_value = (mx - NORMAL_SLIDER_X) / NORMAL_SLIDER_WIDTH
                            INTERP_SPEED = normal_slider_value * (1.0 - 0.90) + 0.90
                            continue
                    if current_time - last_click_time < double_click_threshold:
                        for b in bubbles:
                            if b.rollover(mx, my, CAMERA_DISTANCE, SCREEN_CENTER,
                                          camera_yaw if DATA_IS_3D else 0,
                                          camera_pitch if DATA_IS_3D else 0):
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
                    CAMERA_DISTANCE = max(MIN_CAMERA_DISTANCE, CAMERA_DISTANCE - ZOOM_STEP)
                elif event.button == 5:
                    CAMERA_DISTANCE += ZOOM_STEP
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    rotating = False
                    prev_mouse_x, prev_mouse_y = None, None
                if orbit_slider_active:
                    orbit_slider_active = False
                if normal_slider_active:
                    normal_slider_active = False
            elif event.type == pygame.MOUSEMOTION:
                pos = event.pos
                if fullscreen:
                    pos = convert_mouse_coords(event.pos)
                cursor_x, cursor_y = pos
                if orbit_slider_active:
                    orbit_slider_value = (pos[0] - ORBIT_SLIDER_X) / ORBIT_SLIDER_WIDTH
                    orbit_slider_value = max(0, min(1, orbit_slider_value))
                    orbit_speed = orbit_slider_value * (ORBIT_SPEED_MAX - ORBIT_SPEED_MIN) + ORBIT_SPEED_MIN
                if normal_slider_active:
                    normal_slider_value = (pos[0] - NORMAL_SLIDER_X) / NORMAL_SLIDER_WIDTH
                    normal_slider_value = max(0, min(1, normal_slider_value))
                    INTERP_SPEED = normal_slider_value * (1.0 - 0.90) + 0.90
                if rotating and prev_mouse_x is not None and prev_mouse_y is not None:
                    dx = pos[0] - prev_mouse_x
                    dy = pos[1] - prev_mouse_y
                    camera_yaw += dx * ROTATION_SENSITIVITY
                    camera_pitch += dy * ROTATION_SENSITIVITY
                    prev_mouse_x, prev_mouse_y = pos

        # if len(orbit_points) >= 2:
        #     orbit_t += orbit_speed * dt
        #     if orbit_t >= 1.0:
        #         orbit_t -= 1.0
        #         orbit_index = (orbit_index + 1) % len(orbit_points)
        #     next_index = (orbit_index + 1) % len(orbit_points)
        #     current_orbit_latent = slerp_nd(orbit_points[orbit_index].latent,
        #                                     orbit_points[next_index].latent,
        #                                     orbit_t)
        #     # current_orbit_latent = orbit_points[orbit_index].latent
        #     osc_client.send_message("/latent", current_orbit_latent)

        if len(orbit_points) >= 2:
            orbit_t += orbit_speed * dt
            if orbit_t >= 1.0:
                steps = int(orbit_t)        # Number of complete transitions.
                orbit_t %= 1.0              # Remainder becomes the new interpolation parameter.
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
                              camera_yaw if DATA_IS_3D else 0,
                              camera_pitch if DATA_IS_3D else 0):
                    hovered_bubble = b
                    break
            if hovered_bubble is not None:
                target_latent = hovered_bubble.latent
                last_selected_bubble = hovered_bubble
            if current_latent is not None and target_latent is not None:
                current_latent = slerp(current_latent, target_latent, INTERP_SPEED)
                osc_client.send_message("/latent", current_latent)

        virtual_surface.fill((0, 0, 0))
        for b in bubbles:
            b.display(virtual_surface, CAMERA_DISTANCE, SCREEN_CENTER,
                      camera_yaw if DATA_IS_3D else 0,
                      camera_pitch if DATA_IS_3D else 0)
        pygame.draw.circle(virtual_surface, (0, 202, 0), (int(cursor_x), int(cursor_y)), 5)
        if len(orbit_points) >= 2:
            p1 = orbit_points[orbit_index].project(CAMERA_DISTANCE, SCREEN_CENTER,
                                                   camera_yaw if DATA_IS_3D else 0,
                                                   camera_pitch if DATA_IS_3D else 0)
            p2 = orbit_points[(orbit_index+1) % len(orbit_points)].project(CAMERA_DISTANCE, SCREEN_CENTER,
                                                                           camera_yaw if DATA_IS_3D else 0,
                                                                           camera_pitch if DATA_IS_3D else 0)
            marker_x = p1[0] + orbit_t * (p2[0] - p1[0])
            marker_y = p1[1] + orbit_t * (p2[1] - p1[1])
            info_text = f"Orbit: {len(orbit_points)} pts | Pos: ({marker_x:.1f}, {marker_y:.1f})"
            info_color = (128, 0, 128)
        else:
            if last_selected_bubble is not None:
                proj = last_selected_bubble.project(CAMERA_DISTANCE, SCREEN_CENTER,
                                                    camera_yaw if DATA_IS_3D else 0,
                                                    camera_pitch if DATA_IS_3D else 0)
                info_text = f"PCA: ({proj[0]:.1f}, {proj[1]:.1f}) | Latent: {last_selected_bubble.latent}"
                info_color = (0, 102, 0)
            else:
                info_text = "PCA: None | Latent: None"
                info_color = (255, 255, 255)
        info_surface = font.render(info_text, True, info_color)
        virtual_surface.blit(info_surface, (VIRTUAL_WIDTH - info_surface.get_width() - 10, VIRTUAL_HEIGHT - 60))
        if len(orbit_points) >= 2:
            pygame.draw.rect(virtual_surface, (100, 100, 100), (ORBIT_SLIDER_X, ORBIT_SLIDER_Y, ORBIT_SLIDER_WIDTH, ORBIT_SLIDER_HEIGHT))
            knob_x = ORBIT_SLIDER_X + int(orbit_slider_value * ORBIT_SLIDER_WIDTH) - 5
            pygame.draw.rect(virtual_surface, (200, 200, 200), (knob_x, ORBIT_SLIDER_Y - 2, 10, ORBIT_SLIDER_HEIGHT + 4))
            orbit_label = font.render(f"Orbit Speed: {orbit_speed:.3f} Hz", True, (255, 255, 255))
            virtual_surface.blit(orbit_label, (ORBIT_SLIDER_X, ORBIT_SLIDER_Y - 25))
            pygame.draw.rect(virtual_surface, (150, 0, 0), (ORBIT_CLEAR_BTN_X, ORBIT_CLEAR_BTN_Y, ORBIT_CLEAR_BTN_WIDTH, ORBIT_CLEAR_BTN_HEIGHT))
            clear_label = font.render("Clear Trajectories", True, (255, 255, 255))
            clear_rect = clear_label.get_rect(center=(ORBIT_CLEAR_BTN_X + ORBIT_CLEAR_BTN_WIDTH/2, ORBIT_CLEAR_BTN_Y + ORBIT_CLEAR_BTN_HEIGHT/2))
            virtual_surface.blit(clear_label, clear_rect)
            proj_points = []
            for b in orbit_points:
                pt = b.project(CAMERA_DISTANCE, SCREEN_CENTER,
                               camera_yaw if DATA_IS_3D else 0,
                               camera_pitch if DATA_IS_3D else 0)
                proj_points.append((int(pt[0]), int(pt[1])))
            if len(proj_points) >= 2:
                pygame.draw.lines(virtual_surface, (128, 0, 128), True, proj_points, 2)
            pygame.draw.circle(virtual_surface, (128, 0, 128), (int(marker_x), int(marker_y)), 5)
        else:
            pygame.draw.rect(virtual_surface, (100, 100, 100), (NORMAL_SLIDER_X, NORMAL_SLIDER_Y, NORMAL_SLIDER_WIDTH, NORMAL_SLIDER_HEIGHT))
            knob_x = NORMAL_SLIDER_X + int(normal_slider_value * NORMAL_SLIDER_WIDTH) - 5
            pygame.draw.rect(virtual_surface, (200, 200, 200), (knob_x, NORMAL_SLIDER_Y - 2, 10, NORMAL_SLIDER_HEIGHT + 4))
            norm_label = font.render(f"Smoothness: {INTERP_SPEED:.3f}", True, (255, 255, 255))
            virtual_surface.blit(norm_label, (NORMAL_SLIDER_X, NORMAL_SLIDER_Y - 25))
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
        clock.tick(90)
    pygame.quit()

if __name__ == '__main__':
    main()
