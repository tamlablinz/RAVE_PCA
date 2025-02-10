import pygame
import json
import math
import threading
from pythonosc import dispatcher, osc_server, udp_client

# ---------------------
# Global Variables
# ---------------------
# List to hold all Bubble objects.
bubbles = []

# Global cursor position; now updated by the mouse.
cursor_x = 0.0
cursor_y = 0.0

# The expected number of latent values per bubble.
latent_coordinates_size = 0

# OSC client for sending OSC messages.
osc_client = None

# Screen dimensions.
WIDTH, HEIGHT = 1200, 1200

# ---------------------
# Bubble Class Definition
# ---------------------
class Bubble:
    def __init__(self, x, y, diameter, latent):
        self.x = x
        self.y = y
        self.diameter = diameter  # In pixels; note that in the original code the diameter is 4.
        self.latent = list(latent)  # Copy the list of latent values.
        self.over = False         # Whether the cursor is over the bubble.

    def rollover(self, px, py, surface):
        """
        Check if the point (px, py) is within the bubble.
        Also, draw an indicator circle at the cursor position.
        """
        # Draw a filled circle at the cursor position (diameter 20, radius 10) with color (204,202,0).
        indicator_color = (204, 202, 0)
        pygame.draw.circle(surface, indicator_color, (int(px), int(py)), 10)

        # Compute Euclidean distance.
        d = math.hypot(self.x - px, self.y - py)
        self.over = d < self.diameter
        return self.over

    def display(self, surface):
        """
        Draw the bubble.
        If hovered, draw it filled with color (204,102,0);
        otherwise, draw an outline in black.
        """
        if self.over:
            color = (204, 102, 0)
            # Draw a filled circle.
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), int(self.diameter / 2))
        else:
            # Draw only the outline in black.
            color = (0, 0, 0)
            # The "width" parameter of 1 draws an outline.
            pygame.draw.circle(surface, color, (int(self.x), int(self.y)), int(self.diameter / 2), 1)


# ---------------------
# OSC Callback Functions (optional)
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
    Set up and run the OSC server on port 12000.
    This function is run in a separate thread.
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
def load_bubbles():
    """
    Loads the JSON file and creates Bubble objects.
    The JSON is expected to contain a key "pca", which is a list of objects.
    Each object should have:
      - a dictionary "PCA coordinate" with keys "x" and "y"
      - an array "Latent coordinate"
    """
    global latent_coordinates_size, bubbles

    try:
        with open("pca_latent_mapping_reza_stereo.json", "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print("Error loading JSON file:", e)
        return

    pca_values = json_data.get("pca", [])
    print("Number of objects:", len(pca_values))
    bubbles = []  # Clear any existing bubbles

    for pca in pca_values:
        position = pca.get("PCA coordinate", {})
        # Scale the coordinates by 50 and convert to int.
        x_val = int(float(position.get("x", 0)) * 50)
        y_val = int(float(position.get("y", 0)) * 50)

        # Get the latent coordinate list.
        latent = pca.get("Latent coordinate", [])
        latent_coordinates_size = len(latent)
        latent_floats = [float(v) for v in latent]

        # The original code offsets x by WIDTH/2 and y by HEIGHT/3.
        bubble = Bubble(x_val + WIDTH / 2, y_val + HEIGHT / 3, 4, latent_floats)
        bubbles.append(bubble)
    print("Done loading bubbles.")


# ---------------------
# Main Loop using pygame
# ---------------------
def main():
    global cursor_x, cursor_y, osc_client

    # Initialize pygame.
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Processing Sketch in Pygame")
    clock = pygame.time.Clock()

    # Set initial cursor position as in the original sketch.
    cursor_x = WIDTH / 2
    cursor_y = HEIGHT / 4

    # Load bubbles from the JSON file.
    load_bubbles()

    # Set up the OSC client to send messages to 127.0.0.1:9999.
    osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9999)

    # Start the OSC server in a separate daemon thread.
    osc_thread = threading.Thread(target=start_osc_server, daemon=True)
    osc_thread.start()

    running = True
    while running:
        # Process events (quit event, etc.)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Update cursor position from the mouse movement.
        cursor_x, cursor_y = pygame.mouse.get_pos()

        # Clear the screen with white.
        screen.fill((255, 255, 255))

        # For each bubble, display it and check for rollover.
        for b in bubbles:
            # Check if the current mouse position (cursor_x, cursor_y) is over the bubble.
            if b.rollover(cursor_x, cursor_y, screen):
                # If so, send an OSC message with the bubble's latent coordinates.
                osc_client.send_message("/latent", b.latent)
            # Draw the bubble.
            b.display(screen)

        # Update the display.
        pygame.display.flip()

        # Limit frame rate to 60 FPS.
        clock.tick(60)

    pygame.quit()


if __name__ == '__main__':
    main()
