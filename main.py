### Imports ###
import os
import win32api
import win32con
import threading
import atexit
import sys
import random
import time
import math
import numpy as np
import scipy
import skimage

import glfw
import OpenGL.GL as gl
import numpy as np
import math
import scipy.ndimage

try:
    import cupy as cp
    CUPY_SUPPORTED = True
    XP = cp
except ImportError:
    CUPY_SUPPORTED = False
    XP = np
    print("CUPY NOT SUPPORTED")

from PIL import Image

from src import COLOURS

### Constants ###
TPS = 30
FPS = 60

SCREEN_WIDTH = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
SCREEN_HEIGHT = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

#print(SCREEN_WIDTH, SCREEN_HEIGHT)

GRID_DIMENSIONS = (SCREEN_WIDTH, SCREEN_HEIGHT) # (HEIGHT, WIDTH)

class Tile:
    def __init__(self, name, colour):
        self.name = name
        self.colour = colour

TILES = [Tile("empty", (0.0, 0.0, 0.0)), Tile("solid", (1.0, 0.3, 0.3))]
TILES_COLOUR_LOOKUP = np.array([tile.colour for tile in TILES], dtype=np.float16)

CAMERA_SPEED = 10
ZOOM_MULTI = 1.2
MIN_ZOOM = pow(ZOOM_MULTI, -20)
MAX_ZOOM = pow(ZOOM_MULTI, 20)

### Thread Handling ###
events = {"exit": threading.Event()}
locks = {}

### Exit Handling ###
def exit_handler() -> None:
    """
    Runs before main threads terminates.
    """
    events["exit"].set()

atexit.register(exit_handler)

def crop_array(array: np.ndarray, coord: tuple, size: tuple) -> np.ndarray:
    """
    Crops a 2D numpy array from a given starting coordinate to a specified size.
    
    :param array: 2D numpy array to crop from.
    :param coord: Tuple (x, y) representing the starting coordinate for cropping.
    :param size: Tuple (width, height) representing the size of the cropped area.
    :return: Cropped 2D numpy array.
    """
    x, y = coord
    width, height = size

    if x >= array.shape[1] or y >= array.shape[0]:
        return np.zeros((0, 0), dtype=np.uint8)

    x_end = min(x + width, array.shape[1])
    y_end = min(y + height, array.shape[0])

    cropped_array = array[y:y_end, x:x_end]
    
    return cropped_array

### DEBUGGING ###
class FrameRateMonitor:
    def __init__(self, name=""):
        self.frame_times = []
        self.last_update_time = None
        self.name = name

        self.total_elapsed = 0

    def print_fps(self):
        if len(self.frame_times) and self.total_elapsed:
            fps = len(self.frame_times) / self.total_elapsed

            print(f"[{self.name}] FPS: {round(fps, 3)} LOW: {round(len(self.frame_times) / (max(*self.frame_times, 0.001) * len(self.frame_times)), 3)} HIGH: {round(len(self.frame_times) / (max(min(self.frame_times), 0.001) * len(self.frame_times)), 3)}")

        self.frame_times = []
        self.total_elapsed = 0

    def run(self):
        current_time = time.time()

        if self.last_update_time == None:
            self.last_update_time = current_time
            return

        elapsed = current_time - self.last_update_time
        self.last_update_time = current_time
        self.total_elapsed += elapsed
        self.frame_times.append(elapsed)

        if self.total_elapsed > 1:
            self.print_fps()


### Rendering [main thread] ###
class Window:
    def __init__(self, simulation) -> None:
        if not glfw.init():
            raise Exception("GLFW can't be initialized")
        
        self.monitor = glfw.get_primary_monitor()
        if not self.monitor:
            raise Exception("GLFW can't find primary monitor")

        self.video_mode = glfw.get_video_mode(self.monitor)
        if not self.video_mode:
            raise Exception("GLFW can't get video mode")
        
        self.screen_width = self.video_mode.size.width
        self.screen_height = self.video_mode.size.height

        self.simulation = simulation
        self.running = True
        
        self.ASPECT_RATIO = GRID_DIMENSIONS[0] / GRID_DIMENSIONS[1]
        self.original_size = np.array([self.screen_width, self.screen_height])
        self.current_size = self.original_size
        self.home_offset = np.array([0, 0])
        self.pos_offset = np.array([0, 0], dtype=np.float32)
        self.zoom_offset = np.array([0, 0])
        self.zoom = 1 

        self.prev_grid = np.zeros((0, 0), dtype=np.uint8)

        self.window = glfw.create_window(self.screen_width, self.screen_height, "Living Tiles", self.monitor, None)
        if not self.window:
            glfw.terminate()
            raise Exception("GLFW window can't be created")
        
        glfw.make_context_current(self.window)

        gl.glViewport(0, 0, self.screen_width, self.screen_height)
        
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glOrtho(0, self.screen_width, 0, self.screen_height, -1, 1)
        
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        self.texture = gl.glGenTextures(1)

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture)

        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        gl.glEnable(gl.GL_TEXTURE_2D)

        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_BLEND)

        self.fps_monitor = FrameRateMonitor("WINDOW")

    def resize_grid_surface(self, zoom):
        width, height = self.original_size

        new_width = max(round(width * zoom), 20)
        new_height = max(round(height * zoom), round(20 * self.ASPECT_RATIO))

        if not np.array_equal(self.current_size, (new_width, new_height)):
            self.zoom_offset[0] += (self.current_size[0] - new_width) / 2
            self.zoom_offset[1] += (self.current_size[1] - new_height) / 2

            self.current_size = np.array([new_width, new_height])

    def handle_window_events(self) -> None:
        """
        Handle GLFW events, like key presses and closing the window.
        """
        if glfw.window_should_close(self.window) or glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
            self.close()
            return

        move_x, move_y = 0, 0
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            move_y -= 1
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            move_y += 1
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            move_x += 1
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            move_x -= 1

        if glfw.get_key(self.window, glfw.KEY_SPACE) == glfw.PRESS:
            self.simulation.randomize()

        zoom = self.zoom
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            zoom *= ZOOM_MULTI
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            zoom /= ZOOM_MULTI

        if zoom != self.zoom:
            self.zoom = min(max(zoom, MIN_ZOOM), MAX_ZOOM)
            self.resize_grid_surface(self.zoom)

        length = math.sqrt(move_x ** 2 + move_y ** 2)
        if length != 0:
            move_x /= length
            move_y /= length

        self.pos_offset[0] += (move_x * CAMERA_SPEED) / self.zoom
        self.pos_offset[1] += (move_y * CAMERA_SPEED) / self.zoom
    
    def render(self, grid) -> None:
        """
        Render the pixel grid using textures.
        """
        array_height, array_width = grid.shape

        left = self.pos_offset[0] * self.zoom + self.zoom_offset[0]
        right = left + self.current_size[0]
        bottom = self.pos_offset[1] * self.zoom + self.zoom_offset[1]
        top = bottom + self.current_size[1]

        pixel_width = self.current_size[0] / array_width
        pixel_height = self.current_size[1] / array_height

        left_overflow = min(math.floor(- left / pixel_width), array_width) if left < 0 else 0
        bottom_overflow = min(math.floor(- bottom / pixel_height), array_height) if bottom < 0 else 0
        right_overflow = min(math.floor((right - self.screen_width) / pixel_width), array_width) if right > self.screen_width else 0
        top_overflow = min(math.floor((top - self.screen_height) / pixel_height), array_height) if top > self.screen_height else 0

        cropped_array_x = array_width - left_overflow - right_overflow
        cropped_array_y = array_height - top_overflow - bottom_overflow

        left += left_overflow * pixel_width
        right = left + cropped_array_x * pixel_width
        bottom += bottom_overflow * pixel_height
        top = bottom + cropped_array_y * pixel_height

        cropped_grid = crop_array(grid, (left_overflow, top_overflow), (cropped_array_x, cropped_array_y))

        #mask_grid = (self.prev_grid != cropped_grid)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        if cropped_array_x or cropped_array_y:
            start = time.time()

            if CUPY_SUPPORTED:
                color_array = cp.asnumpy(cp.asarray(TILES_COLOUR_LOOKUP)[cp.asarray(cropped_grid)]) # VERY GOOD
            else:
                color_array = TILES_COLOUR_LOOKUP[cropped_grid]

            end = time.time()
            duration = end-start
            if duration:
                print(f"FPS: {1 / duration}")

            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, cropped_array_x, cropped_array_y, 0, gl.GL_RGB, gl.GL_FLOAT, color_array)

            """
            if self.prev_grid.shape != cropped_grid.shape:
                color_array = TILES_COLOUR_LOOKUP[cropped_grid]
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, cropped_array_x, cropped_array_y, 0, gl.GL_RGB, gl.GL_FLOAT, color_array)
            else:
                color_array = TILES_COLOUR_LOOKUP[cropped_grid]
                gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, cropped_array_x, cropped_array_y, 0, gl.GL_RGB, gl.GL_FLOAT, color_array)
            """

            gl.glBegin(gl.GL_QUADS)
            gl.glTexCoord2f(0.0, 0.0)
            gl.glVertex2f(left, top)

            gl.glTexCoord2f(1.0, 0.0)
            gl.glVertex2f(right, top)

            gl.glTexCoord2f(1.0, 1.0)
            gl.glVertex2f(right, bottom)

            gl.glTexCoord2f(0.0, 1.0)
            gl.glVertex2f(left, bottom)
            gl.glEnd()
            
        glfw.swap_buffers(self.window)

        #self.prev_grid = cropped_grid

    def tick(self) -> None:
        """
        Tick (manage frame rate).
        """
        glfw.poll_events()

        self.fps_monitor.run()

    def close(self) -> None:
        """
        Close the GLFW window and terminate.
        """
        self.running = False
        self.simulation.quit()
        glfw.terminate()

    @staticmethod
    def check_gl_error():
        error = gl.glGetError()
        if error != gl.GL_NO_ERROR:
            print(f"OpenGL error: {error}")

    def main(self) -> None:
        """
        Main window loop.
        """
        while self.running:
            grid = self.simulation.get_grid()
            self.render(grid)
            self.tick()
            self.check_gl_error()
            self.handle_window_events()


### Simulation [multithread] ###
class Simulation:
    def __init__(self, events, grid_dimensions) -> None:
        self.lock = threading.Lock()

        with self.lock:
            self.events = events
            self.grid_dimensions = grid_dimensions

            self.running = True

            self.grid = np.zeros(self.grid_dimensions, dtype=np.uint8)

            self.fps_monitor = FrameRateMonitor("SIMULATION")

    def randomize(self):
        with self.lock:
            self.grid = np.random.randint(0, 2, size=self.grid_dimensions, dtype=np.uint8)

    def draw_face(self):
        """
        Draws a simple face on the grid with three lines: eyes and a mouth.
        """
        with self.lock:
            # Create a blank grid
            self.grid[:] = 0  # Clear the grid before drawing

            height, width = self.grid.shape

            # Define the size and position of the face components
            eye_size = 3
            eye_y = height // 3
            eye_x_left = width // 3
            eye_x_right = 2 * width // 3

            mouth_y = 2 * height // 3
            mouth_width = width // 3
            mouth_height = 2

            # Draw eyes
            self.grid[eye_y:eye_y + eye_size, eye_x_left:eye_x_left + eye_size] = 1  # Left eye
            self.grid[eye_y:eye_y + eye_size, eye_x_right:eye_x_right + eye_size] = 1  # Right eye

            # Draw mouth
            mouth_start_x = (width - mouth_width) // 2
            self.grid[mouth_y, mouth_start_x:mouth_start_x + mouth_width] = 1

    def update(self):
        with self.lock:
            current_grid = XP.array(self.grid, dtype=XP.uint8)

            neighbors = sum(XP.roll(XP.roll(current_grid, i, axis=0), j, axis=1)
                            for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))

            new_grid = (neighbors == 3) | ((current_grid == 1) & (neighbors == 2))

            self.grid = XP.asnumpy(new_grid.astype(XP.uint8)) if CUPY_SUPPORTED else new_grid.astype(np.uint8)

    def get_grid(self):
        with self.lock:
            return self.grid[:]
        
    def quit(self):
        with self.lock:
            self.running = False

    def main(self):
        while self.running:
            self.update()

            if self.events["exit"].is_set():
                self.running = False

### Entry point ###
def main():
    simulation = Simulation(events, GRID_DIMENSIONS)
    simulation.randomize()

    window = Window(simulation)

    simulation_thread = threading.Thread(target=simulation.main)
    simulation_thread.start()

    window.main()

    simulation_thread.join()
    sys.exit(0)

if __name__ == "__main__":
    main()