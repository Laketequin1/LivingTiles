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
import cupy as cp

import glfw
import OpenGL.GL as gl
import numpy as np
import math
import scipy.ndimage

from src import COLOURS

### Constants ###
TPS = 30
FPS = 60

SCREEN_WIDTH = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
SCREEN_HEIGHT = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)

GRID_DIMENSIONS = (2000, 2000)

class Tile:
    def __init__(self, name, colour):
        self.name = name
        self.colour = colour

TILES = [Tile("empty", (0, 0, 0)), Tile("solid", (255, 100, 100))]
TILES_COLOUR_LOOKUP = np.array([tile.colour for tile in TILES])

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

### Rendering [main thread] ###
class Window:
    def __init__(self, simulation) -> None:
        self.simulation = simulation
        self.running = True

        # Initialize glfw
        if not glfw.init():
            raise Exception("glfw can't be initialized")

        self.screen_width = 800
        self.screen_height = 600

        # Create the OpenGL window
        self.window = glfw.create_window(self.screen_width, self.screen_height, "Living Tiles", None, None)
        if not self.window:
            glfw.terminate()
            raise Exception("glfw window can't be created")

        glfw.set_window_pos(self.window, 100, 100)
        glfw.make_context_current(self.window)

        # Set aspect ratio
        self.ASPECT_RATIO = GRID_DIMENSIONS[0] / GRID_DIMENSIONS[1]

        if self.screen_height / self.screen_width >= self.ASPECT_RATIO:
            new_height = self.screen_height
            new_width = int(new_height * self.ASPECT_RATIO)
            offset_x = (self.screen_width - new_width) // 2
            offset_y = 0
        else:
            new_width = self.screen_width
            new_height = int(new_width / self.ASPECT_RATIO)
            offset_x = 0
            offset_y = (self.screen_height - new_height) // 2

        self.original_size = np.array([new_width, new_height])
        self.current_size = self.original_size
        self.home_offset = np.array((offset_x, offset_y))

        self.pos_offset = np.array([0, 0])
        self.camera_speed = 20
        self.zoom = 1
        self.zoom_multi = 1.2

    def resize_grid_surface(self, zoom):
        width, height = self.original_size

        new_width = max(min(round(width * zoom), self.screen_width), 20)
        new_height = max(min(round(height * zoom), self.screen_height), 20)

        if new_width / new_height > self.ASPECT_RATIO:
            new_width = round(new_height * self.ASPECT_RATIO)
        else:
            new_height = round(new_width / self.ASPECT_RATIO)

        if new_width != self.current_size[0] or new_height != self.current_size[1]:
            self.current_size = (new_width, new_height)

    def handle_window_events(self) -> None:
        """
        Handle GLFW events, like key presses and closing the window.
        """
        if glfw.window_should_close(self.window):
            self.close()
            return

        move_x, move_y = 0, 0
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            move_y -= 1
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            move_y += 1
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            move_x -= 1
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            move_x += 1

        prev_zoom = self.zoom
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.zoom *= self.zoom_multi
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.zoom /= self.zoom_multi

        if self.zoom != prev_zoom:
            self.resize_grid_surface(self.zoom)

        length = math.sqrt(move_x ** 2 + move_y ** 2)
        if length != 0:
            move_x /= length
            move_y /= length

        self.pos_offset[0] += move_x * self.camera_speed
        self.pos_offset[1] += move_y * self.camera_speed

    def render(self, grid) -> None:
        """
        Render the tiles using OpenGL.
        """
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        array_width, array_height = grid.shape
        resize_width, resize_height = self.current_size

        zoom_x = resize_width / array_width
        zoom_y = resize_height / array_height

        resized_grid = scipy.ndimage.zoom(grid, (zoom_x, zoom_y), order=0)
        color_array = TILES_COLOUR_LOOKUP[resized_grid]

        # Ensure color_array is in the right format (e.g., RGB or RGBA)
        color_array = np.ascontiguousarray(color_array)

        # Generate texture
        texture = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)

        # Set texture parameters
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # Upload the texture data
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
            color_array.shape[1], color_array.shape[0], 0,
            gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, color_array
        )

        # Draw a quad that spans the screen, with the texture applied
        gl.glBegin(gl.GL_QUADS)
        
        # Top-left
        gl.glTexCoord2f(0.0, 1.0)
        gl.glVertex2f(-1.0, 1.0)

        # Bottom-left
        gl.glTexCoord2f(0.0, 0.0)
        gl.glVertex2f(-1.0, -1.0)

        # Bottom-right
        gl.glTexCoord2f(1.0, 0.0)
        gl.glVertex2f(1.0, -1.0)

        # Top-right
        gl.glTexCoord2f(1.0, 1.0)
        gl.glVertex2f(1.0, 1.0)

        gl.glEnd()

        # Unbind the texture
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        # Swap the buffers to display the rendered image
        glfw.swap_buffers(self.window)

    def tick(self) -> None:
        """
        Tick (manage frame rate).
        """
        glfw.poll_events()

    def close(self) -> None:
        """
        Close the GLFW window and terminate.
        """
        self.running = False
        self.simulation.quit()
        glfw.terminate()

    def main(self) -> None:
        """
        Main window loop.
        """
        while self.running:
            grid = self.simulation.get_grid()
            self.render(grid)
            self.tick()
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

    def randomize(self):
        with self.lock:
            self.grid = np.random.randint(0, 2, size=self.grid_dimensions, dtype=np.uint8)

    def update(self):
        with self.lock:
            current_grid = cp.array(self.grid)

        neighbors = sum(cp.roll(cp.roll(current_grid, i, axis=0), j, axis=1) 
                        for i in (-1, 0, 1) for j in (-1, 0, 1) if (i != 0 or j != 0))

        new_grid = (neighbors == 3) | ((current_grid == 1) & (neighbors == 2))

        with self.lock:
            self.grid = cp.asnumpy(new_grid.astype(cp.uint8))

    def get_grid(self):
        with self.lock:
            return self.grid[:]
        
    def quit(self):
        with self.lock:
            self.running = False

    def main(self):
        while self.running:
            #self.randomize()
            self.update()

            if self.events["exit"].is_set():
                self.running = False

            #time.sleep(0)

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