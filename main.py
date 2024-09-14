### Imports ###
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = 'hide'

import pygame
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
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)

        pygame.display.set_caption("Living Tiles")

        #self.grid_surface = pygame.Surface(GRID_DIMENSIONS)

        self.ASPECT_RATIO = GRID_DIMENSIONS[0] / GRID_DIMENSIONS[1]

        if SCREEN_HEIGHT / SCREEN_HEIGHT >= self.ASPECT_RATIO:
            new_height = SCREEN_HEIGHT
            new_width = int(new_height * self.ASPECT_RATIO)
            offset_x = (SCREEN_WIDTH - new_width) // 2
            offset_y = 0
        else:
            new_width = SCREEN_WIDTH
            new_height = int(new_width / self.ASPECT_RATIO)
            offset_x = 0
            offset_y = (SCREEN_HEIGHT - new_height) // 2

        self.origional_size = np.array([new_width, new_height])
        self.current_size = self.origional_size
        self.home_offset = np.array((offset_x, offset_y))

        self.grid_surface = pygame.Surface(self.origional_size)

        self.pos_offset = np.array([0, 0])
        self.camera_speed = 20

        self.zoom = 1
        self.zoom_multi = 1.2

    def resize_grid_surface(self, zoom):
        width, height = self.origional_size

        new_width = max(min(round(width * zoom), SCREEN_WIDTH), 20)
        new_height = max(min(round(height * zoom), SCREEN_HEIGHT), 20)

        if new_width / new_height > self.ASPECT_RATIO:
            new_width = round(new_height * self.ASPECT_RATIO)
        else:
            new_height = round(new_width / self.ASPECT_RATIO)

        if new_width != self.current_size[0] or new_height != self.current_size[1]:
            self.current_size = (new_width, new_height)
            self.grid_surface = pygame.Surface(self.current_size)

    def handle_window_events(self) -> None:
        """
        Handles pygame events, closes the window.
        """        
        keys_pressed = pygame.key.get_pressed()
        move_x, move_y = 0, 0

        if keys_pressed[pygame.K_w]:
            move_y -= 1
        if keys_pressed[pygame.K_s]:
            move_y += 1
        if keys_pressed[pygame.K_a]:
            move_x -= 1
        if keys_pressed[pygame.K_d]:
            move_x += 1
        
        prev_zoom = self.zoom

        if keys_pressed[pygame.K_q]:
            self.zoom *= self.zoom_multi
        if keys_pressed[pygame.K_e]:
            self.zoom /= self.zoom_multi

        if self.zoom != prev_zoom:
            self.resize_grid_surface(self.zoom)

        length = math.sqrt(move_x ** 2 + move_y ** 2)
        if length != 0:
            move_x /= length
            move_y /= length

        self.pos_offset[0] += move_x * self.camera_speed
        self.pos_offset[1] += move_y * self.camera_speed

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.close()

    def render(self, grid) -> None:
        """
        Renders the tiles.
        """
        self.screen.fill(COLOURS.DARKGRAY)

        array_width, array_height = grid.shape
        resize_width, resize_height = self.current_size

        zoom_x = resize_width / array_width
        zoom_y = resize_height / array_height

        resized_grid = scipy.ndimage.zoom(grid, (zoom_x, zoom_y), order=0)
        #resized_grid = np.clip(resized_grid, 0, TILES_COLOUR_LOOKUP.shape[0] - 1).astype(int)
        #resized_grid = crop_array(resized_grid, (0, 0), self.current_size)

        color_array = TILES_COLOUR_LOOKUP[resized_grid]

        print(color_array.shape)
        print(self.current_size)

        pygame.surfarray.blit_array(self.grid_surface, color_array)
        #pygame.transform.scale(self.grid_surface, self.current_size * 2, self.grid_surface)
        self.screen.blit(self.grid_surface, self.home_offset + self.pos_offset)
        
        pygame.display.flip()

    def tick(self) -> None:
        """
        Tick
        """
        self.clock.tick(FPS)

    def close(self) -> None:
        """
        Close the pygame window, and end the loop
        """
        self.running = False
        self.simulation.quit()
        pygame.quit()

    def main(self) -> None:
        """
        Main window loop
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