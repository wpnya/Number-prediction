import pygame
import math
import joblib

from dependeces import useful_factors
from PIL import Image
import numpy as np
import pandas as pd
import csv

model_decision_forest = joblib.load('model_decision_forest')


def num_predict():
    image = Image.open('screenshot.png')

    image = image.convert('L')

    image_array = np.array(image)

    image_vector = image_array.reshape(784)

    with open('image_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)

        header = ['pixel{}'.format(i) for i in range(784)]
        writer.writerow(header)

        row = list(image_vector)
        writer.writerow(row)

    image = pd.read_csv('image_data.csv')
    image = image[useful_factors]

    print(model_decision_forest.predict(image)[0])


pygame.init()

window_width = 600
window_height = 600

grid_width = 28
grid_height = 28
game_area_size = (28, 28)

pixel_size = min(window_width // grid_width, window_height // grid_height)

offset_x = (window_width - grid_width * pixel_size) // 2
offset_y = (window_height - grid_height * pixel_size) // 2

screen = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Pixel Drawing App")

drawing = False

pixels = [[0] * grid_height for _ in range(grid_width)]
pixel_count = 0


def screen_clear():
    screen.fill((0, 0, 0))
    for x in range(grid_width):
        for y in range(grid_height):
            pixels[x][y] = 0


def calculate_color_intensity(distance):
    max_distance = 3
    intensity = max(0, 255 - (distance / max_distance) * 255)
    return int(intensity)


def save_screenshot():
    game_area_surface = pygame.Surface(game_area_size)
    for x in range(grid_width):
        for y in range(grid_height):
            color = (pixels[x][y], pixels[x][y], pixels[x][y])
            pygame.draw.rect(game_area_surface, color, (x, y, 1, 1))
    pygame.image.save(game_area_surface, f"screenshot.png")
    num_predict()


running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            drawing = True
        elif event.type == pygame.MOUSEBUTTONUP:
            drawing = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                screen_clear()
                pixel_count = 0

    if drawing:
        x, y = pygame.mouse.get_pos()
        grid_x = (x - offset_x) // pixel_size
        grid_y = (y - offset_y) // pixel_size
        if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
            for dx in range(-5, 6):
                for dy in range(-5, 6):
                    nx, ny = grid_x + dx, grid_y + dy
                    if 0 <= nx < grid_width and 0 <= ny < grid_height:
                        distance = math.sqrt(dx ** 2 + dy ** 2)
                        intensity = calculate_color_intensity(distance)
                        if pixels[nx][ny] < intensity:
                            pixels[nx][ny] = intensity
                            pixel_count += 1
                            if pixel_count % 20 == 0:
                                save_screenshot()

    for x in range(grid_width):
        for y in range(grid_height):
            color = (pixels[x][y], pixels[x][y], pixels[x][y])
            pygame.draw.rect(screen, color,
                             (offset_x + x * pixel_size, offset_y + y * pixel_size, pixel_size, pixel_size))

    pygame.display.flip()

pygame.quit()
