import pygame
import os
import sys
import time
from rpi_vision.agent.capture import PiCameraStream
import numpy as np

os.environ['SDL_FBDEV'] = "/dev/fb1"
os.environ['SDL_VIDEODRIVER'] = "fbcon"
capture_manager = PiCameraStream(resolution=(320, 320), rotation=180, preview=False)

pygame.init()
screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)
screen.fill((255,0,0))
splash = pygame.image.load(os.path.dirname(sys.argv[0])+'/bchatsplash.bmp')
screen.blit(splash, (0, 0))
pygame.display.update()
font = pygame.font.Font(None, 48)
print(screen.get_size())
capture_manager.start()

while not capture_manager.stopped:
    if capture_manager.frame is None:
        continue
    frame = capture_manager.frame
    t = time.monotonic()
    # swap red & blue channels
    npframe = np.ascontiguousarray(np.flip(np.array(frame), 2))
    # make it an image
    img = pygame.image.frombuffer(npframe, capture_manager.camera.resolution, 'RGB')
    # draw it!
    screen.blit(img, (0, 0))
    # add some text
    text_surface = font.render("Hi!", True, (255, 255, 255))
    text_position = (screen.get_width()//2, screen.get_height()-24)
    rect = text_surface.get_rect(center=text_position)
    screen.blit(text_surface, rect)
    pygame.display.update()


