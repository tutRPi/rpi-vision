# Python
import argparse
import fcntl
import io
import logging
import os
import subprocess
import sys
import zipfile
import time

import numpy as np
import pygame
import PIL.Image

CONFIDENCE_THRESHOLD = 0.5   # at what confidence level do we say we detected a thing
PERSISTANCE_THRESHOLD = 0.25  # what percentage of the time we have to have seen a thing

os.environ['SDL_FBDEV'] = "/dev/fb1"
os.environ['SDL_VIDEODRIVER'] = "fbcon"

# App
from rpi_vision.agent.capture import PiCameraStream

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# initialize the display
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--include-top', type=bool,
                        dest='include_top', default=True,
                        help='Include fully-connected layer at the top of the network.')

    parser.add_argument(
        'zip_file_name',
        help=('Specifies the name of a zip file which will contain the '
              'captured images.'))

    args = parser.parse_args()
    return args

def real_main(args, zip_f, screen, capture_manager):
    orig_fl = fcntl.fcntl(sys.stdin, fcntl.F_GETFL)
    fcntl.fcntl(sys.stdin, fcntl.F_SETFL, orig_fl | os.O_NONBLOCK)

    pygame.mouse.set_visible(False)
    screen.fill((0,0,0))
    try:
        splash = pygame.image.load(os.path.dirname(sys.argv[0])+'/bchatsplash.bmp')
        screen.blit(splash, ((screen.get_width() / 2) - (splash.get_width() / 2),
                    (screen.get_height() / 2) - (splash.get_height() / 2)))
    except pygame.error:
        pass
    pygame.display.update()

    # use the default font
    smallfont = pygame.font.Font(None, 24)
    medfont = pygame.font.Font(None, 36)
    bigfont = pygame.font.Font(None, 48)

    capture_manager.start()
    start_sample_num = 0
    img_number = 0
    zip_file_base = os.path.splitext(os.path.basename(args.zip_file_name))[0]

    timestamp = time.monotonic()
    is_recording = False
    print('')
    print('Ready. Press ENTER to toggle capture/standby. Press Ctrl+C to quit.')
    while not capture_manager.stopped:
        if capture_manager.frame is None:
            continue
        frame = capture_manager.read()
        # get the raw data frame & swap red & blue channels
        previewframe = np.ascontiguousarray(np.flip(np.array(capture_manager.frame), 2))
        # make it an image
        img = pygame.image.frombuffer(previewframe, capture_manager.camera.resolution, 'RGB')
        # draw it!
        screen.blit(img, (0, 0))
        if is_recording:
            zip_f_io = zip_f.open(
                '{0}/{0}-{1}.png'.format(zip_file_base, img_number),
                mode='w')
            img_bytes = bytes(img.get_view('3'))
            img_size = (img.get_height(), img.get_width())
            bio = io.BytesIO()
            PIL.Image.frombuffer('RGB', img_size, img_bytes).transpose(PIL.Image.ROTATE_270).save(
                bio, 'PNG')
            zip_f_io.write(bio.getbuffer())
            zip_f_io.close()
            img_number += 1

        # add FPS on top corner of image
        new_timestamp = time.monotonic()
        delta = new_timestamp - timestamp
        timestamp = new_timestamp
        fpstext = "total: %04d frames" % (img_number,)
        if is_recording:
            fpstext = 'this run: %04d; %s' % (
                img_number - start_sample_num, fpstext)
        fpstext_surface = smallfont.render(fpstext, True, (255, 0, 0))
        fpstext_position = (screen.get_width()-10, 10) # near the top right corner
        screen.blit(fpstext_surface, fpstext_surface.get_rect(
            topright=fpstext_position))

        status = 'RECORD' if is_recording else 'STANDBY'
        status_color = (255, 0, 0)
        status_surface = bigfont.render(status, True, status_color)
        status_position = (screen.get_width() // 2,
                               screen.get_height() - bigfont.size(status)[1])
        screen.blit(status_surface, status_surface.get_rect(
            center=status_position))

        k = sys.stdin.read(1)
        while k:
            if k == '\n':
                is_recording = not is_recording
                start_sample_num = img_number
            k = sys.stdin.read(1)

        pygame.display.update()

def main():
    args = parse_args()
    if not args.zip_file_name.endswith('.zip'):
        logging.error('zip file should just end with plain ".zip"; got %s',
                      args.zip_file_name)
        sys.exit(2)

    zip_f = zipfile.ZipFile(args.zip_file_name, mode='w')
    try:
        pygame.init()
        screen = pygame.display.set_mode((0,0), pygame.FULLSCREEN)
        capture_manager = PiCameraStream(
            resolution=(screen.get_width(), screen.get_height()),
            rotation=180, preview=False)
        try:
            real_main(args, zip_f, screen, capture_manager)
        finally:
            capture_manager.stop()
    finally:
        zip_f.close()
        if os.environ['SUDO_UID'] and os.environ['SUDO_GID']:
            os.chown(args.zip_file_name,
                     int(os.environ['SUDO_UID']), int(os.environ['SUDO_GID']))


if __name__ == "__main__":
    main()
