from PIL import Image
import numpy as np

# Converts an image into a pointcloud centered at (0, 80, 0)
def get_image_pc_bytes(imagePath: str) -> np.ndarray:
  im = Image.open(imagePath)

  width, height = im.size

  pixels = []
  for x in range(width):
    pixels.append([])
    for y in range(height):
      pixels[x].append([])

  for x in range(width):
    for y in range(height):
      color = im.getpixel((x, y))
      pixels[x][y] = (color, (x - width // 2, 80, y - height // 2))

  ndPixels = np.zeros((width * height), dtype='float32, float32, float32, float32, float32, float32, float32')

  count = 0
  for x in range(width):
    for y in range(height):
      color, position = pixels[x][y]
      if color[3] == 0:
        continue
      ndPixels[count] = (position[0] / 40, position[1] / 40, position[2] / 40, color[0] / 255, color[1] / 255, color[2] / 255, color[3] / 255)
      count += 1

  ndPixels.resize(count)
  return ndPixels.tobytes()

if __name__ == "__main__":
  with open("splashscreenPointcloud.bin", "wb") as file:
    file.write(get_image_pc_bytes("logo.png"))
