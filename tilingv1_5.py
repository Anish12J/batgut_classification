import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_filtered_tiles(image: np.ndarray, tile_size: tuple, stride: int, contrast_threshold: float, stain_threshold: float) -> list:
    """
    Create tiles and filter out low-contrast or low-stained (white background) tiles.
    :param image: Input image (height, width, channels)
    :param tile_size: Tuple (tile_h, tile_w)
    :param stride: Stride for tile extraction
    :param contrast_threshold: Threshold for tile contrast filtering
    :param stain_threshold: Proportion of stained pixels required to keep the tile
    :return: List of required tiles
    """
    img_h, img_w, _ = image.shape
    tile_h, tile_w = tile_size
    required_tiles = []

    for y in range(0, img_h - tile_h + 1, stride):
        for x in range(0, img_w - tile_w + 1, stride):
            tile = image[y:y + tile_h, x:x + tile_w]

            gray_tile = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            contrast = np.std(gray_tile)

            # Filtering tiles based on given contrast treshold value 
            if contrast < contrast_threshold:
                continue

            # Detect stained (non-white) pixels using thresholding
            mask = cv2.inRange(tile, (200, 200, 200), (255, 255, 255))
            non_stained_proportion = np.sum(mask) / (tile_h * tile_w * 255)

            # Eliminating tiles which are white
            if non_stained_proportion < stain_threshold:
                required_tiles.append(tile)

    return required_tiles

# def show_tiles(tiles: list, tile_size: tuple, border_size: int = 1, title: str = "Tile Visualization"):
#     """
#     Visualize filtered tiles.
#     """
#     num_tiles = len(tiles)
#     if num_tiles == 0:
#         print(f"No informative tiles found for {title}.")
#         return

#     cols = min(10, num_tiles)
#     rows = (num_tiles + cols - 1) // cols
#     img_h, img_w = tile_size

#     bordered_image = np.ones(((img_h + border_size) * rows, 
#                                (img_w + border_size) * cols, 3), dtype=np.uint8) * 255

#     for idx, tile in enumerate(tiles):
#         row, col = divmod(idx, cols)
#         y_start = row * (img_h + border_size)
#         x_start = col * (img_w + border_size)
#         bordered_image[y_start:y_start + img_h, x_start:x_start + img_w] = tile

#     plt.imshow(bordered_image)
#     plt.title(title)
#     plt.axis('off')
#     plt.show()

def main():

    image_path = 'sample_image.png' 
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tile_size = (32, 32)
    stride = 16
    contrast_threshold = 15.0
    stain_threshold = 0.3

    tiles_at_different_scales = []

    # Scaling the images at 4 different scales
    scales = [1.0, 1.25, 1.5, 1.75,2.0]
    scale_titles = ["100% Scale", "125% Scale", "150% Scale", "175% Scale","200% Scale"]

    for i, scale in enumerate(scales):
        # Resize image to given scales
        scaled_image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        tiles = create_filtered_tiles(scaled_image, tile_size, stride, contrast_threshold, stain_threshold)
        tiles_at_different_scales.append(tiles)

        print(f"Number of informative tiles at {scale_titles[i]}: {len(tiles)}")

        # Visualize tiles for the given scales
        #show_tiles(tiles, tile_size, title=scale_titles[i])

if __name__ == "__main__":
    main()
