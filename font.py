import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

# Define the path to the TrueType Font (.ttf) file
font_path = "arklatrs-webfont.ttf"

# Set the font size
font_size = 24

# Create a blank image to display the characters
image_width = 800
image_height = 600
image = Image.new("RGB", (image_width, image_height), color=(255, 255, 255))
draw = ImageDraw.Draw(image)

# Load the font
font = ImageFont.truetype(font_path, font_size)

# Define the starting position
x = 20
y = 20

# Define the spacing between characters
spacing = 10

# Get the ASCII code range for characters (32 to 126)
start_char = 65
end_char = 90

ascii_range = list(range(48, 58)) + [num for num in range(65, 91) if num != ord('Q')]
# Display the characters
for char_code in ascii_range:
    char = chr(char_code)

    # Draw the character on the image
    draw.text((x, y), char, font=font, fill=(0, 0, 0))

    # Print the character and its ASCII code
    print(f"Character: {char}  ASCII Code: {char_code}")

    # Update the position for the next character
    x += font.getsize(char)[0] + spacing

    # If the character goes beyond the image width, move to the next line
    if x >= image_width - font.getsize(char)[0]:
        x = 20
        y += font.getsize(char)[1] + spacing

# Convert the PIL image to OpenCV format for display
image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Display the image using OpenCV
cv2.imshow("Characters from Font", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
