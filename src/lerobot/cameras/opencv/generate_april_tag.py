import cv2
import numpy as np

def generate_small_apriltag(tag_id=0, tag_size_pixels=300, border_size=100, filename="apriltag_small.png"):
    # 1. Load the standard AprilTag 36h11 dictionary
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    
    # 2. Create an empty image array for the tag
    tag_image = np.zeros((tag_size_pixels, tag_size_pixels), dtype=np.uint8)
    
    # 3. Draw the marker (black and white squares)
    tag_image = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_size_pixels, tag_image, 1)
    
    # 4. Add a large white border around it to force it to print smaller
    tag_with_border = cv2.copyMakeBorder(
        tag_image, 
        top=border_size, bottom=border_size, left=border_size, right=border_size, 
        borderType=cv2.BORDER_CONSTANT, 
        value=[255, 255, 255] # 255 is White
    )
    
    # 5. Save the image to your hard drive
    cv2.imwrite(filename, tag_with_border)
    print(f"✅ Small AprilTag generated and saved as {filename}!")
    
    # 6. Show the image on screen
    cv2.imshow("Small AprilTag", tag_with_border)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    generate_small_apriltag(tag_id=0)