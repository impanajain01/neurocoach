import cv2

# Load and display a single frame
img = cv2.imread("frames/frame_0000.jpg")

if img is None:
    print("❌ Could not load frame. Check the frames folder.")
else:
    print(f"✅ Frame loaded! Size: {img.shape[1]}x{img.shape[0]} pixels")
    cv2.imshow("Cricket Frame", img)
    cv2.waitKey(0)   # press any key to close the window
    cv2.destroyAllWindows()