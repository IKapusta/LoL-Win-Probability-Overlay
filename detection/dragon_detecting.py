import cv2

def count_dragon_icons(image_path, debug=False):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    # Apply morphological operations to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

    if debug:
        debug_img = img.copy()
        cv2.drawContours(debug_img, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow('Detected icons', debug_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return len(filtered_contours)

if __name__ == "__main__":
    count = count_dragon_icons('./scoreboards/area_1.png', debug=True)
    print(f'Detected {count} icons')
