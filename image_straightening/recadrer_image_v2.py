import cv2
import fitz
import numpy as np
import opencv_hough_lines as lq



pdf_path = "D://GitHub//HOCR//test_pages//test_page_mortstatsh_1905-207.pdf"

def cyclic_intersection_pts(pts):
    """
    Sorts 4 points in clockwise direction with the first point been closest to 0,0
    Assumption:
        There are exactly 4 points in the input and
        from a rectangle which is not very distorted
    """
    if pts.shape[0] != 4:
        return None

    # Calculate the center
    center = np.mean(pts, axis=0)

    # Sort the points in clockwise
    cyclic_pts = [
        # Top-left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] < center[1]))[0][0], :],
        # Top-right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] < center[1]))[0][0], :],
        # Bottom-Right
        pts[np.where(np.logical_and(pts[:, 0] > center[0], pts[:, 1] > center[1]))[0][0], :],
        # Bottom-Left
        pts[np.where(np.logical_and(pts[:, 0] < center[0], pts[:, 1] > center[1]))[0][0], :]
    ]

    return np.array(cyclic_pts)


def drawHoughLines(image, lines, output):
    out = image.copy()
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 10000 * (-b))
        y1 = int(y0 + 10000 * (a))
        x2 = int(x0 - 10000 * (-b))
        y2 = int(y0 - 10000 * (a))
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(output, out)




# Read input
pdf_doc = fitz.open(pdf_path)
page = pdf_doc.load_page(0)  # Charger la première page

# Rendu de la page en image
pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Augmenter la résolution
color = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
color = cv2.resize(color, (0, 0), fx=0.15, fy=0.15)
# RGB to gray
gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray.png', gray)
# cv2.imwrite('output/thresh.png', thresh)
# Edge detection
edges = cv2.Canny(gray, 100, 200, apertureSize=3)
# Save the edge detected image
cv2.imwrite('edges.png', edges)


polar_lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
drawHoughLines(color, polar_lines, 'houghlines.png')
# Detect the intersection points
# https://gist.github.com/arccoder/9a73e0b2d8be1a8fd42d6026d3a7a1e1

intersect_pts = lq.hough_lines_intersection(polar_lines, gray.shape)
# Sort the points in cyclic order
intersect_pts = cyclic_intersection_pts(intersect_pts)
# Draw intersection points and save
out = color.copy()
for pts in intersect_pts:
    cv2.rectangle(out, (pts[0] - 1, pts[1] - 1), (pts[0] + 1, pts[1] + 1), (0, 0, 255), 2)
cv2.imwrite('intersect_points.png', out)



# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Fit a rotated rect
rotatedRect = cv2.minAreaRect(contours[0])
# Get rotated rect dimensions
(x, y), (width, height), angle = rotatedRect
# Get the 4 corners of the rotated rect
rotatedRectPts = cv2.boxPoints(rotatedRect)
rotatedRectPts = np.int0(rotatedRectPts)
# Draw the rotated rect on the image
out = color.copy()
cv2.drawContours(out, [rotatedRectPts], 0, (0, 255, 0), 2)
cv2.imwrite('minRect.png', out)


# List the output points in the same order as input
# Top-left, top-right, bottom-right, bottom-left
dstPts = [[0, 0], [width, 0], [width, height], [0, height]]
# Get the transform
m = cv2.getPerspectiveTransform(np.float32(intersect_pts), np.float32(dstPts))
# Transform the image
out = cv2.warpPerspective(color, m, (int(width), int(height)))
# Save the output
cv2.imwrite('page.png', out)