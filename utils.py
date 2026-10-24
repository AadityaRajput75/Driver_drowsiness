from math import dist, isfinite

# 68-point facial landmark indices (used by OpenCV LBF and dlib)
LEFT_EYE_INDICES = [42, 43, 44, 45, 46, 47]
RIGHT_EYE_INDICES = [36, 37, 38, 39, 40, 41]

def eye_aspect_ratio(eye):
    """
    Calculates the Eye Aspect Ratio (EAR) given 6 eye landmarks.
    eye is a list or array of 6 points:
    [p1 (left corner), p2 (top left), p3 (top right), p4 (right corner), p5 (bottom right), p6 (bottom left)]
    """
    if len(eye) != 6 or not all(isfinite(float(v)) for point in eye for v in point):
        return None
    width = dist(eye[0], eye[3])
    if width < 1e-6:
        return None
    return (dist(eye[1], eye[5]) + dist(eye[2], eye[4])) / (2.0 * width)


def mouth_aspect_ratio(points):
    """Six inner-lip points in the same order as the eye landmarks."""
    return eye_aspect_ratio(points)
