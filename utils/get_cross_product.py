def get_cross_product(p1, p2, p):
    """
    Use for determining which side of the line a point p is on.
    p1, p2 are two points defining a line segment.
    Returns cross product of vectors p1p and p1p2.

    If cross > 0, p is on the left side of the line.
    If cross < 0, p is on the right side of the line.
    If cross = 0, p is on the line.
    """

    return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])
