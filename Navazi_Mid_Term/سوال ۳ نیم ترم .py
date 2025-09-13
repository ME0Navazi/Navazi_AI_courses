def is_inside_trapezoid(x, y, points):
    """Check if a point (x, y) is inside a trapezoid defined by 4 points."""
    def area(x1, y1, x2, y2, x3, y3):
        return abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)) / 2.0)

    A, B, C, D = points
    trapezoid_area = area(*A, *B, *C) + area(*A, *C, *D)
    sum_area = (area(x, y, *A, *B) +
                area(x, y, *B, *C) +
                area(x, y, *C, *D) +
                area(x, y, *D, *A))

    return abs(trapezoid_area - sum_area) < 1e-5

# تعریف نقاط ذوزنقه
points = [(20, 20), (40, 20), (50, 50), (10, 50)]

# نقاطی که بررسی می‌کنیم
test_points = [(25, 30), (10, 5), (40, 50)]

for pt in test_points:
    result = is_inside_trapezoid(pt[0], pt[1], points)
    print(f"Point {pt} is {'INSIDE' if result else 'OUTSIDE'} the trapezoid.")

