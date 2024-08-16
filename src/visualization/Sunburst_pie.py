import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

def pie_chart_with_rectangles(percentages, labels, radius=1, aspect_ratio=1):
    """
    Creates a pie chart with optimized rectangles representing labels using
    scipy.optimize.minimize. Rectangles are initially sized to their arc area,
    and their centers are fixed at the midpoint of their corresponding arcs.

    Args:
        percentages: A list of percentages for each slice of the pie.
        labels: A list of labels corresponding to the percentages.
        radius: The radius of the pie chart.
        aspect_ratio: The desired aspect ratio (width/height) for the rectangles.
    """

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.axis("equal")  # Ensure circular shape

    # Create pie slices (using plt.pie)
    wedges, texts, autotexts = ax.pie(
        percentages,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#C0C0C0", "#FF0000", "#00FF00", "#0000FF", "#FFFF00"],
        radius=radius,
    )

    # Calculate rectangle positions and initial sizes
    rectangle_positions = []
    for i, wedge in enumerate(wedges):
        center_x, center_y = wedge.center
        angle = (wedge.theta1 + wedge.theta2) / 2
        angle_rad = np.deg2rad(angle)

        # Calculate bisector line point (center of the rectangle)
        bisector_x = center_x + radius * np.cos(angle_rad) * 0.5
        bisector_y = center_y + radius * np.sin(angle_rad) * 0.5

        # Initial rectangle size (area = arc area)
        arc_area = 0.5 * (wedge.theta2 - wedge.theta1) / 360 * np.pi * radius**2
        initial_width = np.sqrt(arc_area)
        initial_height = np.sqrt(arc_area)

        rectangle_positions.append(
            (bisector_x, bisector_y, initial_width, initial_height)
        )  # (x, y, width, height)

    # Sort rectangles by initial area (largest to smallest)
    rectangle_positions.sort(key=lambda rect: rect[2] * rect[3], reverse=True)

    # Optimize rectangles recursively, starting with the largest
    optimized_rectangle_positions = []
    for i, rect in enumerate(rectangle_positions):
        x, y, initial_width, initial_height = rect
        width = initial_width  # Start with initial width

        def objective_function_for_rect(width):
            height = width / aspect_ratio
            penalty = 0

            # Proportional Penalty for Leaving the Circle
            if not is_rectangle_within_circle((x, y, width, height), radius):
                outside_height, outside_width = calculate_outside_dimensions(
                    (x, y, width, height), radius
                )
                penalty += (outside_height**2 + outside_width**2) * 5  # Proportional penalty

            # Proportional Penalty for Overlap
            for j in range(i):  # Check overlap with previously optimized rectangles
                prev_rect = optimized_rectangle_positions[j]
                if is_rectangles_overlapping(
                    (x, y, width, height),
                    (
                        prev_rect[0],
                        prev_rect[1],
                        prev_rect[2],
                        prev_rect[3],
                    ),
                ):
                    overlap_height, overlap_width = calculate_overlap_dimensions(
                        (x, y, width, height),
                        (
                            prev_rect[0],
                            prev_rect[1],
                            prev_rect[2],
                            prev_rect[3],
                        ),
                    )
                    penalty += (
                        (overlap_height**2 + overlap_width**2)
                        * 1000
                        * (
                            prev_rect[2] * prev_rect[3] / (width * height)
                        )  # Scaled penalty
                    )

            return penalty

        # Optimize width (using only the penalty function)
        result = minimize(
            objective_function_for_rect,
            width,
            method="SLSQP",
            bounds=[(0.4, radius * 2)],  # Bound width
        )

        optimized_rectangle_positions.append(
            (x, y, result.x[0], result.x[0] / aspect_ratio)
        )

    # Draw optimized rectangles
    for i, rect in enumerate(optimized_rectangle_positions):
        x, y, width, height = rect
        rect_patch = plt.Rectangle(
            (x - width / 2, y - height / 2),
            width,
            height,
            color="black",
            fill=False,  # Don't fill rectangles
        )
        ax.add_patch(rect_patch)
        ax.text(
            x,
            y,
            labels[i],
            ha="center",
            va="center",
            fontsize=8,
        )

    plt.title("Pie Chart with Rectangles", fontsize=18)
    plt.tight_layout()
    plt.show()


# Function to check if a rectangle is entirely within the circle
def is_rectangle_within_circle(rect, radius):
    x, y, width, height = rect
    # Check if any corner of the rectangle is outside the circle
    for dx in [-width / 2, width / 2]:
        for dy in [-height / 2, height / 2]:
            if (x + dx - radius) ** 2 + (y + dy - radius) ** 2 > radius ** 2:
                return False
    return True


# Function to check if two rectangles overlap
def is_rectangles_overlapping(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    return (
        x1 < x2 + w2
        and x1 + w1 > x2
        and y1 < y2 + h2
        and y1 + h1 > y2
    )


# Function to calculate the outside dimensions of a rectangle outside the circle
def calculate_outside_dimensions(rect, radius):
    x, y, width, height = rect
    outside_height = 0
    outside_width = 0
    for dx in [-width / 2, width / 2]:
        for dy in [-height / 2, height / 2]:
            if (x + dx - radius) ** 2 + (y + dy - radius) ** 2 > radius ** 2:
                outside_height += 1
                outside_width += 1
    return outside_height, outside_width


# Function to calculate the overlap dimensions of two rectangles
def calculate_overlap_dimensions(rect1, rect2):
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    overlap_height = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
    overlap_width = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
    return overlap_height, overlap_width


# Example usage
percentages = [20, 30, 15, 10, 25]
labels = ["Label 1", "Label 2", "Label 3", "Label 4", "Label 5"]

pie_chart_with_rectangles(percentages, labels, aspect_ratio=2)  # Set aspect ratio
