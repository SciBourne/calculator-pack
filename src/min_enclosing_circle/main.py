from typing import Annotated, Any, ClassVar

import numpy as np
from matplotlib import pyplot


class MinEnclosingCircle:
    """
    Calculate minimum enclosing circle


    Examples:

    1. Object initialization:

    >>> object = MinEnclosingCircle()


    2. Initializing an object and specifying the type of numbers
    to store the coordinates of points:

    >>> object = MinEnclosingCircle(type_array=np.int32)

    """

    MSG_ERR: ClassVar[tuple[str, ...]] = (
        ("The array of points must include at least two points! "
         "Use add() method to add points."),

        ("The arguments must include at least one point "
         "coordinates: x, y!"),

        ("Wrong number of points coordinates! "
         "The total must be even for a two-dimensional space."),

        ("Coordinates must be numeric!"),

        ("If the argument of this method is a single object,"
         "then it must be of type ndarray, tuple, or list")
    )

    type_array: Annotated[
        np.floating,
        "Coordinates type"
    ]

    points: Annotated[
        np.ndarray[(Any, 2), np.floating],
        "Point coordinates"
    ]

    center: Annotated[
        tuple[float, float] | None,
        "Value of center coordinates"
    ]

    radius: Annotated[
        float | None,
        "Value of radius"
    ]

    switch_complete: Annotated[
        bool,
        "Whether the circle has already been calculated"
    ]

    def __init__(

            self,

            type_array: Annotated[
                np.floating,
                "Coordinates type"
            ] = np.float64

    ):

        self.type_array = type_array
        self.points = np.array([], dtype=self.type_array)
        self.center = None
        self.radius = None
        self.switch_complete = False

    def __str__(self) -> str:
        self.__reshape_(self.points)

        return ("[CURRENT RESULT]\n\n"
                f"Center: {self.center}\n"
                f"Radius: {self.radius}\n\n"

                "[CURRENT SET OF POINTS]\n\n"
                f"Total: {self.points.shape[0]}\n\n"
                f"{self.points}")

    def __len__(self) -> int:
        self.__reshape_(self.points)
        return self.points.shape[0]

    def __reshape_(self, points: np.ndarray) -> None:
        if len(points.shape) == 1:
            points.shape = (points.shape[0] // 2, 2)

    def __section_(self,
                   points: np.ndarray) -> tuple[np.ndarray, np.floating]:

        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[0] - points[1]) / 2

        return center, radius

    def __triangle_(self,
                    points: np.ndarray) -> tuple[np.ndarray, np.floating]:

        vectors = np.array([points[0] - points[1],
                            points[1] - points[2],
                            points[2] - points[0]])

        index_except = np.argmax(np.linalg.norm(vectors, axis=1))
        exception = np.delete(points, index_except - 1, axis=0)

        def target_angle() -> np.floating:
            direct_vectors = exception - points[index_except - 1]
            unit_vectors = direct_vectors / np.linalg.norm(direct_vectors)

            angle = np.arccos(
                np.clip(np.linalg.multi_dot(unit_vectors) /
                        np.prod(np.linalg.norm(unit_vectors, axis=0)), -1, 1)
            )

            return np.degrees(angle)

        if target_angle() >= 90:
            return self.__section_(points=exception)
        else:
            midpoint_a = (points[0] + points[1]) / 2
            midpoint_b = (points[1] + points[2]) / 2

            normal_vec_a = [-vectors[0, 1], vectors[0, 0]]
            normal_vec_b = [-vectors[1, 1], vectors[1, 0]]

            var_coef = np.array([[normal_vec_a[1], -normal_vec_a[0]],
                                 [normal_vec_b[1], -normal_vec_b[0]]])

            free_coef = np.array([normal_vec_a[1] * midpoint_a[0] -
                                  normal_vec_a[0] * midpoint_a[1],

                                  normal_vec_b[1] * midpoint_b[0] -
                                  normal_vec_b[0] * midpoint_b[1]])

            center = np.linalg.solve(var_coef, free_coef)
            radius = np.linalg.norm(center - points[0])

            return center, radius

    def __welzl_recursion(

            self,
            points: np.ndarray,
            test_points: np.ndarray

    ) -> tuple[np.ndarray, np.floating]:

        self.__reshape_(test_points)

        if points.shape[0] == 0 or test_points.shape[0] == 3:
            if test_points.shape[0] == 0:
                return [0, 0], 0

            elif test_points.shape[0] == 1:
                return test_points, 0

            elif test_points.shape[0] == 2:
                return self.__section_(test_points)

            else:
                return self.__triangle_(test_points)

        point = points[0]
        points = np.delete(points, 0, axis=0)

        circle = self.__welzl_recursion(points, test_points)

        if np.linalg.norm(point - circle[0]) <= circle[1]:
            return circle
        else:
            test_points = np.append(test_points, point)
            return self.__welzl_recursion(points, test_points)

    def __welzl_iteration(

            self,
            points: np.ndarray

    ) -> tuple[np.ndarray, np.floating]:

        stack = [{'shift': 0,
                  'center': (0, 0),
                  'radius': 0,
                  'test points': (),
                  'point': points.shape[0],
                  'points': points.shape[0]}]

        while True:
            frame = stack[-1]

            if frame['shift'] == 0:

                if frame['points'] == 0 or len(frame['test points']) == 3:
                    if len(frame['test points']) == 0:

                        stack.pop()
                        frame = stack[-1]
                        continue

                    elif len(frame['test points']) == 1:
                        circle_center = self.points[frame['test points']]

                        stack.pop()
                        frame = stack[-1]
                        frame['center'] = tuple(circle_center)
                        frame['radius'] = 0
                        continue

                    elif len(frame['test points']) == 2:
                        circle = self.__section_(
                            np.array([self.points[frame['test points'][0]],
                                      self.points[frame['test points'][1]]]))

                        stack.pop()
                        frame = stack[-1]
                        frame['radius'] = circle[1]
                        frame['center'] = tuple(circle[0])
                        continue

                    elif len(frame['test points']) == 3:
                        circle = self.__triangle_(
                            np.array([self.points[frame['test points'][0]],
                                      self.points[frame['test points'][1]],
                                      self.points[frame['test points'][2]]]))

                        stack.pop()
                        frame = stack[-1]
                        frame['center'] = tuple(circle[0])
                        frame['radius'] = circle[1]
                        continue

                frame['point'] -= 1
                frame['points'] -= 1
                frame['shift'] = 1

                stack.append(frame.copy())
                stack[-1]['shift'] = 0
                continue

            elif frame['shift'] == 1:

                if np.linalg.norm(
                        self.points[frame['point']] -
                        frame['center']) <= frame['radius']:

                    if len(stack) == 1:
                        return frame['center'], frame['radius']

                    circle = (frame['center'], frame['radius'])

                    stack.pop()
                    frame = stack[-1]
                    frame['center'] = circle[0]
                    frame['radius'] = circle[1]
                    continue

                else:
                    frame['test points'] = (*frame['test points'],
                                            frame['point'])
                    frame['shift'] = 2
                    stack.append(frame.copy())
                    stack[-1]['shift'] = 0
                    continue

            elif frame['shift'] == 2:

                circle = (frame['center'], frame['radius'])

                if len(stack) == 1:
                    return frame['center'], frame['radius']
                else:
                    stack.pop()
                    frame = stack[-1]
                    frame['center'] = circle[0]
                    frame['radius'] = circle[1]

                continue

    def add(

            self,

            *points: Annotated[
                int | float |
                list[int | float] |
                tuple[int | float, ...] |
                np.ndarray[(1, Any), np.dtype],

                "Point coordinates"
            ]

    ) -> None:
        """
        Add point coordinates to point array.


        Examples:

        1. Adding one point:

        >>> object.add(1.0, 2.1)


        2. Adding three points:

        >>> object.add(1.0, 2.1, 3, 4, 5.7, 8.4, 9.2)


        3. Adding two points using a tuple:

        >>> points = (1, 2, 3, 4)
        >>> object.add(points)


        4. Adding three points using a matrix 3x2:

        >>> points = [[1, 2], [3, 4], [5, 6]]
        >>> object.add(points)


        5. Adding points using NumPy array:

        >>> points = np.array([1, 2, 3, 4])
        >>> object.add(points)

        """

        assert len(points) != 0, self.MSG_ERR[1]

        def write_to_points(points: np.ndarray) -> None:
            points = points.flatten()
            assert points.size % 2 == 0, self.MSG_ERR[2]
            assert type(points[0]) != np.dtype('str_'), self.MSG_ERR[3]

            self.points = np.append(self.points, points)
            self.switch_complete = False

        if len(points) > 1:
            write_to_points(np.array(points))
        else:
            assert isinstance(
                *points, (np.ndarray, tuple, list)
            ), self.MSG_ERR[4]

            if isinstance(*points, np.ndarray):
                write_to_points(points[0])
            else:
                write_to_points(np.array(*points))

    def remove(

            self,

            *points: Annotated[
                int | float |
                list[int | float] |
                tuple[int | float, ...] |
                np.ndarray[(1, Any), np.dtype],

                "Point coordinates"
            ]

    ) -> None:
        """
        Remove points from point array.


        Examples:

        1. Removing one point:

        >>> object.remove(1.0, 2.1)


        2. Removing three points:

        >>> object.remove(1.0, 2.1, 3, 4, 5.7, 8.4, 9.2)


        3. Removing two points using a tuple:

        >>> points = (1, 2, 3, 4)
        >>> object.remove(points)


        4. Removing three points using a matrix 3x2:

        >>> points = [[1, 2], [3, 4], [5, 6]]
        >>> object.remove(points)


        5. Removing points using NumPy array:

        >>> points = np.array([1, 2, 3, 4])
        >>> object.remove(points)


        NOTE: if any point to be deleted was not found,
              then it will be ignored and no error will occur.

        """

        assert len(points) != 0, self.MSG_ERR[1]

        def del_points(points: np.ndarray) -> None:
            points = points.flatten()
            assert points.size % 2 == 0, self.MSG_ERR[2]
            assert type(points[0]) != np.dtype('str_'), self.MSG_ERR[3]

            index_remove = []
            self.__reshape_(self.points)
            self.__reshape_(points)

            for x, y in points:
                index_remove.append(
                    np.where((self.points[:, 0] == x) &
                             (self.points[:, 1] == y))
                )

            self.points = np.delete(self.points, index_remove, axis=0)
            self.switch_complete = False

        if len(points) > 1:
            del_points(np.array(points))
        else:
            assert isinstance(
                *points, (np.ndarray, tuple, list)
            ), self.MSG_ERR[4]

            if isinstance(*points, np.ndarray):
                del_points(points[0])
            else:
                del_points(np.array(*points))

    def clear(self) -> None:
        """
        Clear point array.


        Examples:

        1. Removing all points:

        >>> object.clear()

        """

        self.points = np.array([], dtype=self.type_array)
        self.switch_complete = False

    def circle(

            self,

            imp: Annotated[
                str | None,

                """
                Possibility of choosing the implementation
                of the Emo Welzl algorithm, default (imp=None)
                the iterative version is used

                Change to imp='recursive' for classical version

                """
            ] = None,

            printing: Annotated[
                bool,

                """
                Switch to printing flag for view result,
                when set to False, this method will return
                the value tuple

                """
            ] = True

    ) -> Annotated[
        tuple[tuple[float, float], float] | None,

        """
        Return ((1.0, 1.0), 0.5)
        where (1.0, 1.0) is center
        and 0.5 is radius

        """
    ]:
        """
        Calculate minimum enclosing circle


        Examples:

        1. Calculating and printing the center with the radius
           of the minimum circle:

        >>> object.circle()

        ; Center: (3.0, 4.0)
        ; Radius: 2.8284271247461903


        2. Similarly, but using the recursive version of Welzl algorithm:

        >>> object.circle(imp='recursion')

        ; Center: (3.0, 4.0)
        ; Radius: 2.8284271247461903


        3. Calculating and returning values as a tuple:

        >>> result = object.circle(printing=False)
        >>> print(result)

        ; ((3.0, 4.0), 2.8284271247461903)

        """

        self.__reshape_(self.points)
        assert self.points.shape[0] >= 2, self.MSG_ERR[0]

        if not self.switch_complete:

            if self.points.shape[0] == 2:
                self.center, self.radius = self.__section_(self.points)
                self.switch_complete = True

            elif self.points.shape[0] == 3:
                self.center, self.radius = self.__triangle_(self.points.copy())
                self.switch_complete = True

            elif imp == 'recursion':
                self.center, self.radius = self.__welzl_recursion(
                    np.random.permutation(self.points), np.array([])
                )

                self.switch_complete = True

            else:
                self.center, self.radius = self.__welzl_iteration(
                    np.random.permutation(self.points)
                )

                self.switch_complete = True

        if printing:
            print(("Center: {}\n"
                   "Radius: {}")
                  .format(self.center, self.radius))
        else:
            return self.center, self.radius

    def plot(self) -> None:
        """
        Print all points and their minimum enclosing circle using Matplotlib.


        Examples:

        1. Printing plot:

        >>> object.plot()

        """

        if not self.switch_complete:
            self.circle(printing=False)

        figure = pyplot.figure(figsize=(12, 12), facecolor='#EAEAEF')
        axes = figure.add_subplot(facecolor='#EAEAEF')

        angle = np.linspace(0, 2 * np.pi, 100)

        axes.minorticks_on()
        axes.grid(which='major', linewidth=0.3)
        axes.grid(which='minor', linewidth=0.1)

        axes.fill(
            self.center[0] + (self.radius * np.cos(angle)),
            self.center[1] + (self.radius * np.sin(angle)),
            alpha=0.25,
            edgecolor='black',
            facecolor='#AAAAFF',
            linewidth=1.5, label="Circle"
        )

        axes.plot(*self.center, 'r+', label="Center")
        axes.plot(self.points[:, 0], self.points[:, 1], 'ko', label="Points")

        axes.set_title(
            "Minimum Enclosing Circle",
            color='#505050',
            fontsize=24,
            pad=20
        )

        axes.legend(loc='upper right')

        axes.text(
            0.045, 0.92,

            ("Center:  ({0:.2}; {1:.2})\n"
             "Radius:  {2:.3}").format(self.center[0],
                                       self.center[1],
                                       self.radius),

            fontsize=12,
            transform=axes.transAxes,
            bbox=dict(
                boxstyle='round4',
                facecolor='#F7EEFF',
                edgecolor='#AAAAAA',
                pad=1.2
            )
        )

        axes.set_aspect(1)
        pyplot.show()


if (__name__ == "__main__"):
    min_circle = MinEnclosingCircle()

    min_circle.add(np.random.randn(50, 2) - 0.3)

    min_circle.circle()
    min_circle.plot()
