import numpy as np
from matplotlib import pyplot

from typing import Annotated, Any, GenericAlias
import numpy.typing as npt


LineCoordsType: GenericAlias = np.ndarray[
    (Any, 2),
    np.float64
]

PointCoordsType: GenericAlias = np.ndarray[
    (1, 2),
    np.float64
]

ProjectionType: GenericAlias = np.ndarray[
    (Any, 2),
    np.float64
]

DistanceType: GenericAlias = np.ndarray[
    (1, Any),
    np.float64
]


class DistancePointToLines:
    def __init__(

            self,

            x_lines: Annotated[
                npt.ArrayLike,
                [[float, float], ...], "X coordinates of line points"
            ],

            y_lines: Annotated[
                npt.ArrayLike,
                [[float, float], ...], "Y coordinates of line points"
            ],

            point: Annotated[
                npt.ArrayLike,
                [float, float], "X, Y coordinates of point"
            ]

    ):

        self.x_lines: LineCoordsType = np.array(x_lines)
        self.y_lines: LineCoordsType = np.array(y_lines)
        self.point: PointCoordsType = np.array(point)

    def __print_graphic(

            self,
            projection: ProjectionType,
            distance: DistanceType

    ) -> None:

        LINE_COLORS = [
            "grey",
            "blue",
            "green",
            "orange",
            "cyan",
            "olive",
            "violet",
            "maroon",
            "lightsteelblue",
            "orchid"
        ]

        frame = pyplot.figure(figsize=(10, 6))
        graphic = frame.add_subplot(111)

        step_ticks = np.arange(-10, 11, 2)

        pyplot.xticks(step_ticks, [f"${i}$" for i in step_ticks])
        pyplot.yticks(step_ticks, [f"${i}$" for i in step_ticks])

        graphic. set_title(
            "Distance from point to lines",
            fontsize=14,
            color='gray'
        )

        graphic.set_xlim(-12, 30)
        graphic.set_ylim(-12, 12)

        graphic.set_xlabel('X', fontsize=14, color='gray')
        graphic.set_ylabel('Y', fontsize=14, color='gray')

        for i, color in enumerate(LINE_COLORS):
            graphic.axline(
                (self.x_lines[i, 0], self.y_lines[i, 0]),
                (self.x_lines[i, 1], self.y_lines[i, 1]),
                color=color,
                linewidth=0.5,
                alpha=0.5,
                label=f"прямая {i+1}: {distance[i]}"
            )

            graphic.plot(
                (self.point[0], projection[i, 0]),
                (self.point[1], projection[i, 1]),
                color=color,
                linestyle=':',
                marker='o'
            )

        graphic.plot(
            self.point[0],
            self.point[1],
            'ko',
            ms=10,
            mfc='red',
            label="point"
        )

        graphic.legend(
            loc='upper right',
            shadow=True,
            ncol=1,
            title="Distance",
            title_fontsize=14
        )

        pyplot.show()

    def calculate_distance(

            self,
            print_graphic: bool = False

    ) -> tuple[ProjectionType, DistanceType] | None:

        A = self.x_lines.copy()
        B = self.y_lines.copy()
        C = self.point

        A[:, 1] = self.y_lines[:, 0]
        B[:, 0] = self.x_lines[:, 1]

        vec_a = B - A

        var_coef = np.hstack((np.fliplr(vec_a), vec_a))
        var_coef[:, 1] *= -1
        var_coef.resize(10, 2, 2)

        free_coef = np.zeros((10, 2))
        free_coef[:, 0] = (vec_a[:, 1] * A[:, 0]) - (vec_a[:, 0] * A[:, 1])
        free_coef[:, 1] = (vec_a[:, 0] * C[0]) + (vec_a[:, 1] * C[1])

        projection = np.linalg.solve(var_coef, free_coef)
        distance = np.linalg.norm(projection - C, axis=1)

        if print_graphic:
            self.__print_graphic(
                projection=projection,
                distance=distance
            )

        else:
            return projection, distance


if (__name__ == "__main__"):
    calculator = DistancePointToLines(
        x_lines=np.random.uniform(-10, 10, (10, 2)),
        y_lines=np.random.uniform(-10, 10, (10, 2)),
        point=np.random.uniform(-10, 10, (1, 2))[0]
    )

    calculator.calculate_distance(
        print_graphic=True
    )
