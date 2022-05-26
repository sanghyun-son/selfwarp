import sys
import math
import typing
import functools

import numpy as np
import imageio
import cv2

import torch
from matplotlib import pyplot as plt
from srwarp import grid
from srwarp import transform

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QSlider
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt

def deg2rad(theta: float) -> float:
    ret = math.pi * (theta / 180)
    return ret

def get_normal(theta: float, phi: float) -> torch.Tensor:
    theta = deg2rad(theta)
    phi = deg2rad(phi)
    ret = torch.Tensor([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ])
    return ret


class Pyramid(object):

    def __init__(self, height: float, width: float) -> None:
        self.__height = height
        self.__width = width
        self.__directions = torch.Tensor([
            [height // 2, width // 2, 1],
            [-height // 2, width // 2, 1],
            [-height // 2, -width // 2, 1],
            [height // 2, -width // 2, 1],
        ])
        return

    def solve_t(
            self,
            normal: torch.Tensor,
            center: torch.Tensor) -> typing.List[float]:

        num = normal.dot(center)
        den = self.__directions.matmul(normal)
        ret = num / den
        return ret

    def solve_p(
            self,
            normal: torch.Tensor,
            center: torch.Tensor,
            transpose: bool=False) -> torch.Tensor:

        num = normal.dot(center)
        scales = torch.Tensor([num / normal.dot(d) for d in self.__directions])
        scales.unsqueeze_(-1)
        points = scales * self.__directions
        points = points.t()
        points = torch.cat((points, torch.ones(1, points.size(1))), dim=0)

        dz = num / normal[2]
        cos = normal[2]
        sin = torch.sqrt(1 - cos.pow(2))
        u = torch.Tensor([normal[1], -normal[0]])
        u_norm = u.norm()
        if u_norm > 1e-6:
            u /= u_norm

        m = torch.Tensor([
            [cos + (1 - cos) * u[0]**2, u[0] * u[1] * (1 - cos), u[1] * sin, 0],
            [u[0] * u[1] * (1 - cos), cos + (1 - cos) * u[1]**2, -u[0] * sin, 0],
            [-u[1] * sin, u[0] * sin, cos, -dz],
            [0, 0, 0, 1],
        ])
        points = m.matmul(points)
        points = points[:2]
        if transpose:
            points.t_()

        return points

class ViewTransform(object):

    def __init__(self, offset_x: float, offset_y: float) -> None:
        p_eye = torch.Tensor([-700, 500, 1800])
        p_at = torch.Tensor([0, 0, 0])
        p_up = torch.Tensor([0, 1, 0])

        near = 100
        far = 2000
        fov = 100

        forward = p_eye - p_at
        forward /= forward.norm()
        side = torch.cross(p_up, forward)
        side /= side.norm()
        up = torch.cross(side, forward)
        camera = torch.Tensor([
            [*side, -torch.dot(p_eye, side)],
            [*up, torch.dot(p_eye, up)],
            [*forward, -torch.dot(p_eye, forward)],
            [0, 0, 0, 1],
        ])
        proj = torch.Tensor([
            [near / fov, 0, 0, 0],
            [0, near / fov, 0, 0],
            [0, 0, (near + far) / (near - far), 2 * near * far / (near - far)],
            [0, 0, -1, 0],
        ])
        self.__view = torch.matmul(proj, camera)
        self.__rescale = torch.Tensor([
            [1000, 0, 0, offset_x],
            [0, 1000, 0, offset_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        return

    @torch.no_grad()
    def transform(self, p: torch.Tensor) -> torch.Tensor:
        p = torch.matmul(self.__view, p)
        p = p / p[-1]
        p = torch.matmul(self.__rescale, p)
        p = p[:3]
        return p


class CtxPainter(QPainter):

    def __init__(self, window: QMainWindow, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__window = window
        return

    def __enter__(self):
        self.begin(self.__window)
        return self

    def __exit__(self, *args, **kwargs):
        self.end()
        return


class Interactive(QMainWindow):

    def __init__(self, app: QApplication) -> None:
        super().__init__()
        self.setStyleSheet('background-color: white;')

        self.origin = torch.zeros(3)
        self.height = 128
        self.width = 128
        resolution = app.desktop().screenGeometry()
        self.screen_h = resolution.height()
        self.screen_w = resolution.width()
        self.p = Pyramid(self.height, self.width)
        self.v = ViewTransform(self.screen_w // 2, self.screen_h // 8)
        self.bases = torch.Tensor([
            [1, 1, 1],
            [-1, 1, 1],
            [-1, -1, 1],
            [1, -1, 1],
        ])

        img_name = 'butterfly.png'
        self.img = imageio.imread(img_name)
        self.pre_rendered = {}

        # Additional widgets
        self.slider_phi = QSlider(Qt.Horizontal, self)
        self.slider_phi.setGeometry(
            3 * self.screen_w // 4,
            self.screen_h // 2 - 50,
            100,
            20,
        )
        self.slider_phi.setRange(0, 100)
        self.slider_phi.valueChanged.connect(self.update)

        self.slider_theta = QSlider(Qt.Horizontal, self)
        self.slider_theta.setGeometry(
            3 * self.screen_w // 4,
            self.screen_h // 2 + 50,
            100,
            20,
        )
        self.slider_theta.setRange(0, 360)
        self.slider_theta.valueChanged.connect(self.update)

        self.slider_z = QSlider(Qt.Vertical, self)
        self.slider_z.setGeometry(
            7 * self.screen_w // 10,
            self.screen_h // 4,
            20,
            self.screen_h // 2,
        )
        self.slider_z.setRange(50, 400)
        self.slider_z.valueChanged.connect(self.update)
        self.slider_z.setValue(100)
        self.slider_z.setInvertedAppearance(True)
        return

    def get_transform(
            self,
            points: torch.Tensor) -> typing.Tuple[np.array, int, int, int, int]:

        points_from = np.array([
            [self.img.shape[1] - 1, self.img.shape[0] - 1],
            [0, self.img.shape[0] - 1],
            [0, 0],
            [self.img.shape[1] - 1, 0],
        ])
        points_from = points_from.astype(np.float32)
        points_to = points[:, :2]
        points_to = points_to.numpy()
        m = cv2.getPerspectiveTransform(points_from, points_to)

        corners = np.array([
            [-0.5, -0.5, self.img.shape[1] - 0.5, self.img.shape[1] - 0.5],
            [-0.5, self.img.shape[0] - 0.5, -0.5, self.img.shape[0] - 0.5],
            [1, 1, 1, 1],
        ])
        corners = np.matmul(m, corners)
        corners /= corners[-1, :]
        dx = corners[0].min() + 0.5
        dy = corners[1].min() + 0.5
        w = math.ceil(corners[0].max() - dx + 0.5)
        h = math.ceil(corners[1].max() - dy + 0.5)
        mc = np.array([[1, 0, -dx], [0, 1, -dy], [0, 0, 1]])
        m = np.matmul(mc, m)
        return m, dx, dy, w, h

    @torch.no_grad()
    def get_regular(self, scale: float) -> torch.Tensor:
        directions = scale * self.bases
        directions[:, 0] *= self.width
        directions[:, 1] *= self.height
        return directions

    @torch.no_grad()
    def get_irregular(self, scales: torch.Tensor) -> torch.Tensor:
        directions = scales.unsqueeze(-1) * self.bases
        directions[:, 0] *= self.width
        directions[:, 1] *= self.height
        return directions

    @torch.no_grad()
    def transform(self, points: torch.Tensor) -> torch.Tensor:
        scale = 300
        x, y, z = points.unbind(1)
        w = torch.ones_like(x)
        points = torch.stack((x, -scale * z, y, w), dim=0)
        points = self.v.transform(points)
        points.t_()
        return points

    @torch.no_grad()
    def draw_line(
            self,
            qp: QPainter,
            p1: torch.Tensor,
            p2: torch.Tensor) -> None:

        p1 = p1.round_()
        p2 = p2.round_()
        p1 = p1.long()
        p2 = p2.long()
        qp.drawLine(p1[0], p1[1], p2[0], p2[1])
        return

    def draw_polygon(
            self,
            qp: QPainter,
            points: torch.Tensor,
            is_3d: bool=True,
            pre_rendered: typing.Optional[str]=None,
            save_jacobian: bool=False) -> None:

        warped_noalpha = None
        buffer = None
        if pre_rendered is None or pre_rendered not in self.pre_rendered:
            m, dx, dy, w, h = self.get_transform(points)
            print(m)
            if save_jacobian:
                m_inv = transform.inverse_3x3(torch.from_numpy(m))
                grid_raw, yi = grid.get_safe_projective_grid(
                    m_inv,
                    (h, w),
                    (self.img.shape[0], self.img.shape[1]),
                )
                j = transform.jacobian(m_inv, sizes=(h, w), yi=yi)
                jdet = 3.9688 / transform.determinant(j)
                jdet = jdet.log()
                buffer = jdet.new_full((h * w,), -255)
                buffer[yi] = jdet
                buffer = buffer.view(1, h, w)
                borderValue = (255, 255, 255)
            else:
                borderValue = (255, 0, 255)

            warped = cv2.warpPerspective(
                self.img,
                m,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderValue=borderValue,
            )
            warped_noalpha = warped
            visible = functools.reduce(
                np.logical_or,
                (
                    warped[..., 0] != 255,
                    warped[..., 1] != 0,
                    warped[..., 2] != 255),
            )
            alpha = 255 * visible.astype(np.uint8)
            alpha = alpha.reshape(h, w, 1)
            warped = np.concatenate((warped, alpha), axis=-1)
            qimg = QImage(warped, w, h, 4 * w, QImage.Format_RGBA8888)
            qpix = QPixmap(qimg)
            if pre_rendered is not None:
                self.pre_rendered[pre_rendered] = (dx, dy, w, h, qpix)

        else:
            dx, dy, w, h, qpix = self.pre_rendered[pre_rendered]

        qp.drawPixmap(int(dx), int(dy), w, h, qpix)

        for i, point in enumerate(points):
            imod = (i + 1) % 4
            self.draw_line(qp, point, points[imod])

        return warped_noalpha, buffer

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()
            return

        return

    @torch.no_grad()
    def paintEvent(self, e) -> None:
        regular_x1 = self.transform(self.get_regular(1))
        regular_x4 = self.transform(self.get_regular(4))
        origin = self.transform(torch.zeros(1, 3))
        origin.squeeze_()

        normal = get_normal(
            self.slider_theta.value(),
            self.slider_phi.value() / 200,
        )
        center = torch.Tensor([0, 0, self.slider_z.value() / 100])

        scales = self.p.solve_t(normal, center)
        irregular = self.transform(self.get_irregular(scales))

        points = self.p.solve_p(normal, center, transpose=True)
        points[:, 0] += 1 * self.screen_w // 5
        points[:, 1] += self.screen_h // 2

        with CtxPainter(self) as qp:
            qp.setRenderHint(QPainter.Antialiasing)

            pen_black = QPen(Qt.black, 1)
            qp.setPen(pen_black)
            self.draw_line(qp, origin, regular_x4[2])
            self.draw_line(qp, origin, regular_x4[3])

            pen_blue = QPen(Qt.blue, 2)
            qp.setPen(pen_blue)
            self.draw_polygon(qp, regular_x1, pre_rendered='x1')
            self.draw_polygon(qp, regular_x4, pre_rendered='x4')

            pen_red = QPen(Qt.red, 2)
            qp.setPen(pen_red)
            self.draw_polygon(qp, irregular)
            warped, buffer = self.draw_polygon(qp, points, is_3d=False, save_jacobian=True)
            if buffer is not None:
                imageio.imwrite('warped.png', warped)
                torch.save(buffer, 'jdet.pth')

            pen_black = QPen(Qt.black, 1)
            qp.setPen(pen_black)
            self.draw_line(qp, origin, regular_x4[0])
            self.draw_line(qp, origin, regular_x4[1])

        return


def main() -> None:
    app = QApplication(sys.argv)
    sess = Interactive(app)
    sess.showFullScreen()
    sys.exit(app.exec_())
    return

if __name__ == '__main__':
    main()