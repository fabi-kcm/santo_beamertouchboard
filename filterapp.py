import pygame as pyg
import win32api
import win32con
import win32gui

import config as C


class FilterApp:
    def __init__(self):
        self.screen = pyg.display.set_mode(C.MONITOR_RESOLUTION, pyg.FULLSCREEN)
        self.aux_surface = pyg.Surface(C.MONITOR_RESOLUTION, flags=pyg.SRCALPHA)
        self.TRANSPARENCY_COLOR = (1, 1, 1)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        hwnd = pyg.display.get_wm_info()["window"]
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, win32gui.GetWindowLong(
            hwnd, win32con.GWL_EXSTYLE) | win32con.WS_EX_LAYERED
        )
        win32gui.SetLayeredWindowAttributes(hwnd, win32api.RGB(*self.TRANSPARENCY_COLOR), 0, win32con.LWA_COLORKEY)
        self.screen.fill((0, 0, 255))
        self.screen.fill(self.TRANSPARENCY_COLOR)

        self.click_aux_surf = pyg.Surface(C.MONITOR_RESOLUTION, flags=pyg.SRCALPHA)

    def overlay_at(self, pos: tuple[int, int]):
        self.screen.fill(self.TRANSPARENCY_COLOR)
        # pyg.draw.circle(self.screen, self.WHITE, pos, int(0.2*C.MONITOR_RESOLUTION[1]))
        # pyg.draw.circle(self.screen, (255, 0, 0), pos, 5)

    def draw_arrows(self, points_chain):
        self.aux_surface.fill((0, 0, 0, 0))  # Clear auxiliary surface with transparency
        if len(points_chain) < 2:
            return
        for i in range(len(points_chain) - 1):
            start = points_chain[i]
            end = points_chain[i + 1]
            pyg.draw.line(self.aux_surface, self.WHITE, start, end)
            offset = (end[0] - start[0], end[1] - start[1])
            pyg.draw.polygon(self.aux_surface, self.WHITE, [
                end,
                (end[0] - 0.2 * offset[0] - 0.2 * offset[1], end[1] - 0.2 * offset[1] + 0.2 * offset[0]),
                (end[0] - 0.2 * offset[0] + 0.2 * offset[1], end[1] - 0.2 * offset[1] - 0.2 * offset[0])
            ])

    def draw_click(self, pos):
        self.click_aux_surf.fill((0,0,0,0))
        pyg.draw.circle(self.click_aux_surf, self.RED, pos, 10)
    def update(self):
        self.screen.blit(self.aux_surface, (0, 0))
        self.screen.blit(self.click_aux_surf, (0, 0))
        pyg.display.update()
