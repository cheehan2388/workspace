"""
astar.py — 全域規劃：A* 在「已知的地圖」上找一條從起點到終點的路

要點：
  - A* 只知道「地圖上有的東西」＝牆。突然出現、會移動的障礙物不在地圖上，它不知道。
  - 牆先往外「膨脹」INFLATION（車身半徑 0.11 + 安全餘裕 0.09），路徑就不會貼牆。
  - 全程只規劃一次（跟 Hybrid 論文一樣：關掉重新規劃），之後局部規劃器負責臨場反應。
"""

import heapq
import math
import numpy as np

RES = 0.05            # 格子大小 5 公分
INFLATION = 0.20      # 牆往外膨脹多少（公尺）
DYN_INFLATION = 0.17  # 重新規劃時，光達看到的障礙物往外膨脹多少（比牆小一點，才擠得過去）
WP_COUNT = 8          # 路點數量（Hybrid 論文：8 個）
WP_SPACING = 0.5      # 相鄰路點沿著路徑的間隔（公尺）。論文沒寫，這是我的選擇


class GridMap:
    def __init__(self, walls, res=RES, inflation=INFLATION):
        self.res = res
        xs = [w[0] - w[2] / 2 for w in walls] + [w[0] + w[2] / 2 for w in walls]
        ys = [w[1] - w[3] / 2 for w in walls] + [w[1] + w[3] / 2 for w in walls]
        self.x0, self.y0 = min(xs), min(ys)
        self.nx = int(math.ceil((max(xs) - self.x0) / res))
        self.ny = int(math.ceil((max(ys) - self.y0) / res))
        X = self.x0 + (np.arange(self.nx) + 0.5) * res
        Y = self.y0 + (np.arange(self.ny) + 0.5) * res
        X, Y = np.meshgrid(X, Y)                       # 形狀 (ny, nx)
        dmin = np.full(X.shape, np.inf)
        for cx, cy, sx, sy in walls:                   # 每個格子中心到每面牆（長方形）的距離
            dx = np.maximum(np.abs(X - cx) - sx / 2, 0)
            dy = np.maximum(np.abs(Y - cy) - sy / 2, 0)
            dmin = np.minimum(dmin, np.hypot(dx, dy))
        self.blocked = dmin < inflation

    def to_cell(self, x, y):
        return (int((x - self.x0) / self.res), int((y - self.y0) / self.res))   # (ix, iy)

    def to_xy(self, ix, iy):
        return (self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res)

    def free(self, ix, iy, blk=None):
        blk = self.blocked if blk is None else blk
        return 0 <= ix < self.nx and 0 <= iy < self.ny and not blk[iy, ix]

    def nearest_free(self, ix, iy, max_r=20, blk=None):
        """起點／終點若剛好落在膨脹區，就找最近的自由格子。"""
        if self.free(ix, iy, blk):
            return ix, iy
        for r in range(1, max_r + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if max(abs(dx), abs(dy)) == r and self.free(ix + dx, iy + dy, blk):
                        return ix + dx, iy + dy
        raise ValueError("附近找不到自由格子")

    def mark_hits(self, hits, inflation=DYN_INFLATION):
        """把光達打到的點（含突然出現、會動的障礙物）也標成障礙物，回傳多出來的障礙格。"""
        extra = np.zeros_like(self.blocked)
        if len(hits) == 0:
            return extra
        r = int(math.ceil(inflation / self.res))
        off = np.array([(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                        if math.hypot(dx, dy) * self.res <= inflation])
        cells = np.stack([((hits[:, 0] - self.x0) / self.res).astype(int),
                          ((hits[:, 1] - self.y0) / self.res).astype(int)], axis=1)
        allc = (cells[:, None, :] + off[None, :, :]).reshape(-1, 2)
        ok = (allc[:, 0] >= 0) & (allc[:, 0] < self.nx) & (allc[:, 1] >= 0) & (allc[:, 1] < self.ny)
        extra[allc[ok, 1], allc[ok, 0]] = True
        return extra

    def plan(self, start_xy, goal_xy, extra=None):
        """A*（8 方向）。extra：額外的障礙格（重新規劃時放入光達看到的東西）。回傳 Path，找不到回傳 None。"""
        blk = self.blocked if extra is None else (self.blocked | extra)
        s = self.nearest_free(*self.to_cell(*start_xy), blk=blk)
        g = self.nearest_free(*self.to_cell(*goal_xy), blk=blk)
        moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                 (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
        best = {s: 0.0}
        parent = {}
        heap = [(math.hypot(g[0] - s[0], g[1] - s[1]), 0.0, s)]
        closed = set()
        while heap:
            _, cost, cur = heapq.heappop(heap)
            if cur in closed:
                continue
            closed.add(cur)
            if cur == g:
                cells = [cur]
                while cur in parent:
                    cur = parent[cur]
                    cells.append(cur)
                pts = [self.to_xy(*c) for c in reversed(cells)]
                pts[0], pts[-1] = tuple(start_xy), tuple(goal_xy)
                return Path(pts)
            for dx, dy, step in moves:
                nx_, ny_ = cur[0] + dx, cur[1] + dy
                if not self.free(nx_, ny_, blk):
                    continue
                if dx != 0 and dy != 0 and not (self.free(cur[0] + dx, cur[1], blk) and self.free(cur[0], cur[1] + dy, blk)):
                    continue                              # 不准斜著穿過牆角
                nc = cost + step
                if nc < best.get((nx_, ny_), np.inf):
                    best[(nx_, ny_)] = nc
                    parent[(nx_, ny_)] = cur
                    heapq.heappush(heap, (nc + math.hypot(g[0] - nx_, g[1] - ny_), nc, (nx_, ny_)))
        return None


class Path:
    def __init__(self, pts):
        self.pts = np.array(pts, dtype=np.float64)
        seg = np.hypot(*np.diff(self.pts, axis=0).T)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])   # 沿路徑累積長度
        self.i = 0                                          # 車子目前大概在路徑的哪一點（只准往前）

    @property
    def length(self):
        return float(self.s[-1])

    def progress(self, x, y, window=30):
        """車子最靠近路徑的哪個點。只往前找一小段，避免跳到隔壁走廊的那一段路。"""
        hi = min(self.i + window, len(self.pts))
        d = np.hypot(self.pts[self.i:hi, 0] - x, self.pts[self.i:hi, 1] - y)
        self.i += int(d.argmin())
        return self.i

    def ahead(self, x, y, length=3.0):
        """車子前方 length 公尺的真實路徑（A* 算出來的那條線，不是車子到路點的直線）。"""
        i = self.progress(x, y)
        j = int(np.searchsorted(self.s, self.s[i] + length)) + 1
        return self.pts[i:min(j, len(self.pts))] if j - i >= 2 else self.pts[-2:]

    def waypoints(self, x, y, n=WP_COUNT, spacing=WP_SPACING):
        """從車子目前位置往前，沿路徑每隔 spacing 公尺取一點，共 n 個。超過終點就重複終點。"""
        i = self.progress(x, y)
        targets = self.s[i] + spacing * np.arange(1, n + 1)
        idx = np.clip(np.searchsorted(self.s, targets), 0, len(self.pts) - 1)
        return self.pts[idx]
