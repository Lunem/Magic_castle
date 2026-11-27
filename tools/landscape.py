"""Procedural scenery added around the STL kit (model coordinates: mm, z up, the
castle front faces -x, its left side is +y):

- a grassy hill that carries the kit's rock massif on, further to the left, and
  steps down to the ground in terraces, with bare rock on the steep risers;
- topiary trees (thin trunks, foliage clipped into boxes), on the hill and on
  top of the rock massif.

Everything is generated from a fixed seed, so the model rebuilds identically.
Each piece can be switched off in ENABLED without losing its code.
"""
import numpy as np

# Switch pieces of scenery on/off (set to True and rebuild the model to bring one back)
ENABLED = {
    "hill": False,             # "Herbe colline" + "Rochers colline": the terraced hill on the left
    "trees_on_hill": False,    # topiaries standing on that hill (need "hill")
    "trees_on_rocks": True,    # topiaries on the grass on top of the kit's own rocks
}
TREES_ON_ROCKS = 5           # how many
TREES_ON_HILL = 7

GROUND = -138.0      # the castle's ground level
HILL_TOP = 82.0      # level with the grass on top of the kit's rocks (z 85-97), so the two run into each other
TERRACES = 3         # number of steps on the way down
HILL_REACH = 330.0   # mm from the edge of the rocks to the foot of the slope

# Footprint of the hill (model x, y), measured on an aerial photo, scaled with the
# two entrance towers. It leans on the left flank of the rock massif (y ~ 180-300),
# runs left to the walkway (y ~ 600) and, past the cave, forward to the moat, level
# with the bridge. In front of the rocks (cave, entrance) it stays clear.
FOOTPRINT = np.array([(140, 180), (140, 560), (-10, 612), (-160, 592), (-272, 500),
                      (-335, 400), (-290, 340), (-200, 316), (-60, 316), (-60, 180)], float)
# In front (x < -60) it starts only past the left side of "Gros rocher entree bas" (y < 303),
# which is hollow: the dragon's cave must stay clear.
ROCK_TOP = ((-200.0, 140.0), (180.0, 290.0))    # x and y range of the grassy top of the kit's rocks
FRONT_STEEPNESS = 1.8                            # the slope falls this much faster towards the front
# Height of the front rock wall's top (0 = ground, 1 = hill top) next to the castle (y ~ 300),
# a third of the way along (y ~ 430) and towards the far end (y ~ 560); the grass behind it
# rises by FRONT_RISE per mm. Measured on a photo against the left entrance tower.
FRONT_EDGE = ((20 - GROUND) / (HILL_TOP - GROUND), (-40 - GROUND) / (HILL_TOP - GROUND), (-70 - GROUND) / (HILL_TOP - GROUND))
FRONT_RISE = 0.0028
FRONT_LINE = FOOTPRINT[[7, 6, 5, 4, 3]]       # the footprint's front side, from the cave to the far left


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def inside_distance(x, y, poly=FOOTPRINT):
    """Signed distance to the footprint outline: > 0 inside, < 0 outside (mm)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    inside = np.zeros(np.broadcast(x, y).shape, bool)
    dist = np.full(inside.shape, np.inf)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, 0)):
        cross = ((y1 > y) != (y2 > y)) & (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        inside ^= cross
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0, 1)
        dist = np.minimum(dist, np.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return np.where(inside, dist, -dist)


def polyline_distance(x, y, line):
    x, y = np.asarray(x, float), np.asarray(y, float)
    dist = np.full(np.broadcast(x, y).shape, np.inf)
    for (x1, y1), (x2, y2) in zip(line[:-1], line[1:]):
        dx, dy = x2 - x1, y2 - y1
        t = np.clip(((x - x1) * dx + (y - y1) * dy) / (dx * dx + dy * dy), 0, 1)
        dist = np.minimum(dist, np.hypot(x - x1 - t * dx, y - y1 - t * dy))
    return dist


def hill_shape(x, y):
    """0 = ground, 1 = top of the hill (smooth, before terracing).

    Highest against the rocks (its grass carries straight on from theirs) and falling
    away from them: gently to the left, towards the walkway, and faster towards the
    front, down to the moat."""
    (x0, x1), (y0, y1) = ROCK_TOP
    dx = np.maximum(x0 - x, 0) * FRONT_STEEPNESS + np.maximum(x - x1, 0)
    dy = np.maximum(y - y1, 0)
    d = np.hypot(dx, dy)                                         # 0 over the rocks
    f = (1 - smoothstep(0, HILL_REACH, d)) ** 1.15
    f += 0.04 * np.sin(x * 0.019 + 1.3) * np.sin(y * 0.023 + 0.4) * smoothstep(0.05, 0.3, f)   # gentle irregularities
    # Front edge: a lower rock wall whose top runs from the height of the left entrance
    # tower's battlements (z ~ 20, next to the castle) down towards the left (z ~ -40),
    # with the grass climbing back up behind it, in steps, to the top of the rocks.
    edge_top = FRONT_EDGE[0] - (FRONT_EDGE[0] - FRONT_EDGE[1]) * smoothstep(300, 430, y) \
               - (FRONT_EDGE[1] - FRONT_EDGE[2]) * smoothstep(430, 560, y)
    front = edge_top + polyline_distance(x, y, FRONT_LINE) * FRONT_RISE
    return np.clip(np.minimum(f, front), 0, 1)


def terrace(f, x, y):
    """Soft steps: part smooth slope, part terraces (their risers come out as bare rock).
    The steps meander a little instead of following the slope exactly."""
    g = (f + 0.05 * np.sin(x * 0.013 - y * 0.009) + 0.03 * np.sin(y * 0.031)) * TERRACES
    steps = (np.floor(g) + smoothstep(0.78, 1.0, g - np.floor(g))) / TERRACES
    return np.clip(0.55 * f + 0.45 * steps, 0, 1)


def hill_height(x, y):
    """Terraced slope inside the footprint, ending in a rock wall all around its edge."""
    h = (HILL_TOP - GROUND) * terrace(hill_shape(x, y), x, y)
    # jagged, slightly sloping rock wall along the edge rather than a clean vertical cut
    edge = inside_distance(x, y) + 5 * np.sin(x * 0.15) * np.sin(y * 0.11) + 3 * np.sin(x * 0.37 + y * 0.29)
    return GROUND + h * smoothstep(0.0, 14.0, edge)


def grid_mesh(x0, x1, y0, y1, step, height):
    xs = np.arange(x0, x1 + step / 2, step)
    ys = np.arange(y0, y1 + step / 2, step)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    Z = height(X, Y)
    v = np.stack([X, Y, Z], -1).reshape(-1, 3)
    nx, ny = len(xs), len(ys)
    i = np.arange(nx - 1)[:, None] * ny + np.arange(ny - 1)[None, :]
    i = i.ravel()
    a, b, c, d = i, i + ny, i + ny + 1, i + 1                    # quad corners
    faces = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    return v, faces


def hill():
    if not ENABLED["hill"]:
        return []
    v, f = grid_mesh(-370, 160, 170, 630, 4.0, hill_height)
    t = v[f]
    keep = (t[:, :, 2] > GROUND + 0.5).any(1)                   # drop the flat ground around it
    f = f[keep]
    t = v[f]
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    nz = n[:, 2] / (np.linalg.norm(n, axis=1) + 1e-12)
    steep = nz < 0.62
    return [("ROCHERS/Colline/Herbe colline.stl", t[~steep]),
            ("ROCHERS/Colline/Rochers colline.stl", t[steep])]


# ---------- topiary trees ----------
def tube(points, r0, r1, sides=7):
    """A tapered tube through the points (Nx3), closed at the top."""
    tris = []
    rings = []
    for k, p in enumerate(points):
        d = points[min(k + 1, len(points) - 1)] - points[max(k - 1, 0)]
        d /= np.linalg.norm(d)
        a = np.cross(d, [1, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1, 0])
        a /= np.linalg.norm(a)
        b = np.cross(d, a)
        r = r0 + (r1 - r0) * k / (len(points) - 1)
        ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        rings.append(p + r * (np.cos(ang)[:, None] * a + np.sin(ang)[:, None] * b))
    for k in range(len(rings) - 1):
        A, B = rings[k], rings[k + 1]
        for s in range(sides):
            s2 = (s + 1) % sides
            tris += [[A[s], A[s2], B[s2]], [A[s], B[s2], B[s]]]
    top = points[-1]
    for s in range(sides):
        tris.append([rings[-1][s], rings[-1][(s + 1) % sides], top])
    return np.array(tris)


def box(center, size, yaw, bevel=0.18):
    """A box with chamfered vertical edges (clipped hedge look)."""
    w, d, h = size
    cw, cd = w / 2, d / 2
    e = min(w, d) * bevel
    plan = np.array([[cw - e, -cd], [cw, -cd + e], [cw, cd - e], [cw - e, cd],
                     [-cw + e, cd], [-cw, cd - e], [-cw, -cd + e], [-cw + e, -cd]])
    c, s = np.cos(yaw), np.sin(yaw)
    plan = plan @ np.array([[c, s], [-s, c]])
    lo = np.c_[plan, np.full(8, -h / 2)] + center
    hi = np.c_[plan, np.full(8, h / 2)] + center
    tris = []
    for k in range(8):
        k2 = (k + 1) % 8
        tris += [[lo[k], lo[k2], hi[k2]], [lo[k], hi[k2], hi[k]]]
    for k in range(1, 7):
        tris += [[hi[0], hi[k], hi[k + 1]], [lo[0], lo[k + 1], lo[k]]]
    return np.array(tris)


def topiary(base, rng):
    """One tree: a slender, slightly bent trunk (sometimes forked) under box-clipped foliage."""
    trunks, leaves = [], []
    n_heads = 2 if rng.random() < 0.3 else 1
    height = rng.uniform(62, 88)
    for k in range(n_heads):
        side = rng.uniform(0, 2 * np.pi)
        lean = rng.normal(0, 0.04, 2) + (0.16 * (k - 0.5) * np.array([np.cos(side), np.sin(side)]) if n_heads == 2 else 0)
        h = height * (rng.uniform(0.72, 0.88) if k == 1 else 1)
        top = base + np.array([lean[0] * h, lean[1] * h, h])
        mid = base + (top - base) * 0.5 + np.r_[rng.normal(0, 2.5, 2), 0]
        trunks.append(tube(np.array([base, mid, top + [0, 0, 4]]), 2.4, 1.5))
        size = (rng.uniform(38, 52), rng.uniform(32, 44), rng.uniform(22, 30))
        leaves.append(box(top + [0, 0, size[2] / 2 - 2], size, rng.uniform(0, np.pi)))
    return np.concatenate(trunks), np.concatenate(leaves)


def ground_at(tris, x, y):
    """Highest point of these triangles straight above/below (x, y), or None."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    d1 = (x - b[:, 0]) * (a[:, 1] - b[:, 1]) - (a[:, 0] - b[:, 0]) * (y - b[:, 1])
    d2 = (x - c[:, 0]) * (b[:, 1] - c[:, 1]) - (b[:, 0] - c[:, 0]) * (y - c[:, 1])
    d3 = (x - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (y - a[:, 1])
    inside = ~(((d1 < 0) | (d2 < 0) | (d3 < 0)) & ((d1 > 0) | (d2 > 0) | (d3 > 0)))
    n = np.cross(b - a, c - a)
    ok = inside & (np.abs(n[:, 2]) > 1e-9) & (n[:, 2] > 0)
    if not ok.any():
        return None
    k = np.where(ok)[0]
    z = a[k, 2] - (n[k, 0] * (x - a[k, 0]) + n[k, 1] * (y - a[k, 1])) / n[k, 2]
    return z.max()


def place(rng, candidates, surface, count, spacing, taken, accept=lambda x, y, z: True):
    spots = []
    for x, y in candidates[rng.permutation(len(candidates))]:
        if len(spots) >= count:
            break
        if any(np.hypot(x - u, y - v) < spacing for u, v in taken + [(s[0], s[1]) for s in spots]):
            continue
        z = surface(x, y)
        if z is None or not accept(x, y, z):
            continue
        spots.append(np.array([x, y, z - 1.0]))   # sink the foot a little
    return spots


def trees(rock_grass_tris):
    """Topiaries on the hill and on top of the rock massif."""
    rng = np.random.default_rng(1992)
    taken = []
    # On top of the rocks: along the grassy ledges near the castle
    cand = np.stack(np.meshgrid(np.arange(-200, 160, 6.0), np.arange(130, 300, 6.0), indexing="ij"), -1).reshape(-1, 2)
    on_rock = place(rng, cand, lambda x, y: ground_at(rock_grass_tris, x, y), TREES_ON_ROCKS, 70, taken,
                    accept=lambda x, y, z: z > 20)
    taken += [(p[0], p[1]) for p in on_rock]
    # On the hill: on the flat parts of the terraces
    def flat(x, y):
        f = hill_shape(np.array(x), np.array(y))
        g = f * TERRACES - np.floor(f * TERRACES)
        return 0.3 < f < 0.97 and g < 0.65 and y > 300 and inside_distance(x, y) > 25   # upper part of the hill
    cand = np.stack(np.meshgrid(np.arange(-360, 150, 7.0), np.arange(300, 620, 7.0), indexing="ij"), -1).reshape(-1, 2)
    on_hill = place(rng, cand, lambda x, y: float(hill_height(np.array(x), np.array(y))), TREES_ON_HILL, 70, taken,
                    accept=lambda x, y, z: flat(x, y))
    out = []
    groups = []
    if ENABLED["trees_on_rocks"]:
        groups.append(("ROCHERS/Rochers/Arbres", on_rock))
    if ENABLED["trees_on_hill"] and ENABLED["hill"]:
        groups.append(("ROCHERS/Colline/Arbres", on_hill))
    for name, spots in groups:
        if not spots:
            continue
        tr, lv = zip(*(topiary(p, rng) for p in spots))
        out += [(f"{name}/Troncs.stl", np.concatenate(tr)), (f"{name}/Feuillages.stl", np.concatenate(lv))]
    return out
