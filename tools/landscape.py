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
    "basin": False,            # the water in front of the castle, on both sides of the bridge (outline not right yet)
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


# ---------- the basin in front of the castle ----------
# One stretch of water: the big pool on the right of the entrance, the winding channel on
# the left, and the part that runs under the bridge to the dark pool at the foot of the cave.
BASIN = {   # outlines in model (x, y), traced on the drone photo (1.19 mm per pixel)
    "Bassin droite": [(120.9, -399.8), (116.2, -383.2), (109, -376), (90, -371.3), (54.3, -347.5), (40, -354.6), (25.7, -354.6), (4.3, -366.5), (-17.1, -366.5), (-74.2, -352.2), (-83.8, -316.5), (-83.8, -264.2), (-107.6, -254.7), (-109.9, -247.5), (-121.8, -235.6), (-131.4, -233.2), (-150.4, -211.8), (-152.8, -202.3), (-157.5, -199.9), (-159.9, -183.3), (-164.7, -180.9), (-167.1, -161.8), (-179, -149.9), (-202.8, -149.9), (-233.7, -138), (-238.5, -145.2), (-250.4, -145.2), (-259.9, -159.5), (-345.6, -169), (-362.2, -183.3), (-378.9, -180.9), (-390.8, -169), (-400.3, -166.6), (-409.8, -183.3), (-426.5, -183.3), (-433.6, -190.4), (-431.2, -230.9), (-407.4, -257), (-386, -285.6), (-371.7, -288), (-343.2, -316.5), (-324.1, -323.7), (-321.8, -328.4), (-305.1, -330.8), (-298, -338), (-271.8, -342.7), (-259.9, -361.8), (-240.8, -376), (-231.3, -373.7), (-209.9, -390.3), (-138.5, -404.6), (-124.2, -416.5), (-90.9, -423.6), (-83.8, -435.5), (-71.9, -445.1), (-64.7, -445.1), (-57.6, -459.3), (-40.9, -468.9), (-36.2, -480.8), (-19.5, -495), (-17.1, -506.9), (-10, -511.7), (-7.6, -523.6), (4.3, -533.1), (11.4, -554.5), (18.6, -559.3), (21, -573.6), (30.5, -583.1), (37.6, -583.1), (47.1, -573.6), (51.9, -559.3), (63.8, -549.8), (68.6, -537.9), (66.2, -516.5), (78.1, -464.1), (87.6, -454.6), (87.6, -447.4), (101.9, -435.5), (109, -414.1)],
    "Bassin gauche": [(-140.9, 518.8), (-148, 542.6), (-159.9, 554.5), (-174.2, 564.1), (-179, 561.7), (-198, 568.8), (-205.1, 561.7), (-217, 561.7), (-226.6, 549.8), (-243.2, 549.8), (-252.7, 542.6), (-267, 542.6), (-274.2, 547.4), (-319.4, 542.6), (-345.6, 549.8), (-390.8, 552.2), (-409.8, 549.8), (-421.7, 540.3), (-433.6, 537.9), (-457.4, 516.5), (-459.8, 509.3), (-471.7, 502.2), (-476.5, 490.3), (-486, 487.9), (-495.5, 473.6), (-507.4, 468.9), (-521.7, 454.6), (-519.3, 416.5), (-524.1, 409.4), (-524.1, 390.3), (-533.6, 385.6), (-555, 390.3), (-562.1, 387.9), (-571.7, 368.9), (-581.2, 364.1), (-595.5, 345.1), (-605, 342.7), (-633.5, 321.3), (-650.2, 302.3), (-650.2, 295.1), (-657.3, 290.4), (-657.3, 273.7), (-635.9, 252.3), (-602.6, 240.4), (-559.8, 249.9), (-536, 280.8), (-536, 288), (-528.8, 290.4), (-509.8, 311.8), (-505, 311.8), (-490.7, 330.8), (-478.8, 330.8), (-455, 349.9), (-440.8, 354.6), (-393.2, 361.8), (-388.4, 366.5), (-386, 383.2), (-378.9, 387.9), (-378.9, 414.1), (-362.2, 423.6), (-364.6, 466.5), (-369.4, 468.9), (-374.1, 483.1), (-371.7, 495), (-359.8, 504.6), (-343.2, 504.6), (-333.7, 521.2), (-314.6, 521.2), (-302.7, 530.7), (-293.2, 528.4), (-286.1, 514.1), (-293.2, 499.8), (-267, 483.1), (-255.1, 483.1), (-248, 478.4), (-198, 478.4), (-190.9, 483.1), (-183.7, 480.8), (-159.9, 485.5), (-145.6, 502.2)],
    "Bassin sous le pont": [(-215.8, -164.2), (-352.7, -154.7), (-355.1, -71.4), (-334.8, 11.9), (-323, 130.9), (-299.2, 214.2), (-207.5, 223.7), (-219.4, 130.9), (-227.8, 0), (-215.8, -107.1)],
}
WATER_LEVEL = GROUND + 0.6   # just above the ground plane


def triangulate(poly):
    """Ear clipping for a simple polygon (Nx2), returns index triples."""
    P = np.asarray(poly, float)
    if np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]) < 0:
        P = P[::-1]; flip = True
    else:
        flip = False
    idx = list(range(len(P)))
    tris = []
    def inside(p, a, b, c):
        d1 = (p[0] - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (p[1] - b[1])
        d2 = (p[0] - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (p[1] - c[1])
        d3 = (p[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (p[1] - a[1])
        return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = P[i0], P[i1], P[i2]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0:
                continue                                   # reflex corner
            if any(inside(P[j], a, b, c) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break
    if len(idx) == 3:
        tris.append(tuple(idx))
    n = len(P)
    if flip:   # back to the caller's numbering
        tris = [tuple(n - 1 - i for i in t) for t in tris]
    return tris


def basin():
    if not ENABLED["basin"]:
        return []
    out = []
    for name, poly in BASIN.items():
        P = np.asarray(poly, float)
        V = np.c_[P, np.full(len(P), WATER_LEVEL)]
        tris = np.array([[V[a], V[b], V[c]] for a, b, c in triangulate(P)])
        n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        tris[n[:, 2] < 0] = tris[n[:, 2] < 0][:, ::-1]   # facing up
        out.append((f"BASE/Bassin/Eau {name.split(' ', 1)[1]}.stl", tris))
    return out


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
