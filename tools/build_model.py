"""Pack every STL under Assets/ into one compact, web-ready file: models/castle.bin.gz

Usage:  python3 tools/build_model.py [--budget 1200000]

Each STL keeps its own identity (name, folder path, material) so the page can
move parts independently for the Explode animation. Meshes are welded,
simplified, positions quantized to uint16 against the castle bounding box,
indices stored as uint16 per part, then everything is gzipped.

File layout (before gzip):
  uint32 jsonLength | JSON (utf-8, padded to 4 bytes) | binary blob
"""
import argparse, gzip, json, os, re, struct, sys
import numpy as np
import fast_simplification

import landscape   # procedural hill and topiary trees (tools/landscape.py)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "Assets")
OUT = os.path.join(ROOT, "models", "castle.bin.gz")

# Parts left out of the web model (matched on the lowercase path relative to Assets/)
EXCLUDE = [
    r"deco oeil de boeuf",   # print plate of 3 ornaments: placed by ORNAMENTS below instead
    r"support impression",   # 3D-printing supports, not part of the castle
    r"faitage tour g",       # misplaced in the STL; no roof in the kit has a matching ridge
]

# Parts saved at the wrong spot in the STL: (R, t, folder) moves them (new = v @ R + t)
# and files them under the folder of the assembly they belong to
PLACEMENT = {
    # Identical copy of Rotonde/Batiment/Toiture/Faitage, left floating near Batiment J.
    # It covers the empty half of the rotonde roof ridge (x -51..-0.5): the existing
    # crest turned 180° about the vertical axis.
    "batiments/accessoires/faitage rotonde.stl": (
        np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        np.array([-124.2845, -112.3851, -14.1804]),
        "BATIMENTS/Rotonde/Batiment/Toiture",   # explode together with the rotonde roof
    ),
}

# "Deco oeil de boeuf" is a print plate: three identical gilded ornaments (a fleur-de-lys
# between two scrolls) held in a support frame. Each copy sits above one oval stained-glass
# window, on the roof just over the window frame, leaning back with the roof.
ORNAMENT_SRC = "BATIMENTS/Accessoires/Deco oeil de boeuf.stl"
ORNAMENT_FRAME_X = -110.85          # the support frame lies behind this plane (the ornaments' relief is in front)
ORNAMENT_ANCHOR = (-110.44, 119.7, 331.95)   # copy 0: back face, centre, bottom of the central bead
ORNAMENT_STEP_Z = 20.0              # the copies are stacked 20 mm apart on the plate
ORNAMENTS = [  # folder, name, anchor in the castle (x, y, z), lean back (deg), scale, faces +x
    # Tour G: stands on the arched hood over the window (its scrolls follow the arch), leaning towards the roof
    ("BATIMENTS/Tour G/Corps batiment", "Ornement vitrail", (-171.0, 0.0, 131.2), 32.0, 1.0, False),
    # Keep: smaller windows, so smaller ornaments, lying on the spire's slope just above each window frame
    ("DONJON/Corps partie superieure", "Ornement vitrail gauche", (-63.0, -18.0, 477.7), 17.7, 0.7, False),
    ("DONJON/Corps partie superieure", "Ornement vitrail droit", (-17.7, -18.0, 477.7), 17.7, 0.7, True),
]

MATERIAL_RULES = [  # first match wins, tested on the lowercase file name (folder path only where noted)
    ("lantern", r"lanterne"),
    ("flag",    r"drapeau"),
    ("iron",    r"rambarde fer|grille|herse|chaine"),
    ("timber",  r"colombage|porte|pont levis|charpente|volet|tronc"),
    ("grass",   r"herbe|base du jardinet|feuillage"),
    ("rock",    r"^rochers/"),                   # matched on the folder path
    ("roof",    r"toit|faitage|flèche|fleche|chapeau|coupole"),
    ("stone",   r""),
]


def load_stl(path):
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        v = np.array([l.split()[1:4] for l in data.decode(errors="ignore").splitlines()
                      if l.strip().startswith("vertex")], dtype=np.float32)
    else:
        n = struct.unpack("<I", data[80:84])[0]
        rec = np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")])
        v = np.frombuffer(data[84:84 + n * 50], dtype=rec)["v"].reshape(-1, 3)
    return v.astype(np.float64)


def weld(tri_verts, eps=1e-3):
    key = np.round(tri_verts / eps).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    verts = np.zeros((len(uniq), 3))
    np.add.at(verts, inv.ravel(), tri_verts)
    verts /= np.bincount(inv.ravel())[:, None]
    faces = inv.reshape(-1, 3)
    good = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    return verts, faces[good]


def ornament_copies():
    """Split the ornament plate into its three copies, without the support frame."""
    raw = load_stl(os.path.join(ASSETS, ORNAMENT_SRC)).reshape(-1, 3, 3)
    tris = raw[~(raw[:, :, 0] >= ORNAMENT_FRAME_X).all(1)]
    # connected pieces (scrolls, fleur-de-lys, bead), each assigned to a copy by its lowest point
    v, f = weld(tris.reshape(-1, 3))
    parent = np.arange(len(v))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, c in f:
        ra, rb, rc = root(a), root(b), root(c)
        parent[rb] = ra
        parent[root(rc)] = ra
    piece = np.array([root(i) for i in range(len(v))])
    low = {r: v[piece == r, 2].min() for r in np.unique(piece[f[:, 0]])}
    copy_of = np.array([int((low[piece[a]] - (ORNAMENT_ANCHOR[2] - 9.0)) // ORNAMENT_STEP_Z) for a in f[:, 0]])
    return [v[f[copy_of == k]] for k in range(3)]


def placed_ornaments():
    out, copies = [], ornament_copies()
    for k, (folder, name, at, lean, scale, flip) in enumerate(ORNAMENTS):
        tri = copies[k].reshape(-1, 3)
        anchor = np.array(ORNAMENT_ANCHOR) + [0, 0, ORNAMENT_STEP_Z * k]
        a = np.radians(lean)
        ry = np.array([[np.cos(a), 0, -np.sin(a)], [0, 1, 0], [np.sin(a), 0, np.cos(a)]])   # row vectors: top leans to +x
        v = np.einsum("ij,jk->ik", (tri - anchor) * scale, ry)
        if flip:
            v[:, :2] *= -1                                                                 # turn 180° to face +x
        out.append((os.path.join(folder, name + ".stl"), v + np.array(at)))
    return out


def material_for(rel):
    low = rel.lower().replace(os.sep, "/")
    name = os.path.basename(low)
    for mat, rx in MATERIAL_RULES:
        if re.search(rx, low if rx.startswith("^rochers") else name):
            return mat
    return "stone"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=1_200_000, help="target total triangles")
    ap.add_argument("--floor", type=int, default=600, help="parts below this are kept as is")
    args = ap.parse_args()

    files = sorted(os.path.relpath(os.path.join(d, f), ASSETS)
                   for d, _, fs in os.walk(ASSETS) for f in fs if f.lower().endswith(".stl"))
    files = [f for f in files if not any(re.search(rx, f.lower()) for rx in EXCLUDE)]
    meshes = []
    for i, rel in enumerate(files):
        raw = load_stl(os.path.join(ASSETS, rel)).reshape(-1, 3)
        fix = PLACEMENT.get(rel.lower().replace(os.sep, "/"))
        if fix:
            raw = np.einsum("ij,jk->ik", raw, fix[0]) + fix[1]
            rel = os.path.join(fix[2], os.path.basename(rel))
        v, f = weld(raw)
        meshes.append([rel, v, f])
        print(f"\rload {i+1}/{len(files)}", end="", file=sys.stderr)
    print(file=sys.stderr)
    for rel, raw in placed_ornaments():
        v, f = weld(raw)
        meshes.append([rel, v, f])
    # Scenery: the hill on the left, and topiary trees on it and on top of the rocks
    rock_grass = np.concatenate([v[f] for rel, v, f in meshes
                                 if rel.replace(os.sep, "/").startswith("ROCHERS/") and "herbe" in rel.lower()])
    for rel, tris in landscape.hill() + landscape.trees(rock_grass):
        v, f = weld(tris.reshape(-1, 3))
        meshes.append([rel, v, f])

    # Keep small parts intact; scale the rest by one shared ratio to meet the budget.
    sizes = np.array([len(m[2]) for m in meshes])
    small = sizes <= args.floor
    ratio = min(1.0, max(0.02, (args.budget - sizes[small].sum()) / max(1, sizes[~small].sum())))
    print(f"source triangles {sizes.sum():,}  keep ratio for large parts {ratio:.3f}", file=sys.stderr)

    for i, m in enumerate(meshes):
        n = len(m[2])
        target = n if n <= args.floor else max(args.floor, int(n * ratio))
        if target < n:
            v, f = fast_simplification.simplify(m[1].astype(np.float32), m[2].astype(np.int32),
                                                target_reduction=1 - target / n, agg=5)
            m[1], m[2] = v.astype(np.float64), f
        while len(m[1]) > 65535:  # uint16 indices
            v, f = fast_simplification.simplify(m[1].astype(np.float32), m[2].astype(np.int32),
                                                target_reduction=0.3)
            m[1], m[2] = v.astype(np.float64), f
        print(f"\rsimplify {i+1}/{len(meshes)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    allv = np.concatenate([m[1] for m in meshes])
    lo, hi = allv.min(0), allv.max(0)
    scale = (hi - lo) / 65535.0

    parts, pos_chunks, idx_chunks = [], [], []
    v_off = i_off = 0
    for rel, v, f in meshes:
        q = np.clip(np.round((v - lo) / scale), 0, 65535).astype(np.uint16)
        pos_chunks.append(q.ravel())
        idx_chunks.append(f.astype(np.uint16).ravel())
        folder, name = os.path.split(rel)
        parts.append({
            "name": os.path.splitext(name)[0],
            "path": folder.replace(os.sep, "/"),
            "mat": material_for(rel),
            "v": [v_off, len(v)],
            "i": [i_off, f.size],
            "center": [round(x, 2) for x in ((v.min(0) + v.max(0)) / 2)],
        })
        v_off += len(v)
        i_off += f.size

    pos = np.concatenate(pos_chunks)
    idx = np.concatenate(idx_chunks)
    header = {"version": 1, "units": "mm", "up": "z",
              "min": lo.round(4).tolist(), "scale": scale.tolist(),
              "positionsBytes": pos.nbytes, "indicesBytes": idx.nbytes, "parts": parts}
    js = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob = struct.pack("<I", len(js)) + js + pos.tobytes() + idx.tobytes()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with gzip.open(OUT, "wb", compresslevel=9) as fh:
        fh.write(blob)
    print(f"{len(parts)} parts, {idx.size // 3:,} triangles, {v_off:,} vertices -> "
          f"{os.path.getsize(OUT) / 1e6:.1f} MB ({len(blob) / 1e6:.1f} MB raw)", file=sys.stderr)
    counts = {}
    for p in parts:
        counts[p["mat"]] = counts.get(p["mat"], 0) + 1
    print("materials:", counts, file=sys.stderr)


if __name__ == "__main__":
    main()
