# Castle by Night

An interactive 3D castle (Three.js) with an **Explode** button that lifts the castle apart layer by layer, and three looks: **Night**, **Day**, and **Stars** (the castle drawn as glowing stars, floating in the void).

- `index.html`: the whole page (Three.js is loaded from a CDN)
- `models/castle.bin.gz`: all 637 castle parts, simplified and compressed (~7 MB)
- `tools/build_model.py`: rebuilds that file from the STL kit in `Assets/` (not committed)

## View locally

The page loads the model with `fetch`, so it won't work when opened by double-clicking the file. Run a small local server instead:

```sh
python3 -m http.server 8000
# then open http://localhost:8000
```

## Rebuild the model after changing STLs

```sh
pip3 install --user numpy fast-simplification
python3 tools/build_model.py --budget 700000   # bigger budget = more detail, bigger file
```

Parts keep their folder path, which drives the explode: the top folder (`ROCHERS`, `BASE`, `BATIMENTS`, `DONJON`) is the layer, and the second folder is the sub-assembly that slides outward. You can tune the distances in `LAYERS` in `index.html`.

## Publish on GitHub Pages

Push to `main`, then go to **Settings → Pages → Build and deployment → Deploy from a branch → `main` / root**.
