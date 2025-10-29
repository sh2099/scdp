#!/usr/bin/env python3
"""
Load a single molecule using the notebook helper and plot its geometry.

Saves a PNG (or other format via --outfile) showing atom positions colored by atomic number
and lines for edges (if `edge_index` is present on the molecule).

Usage example:
    python scripts/plot_molecule_geometry.py --idx 0 --outdir plots/molecules --vnode False

"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 3D plotting
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# Optional interactive plotting (plotly)
try:
    import plotly.graph_objects as go
    from plotly.colors import n_colors
    _plotly_available = True
except Exception:
    go = None
    _plotly_available = False
# Import the loader from the notebooks package
try:
    from notebooks.mol_loader import load_single_molecule
except Exception:
    # If module path isn't set up, try relative import fallback
    try:
        import sys
        REPO_ROOT = Path(__file__).resolve().parents[1]
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from notebooks.mol_loader import load_single_molecule
    except Exception as e:
        raise ImportError("Failed to import load_single_molecule from notebooks.mol_loader: " + str(e))


def _to_numpy(x):
    if x is None:
        return None
    try:
        if hasattr(x, 'detach'):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def make_color_map(atom_types: np.ndarray):
    """Return color for each atom based on atomic number.

    Uses a qualitative colormap and maps unique atom types to distinct colors.
    """
    if atom_types is None:
        return None, {}
    uniq = np.unique(atom_types)
    cmap = plt.get_cmap('tab20')
    color_map = {}
    for i, z in enumerate(uniq):
        color_map[int(z)] = cmap(i % 20)
    colors = [color_map.get(int(z), (0.4, 0.4, 0.4)) for z in atom_types]
    return colors, color_map


def plot_molecule(mol, outfile: Path, annotate: bool = True, show_edges: bool = True):
    coords = _to_numpy(getattr(mol, 'coords', None))
    atom_types = _to_numpy(getattr(mol, 'atom_types', None))
    edge_index = _to_numpy(getattr(mol, 'edge_index', None))

    if coords is None:
        raise RuntimeError('Molecule has no coords attribute or coords is None')
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        # If coords is flattened or of unexpected shape, try to reshape as (-1,3)
        if coords.size % 3 == 0:
            coords = coords.reshape(-1, 3)
        elif coords.size % 2 == 0:
            coords = coords.reshape(-1, 2)
        else:
            raise RuntimeError(f'Unsupported coords shape: {coords.shape}')

    n_nodes = coords.shape[0]
    atom_types_arr = None
    try:
        if atom_types is not None:
            atom_types_arr = np.asarray(atom_types, dtype=int).reshape(-1)[:n_nodes]
    except Exception:
        atom_types_arr = None

    colors, color_map = make_color_map(atom_types_arr) if atom_types_arr is not None else (['#444444'] * n_nodes, {})

    is_3d = coords.shape[1] == 3
    fig = plt.figure(figsize=(6, 6))
    if is_3d:
        ax = fig.add_subplot(111, projection='3d')
        xs, ys, zs = coords[:, 0], coords[:, 1], coords[:, 2]
        ax.scatter(xs, ys, zs, c=colors, s=80, edgecolor='k', linewidth=0.3)
        if show_edges and edge_index is not None:
            try:
                ei = edge_index.astype(int)
                if ei.shape[0] == 2:
                    pairs = ei.T
                elif ei.shape[1] == 2:
                    pairs = ei
                else:
                    pairs = ei.T
                for u, v in pairs:
                    if int(u) < 0 or int(v) < 0 or int(u) >= n_nodes or int(v) >= n_nodes:
                        continue
                    ax.plot([xs[int(u)], xs[int(v)]], [ys[int(u)], ys[int(v)]], [zs[int(u)], zs[int(v)]], color='k', alpha=0.3, linewidth=0.7)
            except Exception:
                pass
        if annotate and atom_types_arr is not None:
            for i, (x, y, z) in enumerate(coords.tolist()):
                label = f"{i}:{int(atom_types_arr[i])}" if atom_types_arr is not None else str(i)
                ax.text(x, y, z, label, fontsize=6)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_box_aspect((1, 1, 1))
    else:
        ax = fig.add_subplot(111)
        xs, ys = coords[:, 0], coords[:, 1]
        ax.scatter(xs, ys, c=colors, s=120, edgecolor='k', linewidth=0.3)
        if show_edges and edge_index is not None:
            try:
                ei = edge_index.astype(int)
                if ei.shape[0] == 2:
                    pairs = ei.T
                elif ei.shape[1] == 2:
                    pairs = ei
                else:
                    pairs = ei.T
                for u, v in pairs:
                    if int(u) < 0 or int(v) < 0 or int(u) >= n_nodes or int(v) >= n_nodes:
                        continue
                    ax.plot([xs[int(u)], xs[int(v)]], [ys[int(u)], ys[int(v)]], color='k', alpha=0.35, linewidth=0.8)
            except Exception:
                pass
        if annotate and atom_types_arr is not None:
            for i, (x, y) in enumerate(coords.tolist()):
                label = f"{i}:{int(atom_types_arr[i])}" if atom_types_arr is not None else str(i)
                ax.text(x, y, label, fontsize=8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal', adjustable='datalim')

    # Legend for atom types
    if color_map:
        # build proxy artists
        import matplotlib.lines as mlines
        handles = []
        for z, col in sorted(color_map.items()):
            handles.append(mlines.Line2D([], [], color=col, marker='o', linestyle='None', markersize=8, label=f'Z={z}'))
        ax.legend(handles=handles, loc='best', fontsize='small')

    plt.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(outfile), dpi=200)
    plt.close(fig)


def create_interactive_plot(mol, out_html: Path, annotate: bool = True, show_edges: bool = True):
    """Create an interactive Plotly HTML showing the molecule geometry.

    Writes a self-contained HTML (with CDN Plotly JS) to out_html.
    """
    if not _plotly_available:
        raise RuntimeError('Plotly is not available in this environment')

    coords = _to_numpy(getattr(mol, 'coords', None))
    atom_types = _to_numpy(getattr(mol, 'atom_types', None))
    edge_index = _to_numpy(getattr(mol, 'edge_index', None))

    if coords is None:
        raise RuntimeError('Molecule has no coords attribute or coords is None')
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        if coords.size % 3 == 0:
            coords = coords.reshape(-1, 3)
        elif coords.size % 2 == 0:
            coords = coords.reshape(-1, 2)
        else:
            raise RuntimeError(f'Unsupported coords shape: {coords.shape}')

    n_nodes = coords.shape[0]
    is_3d = coords.shape[1] == 3
    atom_types_arr = None
    try:
        if atom_types is not None:
            atom_types_arr = np.asarray(atom_types, dtype=int).reshape(-1)[:n_nodes]
    except Exception:
        atom_types_arr = None

    # Build color map (use matplotlib tab20 converted to hex)
    try:
        import matplotlib.colors as mcolors
        cmap = plt.get_cmap('tab20')
        uniq = np.unique(atom_types_arr) if atom_types_arr is not None else np.array([])
        color_map = {}
        for i, z in enumerate(uniq):
            rgba = cmap(i % 20)
            color_map[int(z)] = mcolors.to_hex(rgba)
    except Exception:
        color_map = {}

    fig = go.Figure()

    # Add edges as one lines trace (with None separators)
    if show_edges and edge_index is not None:
        try:
            ei = edge_index.astype(int)
            if ei.shape[0] == 2:
                pairs = ei.T
            elif ei.shape[1] == 2:
                pairs = ei
            else:
                pairs = ei.T
            xs_lines = []
            ys_lines = []
            zs_lines = [] if is_3d else None
            for u, v in pairs:
                u = int(u); v = int(v)
                if u < 0 or v < 0 or u >= n_nodes or v >= n_nodes:
                    continue
                xs_lines.extend([float(coords[u, 0]), float(coords[v, 0]), None])
                ys_lines.extend([float(coords[u, 1]), float(coords[v, 1]), None])
                if is_3d:
                    zs_lines.extend([float(coords[u, 2]), float(coords[v, 2]), None])

            if is_3d:
                fig.add_trace(go.Scatter3d(x=xs_lines, y=ys_lines, z=zs_lines, mode='lines', line=dict(color='gray', width=2), hoverinfo='none', name='edges'))
            else:
                fig.add_trace(go.Scatter(x=xs_lines, y=ys_lines, mode='lines', line=dict(color='gray', width=1), hoverinfo='none', name='edges'))
        except Exception:
            pass

    # Add atoms grouped by atom type for coloring
    if atom_types_arr is None or atom_types_arr.size == 0:
        # Single unlabeled trace
        if is_3d:
            fig.add_trace(go.Scatter3d(x=coords[:,0], y=coords[:,1], z=coords[:,2], mode='markers', marker=dict(size=6, color='#444444'), name='atoms', hovertext=[f"idx={i}" for i in range(n_nodes)], hoverinfo='text'))
        else:
            fig.add_trace(go.Scatter(x=coords[:,0], y=coords[:,1], mode='markers', marker=dict(size=8, color='#444444'), name='atoms', hovertext=[f"idx={i}" for i in range(n_nodes)], hoverinfo='text'))
    else:
        uniq_at = sorted(list(dict.fromkeys([int(x) for x in atom_types_arr.tolist()])))
        for z in uniq_at:
            mask = (atom_types_arr == z)
            xs = coords[mask, 0]
            ys = coords[mask, 1]
            zs = coords[mask, 2] if is_3d else None
            hover = []
            for i_idx in np.nonzero(mask)[0].tolist():
                hover.append(f"idx={i_idx}<br>Z={int(atom_types_arr[i_idx])}")
            color = color_map.get(int(z), '#444444')
            if is_3d:
                fig.add_trace(go.Scatter3d(x=xs.tolist(), y=ys.tolist(), z=zs.tolist(), mode='markers', marker=dict(size=6, color=color), name=f'Z={z}', hovertext=hover, hoverinfo='text'))
            else:
                fig.add_trace(go.Scatter(x=xs.tolist(), y=ys.tolist(), mode='markers', marker=dict(size=8, color=color), name=f'Z={z}', hovertext=hover, hoverinfo='text'))

    # Layout
    if is_3d:
        fig.update_layout(scene=dict(aspectmode='data'), template='plotly_white')
    else:
        fig.update_layout(template='plotly_white', xaxis=dict(constrain='domain'))

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs='cdn')



def main():
    parser = argparse.ArgumentParser(description='Plot molecule geometry (atoms + edges)')
    parser.add_argument('--idx', type=int, default=0, help='Index in training split to load (passed to loader)')
    parser.add_argument('--vnode', action='store_true', help='Use vnode dataset path variant in loader')
    parser.add_argument('--outdir', type=str, default='plots/molecules', help='Output directory for saved plot')
    parser.add_argument('--outfile', type=str, default='', help='Optional explicit output filename (overrides naming)')
    parser.add_argument('--no-annotate', dest='annotate', action='store_false', help='Disable atom-index/type annotations')
    parser.add_argument('--no-edges', dest='show_edges', action='store_false', help='Do not draw edges')
    parser.add_argument('--interactive', action='store_true', help='Write an interactive Plotly HTML (rotatable) instead of a static PNG')
    args = parser.parse_args()

    mol = load_single_molecule(idx=args.idx, vnode=bool(args.vnode))
    outdir = Path(args.outdir)
    if args.outfile:
        outpath = Path(args.outfile)
    else:
        outpath = outdir / f'molecule_idx{int(args.idx)}_geom.png'

    # If interactive requested, try to create Plotly HTML
    if bool(args.interactive):
        if not _plotly_available:
            print('[WARN] Plotly not available; falling back to static PNG output')
            try:
                plot_molecule(mol, outpath, annotate=bool(args.annotate), show_edges=bool(args.show_edges))
                print(f"Saved static plot to: {outpath}")
            except Exception as e:
                print(f"Failed plotting molecule: {e}")
                raise
        else:
            # prefer .html extension
            if args.outfile:
                out_html = Path(args.outfile)
            else:
                out_html = outdir / f'molecule_idx{int(args.idx)}_geom.html'
            print(f"Writing interactive HTML to: {out_html}")
            try:
                create_interactive_plot(mol, out_html, annotate=bool(args.annotate), show_edges=bool(args.show_edges))
                print(f"Saved interactive HTML to: {out_html}")
            except Exception as e:
                print(f"Failed creating interactive plot: {e}")
                # fallback to static PNG
                try:
                    plot_molecule(mol, outpath, annotate=bool(args.annotate), show_edges=bool(args.show_edges))
                    print(f"Saved static plot to: {outpath}")
                except Exception:
                    raise
    else:
        print(f"Saving molecule geometry plot to: {outpath}")
        try:
            plot_molecule(mol, outpath, annotate=bool(args.annotate), show_edges=bool(args.show_edges))
            print("Saved plot successfully.")
        except Exception as e:
            print(f"Failed plotting molecule: {e}")
            raise


if __name__ == '__main__':
    raise SystemExit(main())
