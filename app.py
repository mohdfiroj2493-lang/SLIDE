
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Callable

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

@dataclass
class Material:
    name: str
    c: float
    phi_deg: float
    gamma: float

@dataclass
class Layer:
    z_top: float
    z_bot: float
    material: Material

@dataclass
class Model:
    ground_xy: List[Tuple[float, float]]
    toe_x: float
    layers: List[Layer]
    water_xy: Optional[List[Tuple[float, float]]]
    ru: Optional[float]
    line_load_q: float
    line_load_x: float
    surcharge_q: float
    surcharge_x1: float
    surcharge_x2: float

@dataclass
class Circle:
    x: float
    y: float
    R: float

@dataclass
class PolylineSurface:
    pts: List[Tuple[float, float]]

@dataclass
class Slice:
    x_left: float
    x_right: float
    width: float
    z_top_left: float
    z_top_right: float
    z_base_left: float
    z_base_right: float
    base_alpha_left: float
    base_alpha_right: float
    W: float
    u_bar: float

def interp_y_on_polyline(poly: List[Tuple[float, float]], x: float) -> float:
    if x <= poly[0][0]:
        return poly[0][1]
    if x >= poly[-1][0]:
        return poly[-1][1]
    for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
        if (x1 <= x <= x2) or (x2 <= x <= x1):
            if x2 == x1:
                return y1
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    return poly[-1][1]

def slope_angle_of_polyline(poly: List[Tuple[float, float]], x: float) -> float:
    # Local inclination angle (radians) of polyline at abscissa x
    if x <= poly[0][0]:
        x1, y1 = poly[0]
        x2, y2 = poly[1]
    elif x >= poly[-1][0]:
        x1, y1 = poly[-2]
        x2, y2 = poly[-1]
    else:
        for (x1, y1), (x2, y2) in zip(poly[:-1], poly[1:]):
            if x1 <= x <= x2:
                break
    dx = x2 - x1
    dy = y2 - y1
    if abs(dx) < 1e-9:
        return math.pi/2.0
    return math.atan2(dy, dx)

def circle_y_at_x(circle: Circle, x: float, choose: str = "lower") -> Optional[float]:
    dx = x - circle.x
    val = circle.R**2 - dx**2
    if val < 0:
        return None
    y0 = circle.y
    dy = math.sqrt(val)
    return (y0 - dy) if choose == "lower" else (y0 + dy)

def circle_base_alpha(circle: Circle, x: float) -> float:
    y = circle_y_at_x(circle, x, "lower")
    if y is None:
        return 0.0
    dy_dx = -(x - circle.x) / (y - circle.y)
    return math.atan(dy_dx)

def build_slices_circle(model: Model, circle: Circle, n_slices: int = 40) -> List[Slice]:
    xs = [p[0] for p in model.ground_xy]
    x_min, x_max = min(xs), max(xs)
    scan_x = np.linspace(x_min, x_max, 400)
    mask = []
    for x in scan_x:
        y_top = interp_y_on_polyline(model.ground_xy, x)
        y_base = circle_y_at_x(circle, x, choose="lower")
        mask.append(y_base is not None and y_base < y_top)
    if not any(mask):
        return []
    idx = np.where(mask)[0]
    x1 = scan_x[idx[0]]
    x2 = scan_x[idx[-1]]
    xs_sl = np.linspace(x1, x2, n_slices + 1)

    slices: List[Slice] = []
    for i in range(n_slices):
        xl = float(xs_sl[i]); xr = float(xs_sl[i + 1])
        ztl = interp_y_on_polyline(model.ground_xy, xl)
        ztr = interp_y_on_polyline(model.ground_xy, xr)
        zbl = circle_y_at_x(circle, xl, choose="lower")
        zbr = circle_y_at_x(circle, xr, choose="lower")
        if zbl is None or zbr is None:
            continue
        width = xr - xl
        h_avg = 0.5 * ((ztl - zbl) + (ztr - zbr))
        xm = 0.5*(xl+xr)
        zc = 0.5 * ((ztl + ztr) + (zbl + zbr)) * 0.5
        mat = material_at_z(model.layers, zc)
        gamma = mat.gamma
        surcharge = 0.0
        if xl <= model.line_load_x:
            surcharge += model.line_load_q
        x1s, x2s = sorted((model.surcharge_x1, model.surcharge_x2))
        if max(0.0, min(xr, x2s) - max(xl, x1s)) > 0:
            surcharge += model.surcharge_q
        W = gamma * h_avg * width + surcharge * width
        if model.water_xy is not None:
            zw = interp_y_on_polyline(model.water_xy, xm)
            head = max(0.0, zw - ((zbl + zbr) * 0.5))
            u_bar = 9.81 * head
        elif model.ru is not None:
            u_bar = model.ru * gamma * h_avg
        else:
            u_bar = 0.0
        slices.append(Slice(xl, xr, width, ztl, ztr, zbl, zbr,
                            circle_base_alpha(circle, xl), circle_base_alpha(circle, xr), W, u_bar))
    return slices

def build_slices_polyline(model: Model, surf: "PolylineSurface", n_slices: int = 60) -> List[Slice]:
    x_min = max(min(p[0] for p in surf.pts), min(p[0] for p in model.ground_xy))
    x_max = min(max(p[0] for p in surf.pts), max(p[0] for p in model.ground_xy))
    xs_sl = np.linspace(x_min, x_max, n_slices + 1)
    slices: List[Slice] = []
    for i in range(n_slices):
        xl = float(xs_sl[i]); xr = float(xs_sl[i + 1])
        ztl = interp_y_on_polyline(model.ground_xy, xl)
        ztr = interp_y_on_polyline(model.ground_xy, xr)
        zbl = interp_y_on_polyline(surf.pts, xl)
        zbr = interp_y_on_polyline(surf.pts, xr)
        if not (zbl < ztl and zbr < ztr):
            continue
        width = xr - xl
        h_avg = 0.5 * ((ztl - zbl) + (ztr - zbr))
        xm = 0.5*(xl+xr)
        zc = 0.5 * ((ztl + ztr) + (zbl + zbr)) * 0.5
        mat = material_at_z(model.layers, zc)
        gamma = mat.gamma
        surcharge = 0.0
        if xl <= model.line_load_x:
            surcharge += model.line_load_q
        x1s, x2s = sorted((model.surcharge_x1, model.surcharge_x2))
        if max(0.0, min(xr, x2s) - max(xl, x1s)) > 0:
            surcharge += model.surcharge_q
        W = gamma * h_avg * width + surcharge * width
        if model.water_xy is not None:
            zw = interp_y_on_polyline(model.water_xy, xm)
            head = max(0.0, zw - ((zbl + zbr) * 0.5))
            u_bar = 9.81 * head
        elif model.ru is not None:
            u_bar = model.ru * gamma * h_avg
        else:
            u_bar = 0.0
        aL = slope_angle_of_polyline(surf.pts, xl)
        aR = slope_angle_of_polyline(surf.pts, xr)
        slices.append(Slice(xl, xr, width, ztl, ztr, zbl, zbr, aL, aR, W, u_bar))
    return slices

def phi_rad(phi_deg: float) -> float:
    return phi_deg * math.pi / 180.0

def material_at_z(layers: List[Layer], z: float) -> Material:
    for lay in layers:
        if lay.z_bot <= z <= lay.z_top:
            return lay.material
    return layers[0].material if layers else Material("Default", 5.0, 30.0, 19.0)

def base_len_from_slice(s: Slice) -> float:
    a = 0.5*(s.base_alpha_left + s.base_alpha_right)
    return s.width / max(1e-6, math.cos(a))

def bishop_simplified_with_slices(slices: List[Slice], layers: List[Layer]):
    if len(slices) < 3:
        return float('nan'), {"slices": slices}
    F = 1.0
    for _ in range(100):
        num = 0.0
        den = 0.0
        for s in slices:
            zc = 0.5 * ((s.z_top_left + s.z_top_right) + (s.z_base_left + s.z_base_right)) * 0.5
            mat = material_at_z(layers, zc)
            c = mat.c
            phi = phi_rad(mat.phi_deg)
            a = 0.5*(s.base_alpha_left + s.base_alpha_right)
            m = math.sin(a) * math.tan(phi) / F + math.cos(a)
            b = base_len_from_slice(s)
            N = (s.W - s.u_bar * b) / m
            num += c * b + (N * math.tan(phi))
            den += s.W * math.sin(a)
        F_new = num / max(1e-9, den)
        if abs(F_new - F) < 1e-5:
            F = F_new
            break
        F = F_new
    return F, {"slices": slices}

class GLEResult(dict):
    pass

def gle_with_slices(slices: List[Slice], layers: List[Layer], fshape: str = "half-sine", spencer: bool = False):
    if len(slices) < 3:
        return float('nan'), GLEResult(slices=slices)
    xcent = np.array([0.5*(s.x_left + s.x_right) for s in slices])
    xm01 = (xcent - xcent.min()) / max(1e-9, (xcent.max() - xcent.min()))

    def interslice_function(name: str) -> Callable[[float], float]:
        if name == "half-sine": return lambda x: math.sin(math.pi * x)
        if name == "parabolic": return lambda x: 4*x*(1-x)
        return lambda x: 1.0

    ffun = interslice_function("constant" if spencer else fshape)
    fvals = np.array([ffun(x) for x in xm01])

    a = np.array([0.5*(s.base_alpha_left + s.base_alpha_right) for s in slices])
    b = np.array([base_len_from_slice(s) for s in slices])
    W = np.array([s.W for s in slices])
    uL = np.array([s.u_bar for s in slices])

    c_arr = []
    tanphi_arr = []
    for s in slices:
        zc = 0.5 * ((s.z_top_left + s.z_top_right) + (s.z_base_left + s.z_base_right)) * 0.5
        mat = material_at_z(layers, zc)
        c_arr.append(mat.c)
        tanphi_arr.append(math.tan(phi_rad(mat.phi_deg)))
    c_arr = np.array(c_arr)
    tanphi_arr = np.array(tanphi_arr)

    def residuals_for_lambda(lmbda: float):
        F = 1.2
        for _ in range(80):
            m_i = (np.sin(a) * tanphi_arr) / F + np.cos(a)
            N_eff = (W - uL * b) / np.maximum(1e-6, m_i)
            drive = np.sum(W * np.sin(a))
            resist = np.sum(c_arr * b + N_eff * tanphi_arr)
            F_new = resist / max(1e-9, drive)
            if abs(F_new - F) < 1e-5:
                F = F_new
                break
            F = F_new
        Nstar = 0.5*(N_eff[:-1] + N_eff[1:])
        Tstar = lmbda * 0.5*(fvals[:-1] + fvals[1:]) * Nstar
        R = np.sum(Tstar) / max(1e-6, np.sum(W))
        return F, R

    lam0, lam1 = 0.0, 0.2
    F_best = None
    for _ in range(25):
        F0, R0 = residuals_for_lambda(lam0)
        F1, R1 = residuals_for_lambda(lam1)
        F_best = F1
        if abs(R1) < 1e-4:
            break
        denom = (R1 - R0)
        if abs(denom) < 1e-9:
            lam1 += 0.1
            continue
        lam2 = lam1 - R1 * (lam1 - lam0) / denom
        lam0, lam1 = lam1, lam2

    return F_best, GLEResult(slices=slices, lambda_=lam1, fshape=fshape, spencer=spencer)

def random_polyline_between(ground_xy: List[Tuple[float, float]], x_entry: float, x_exit: float, n_ctrl: int, depth_frac: float) -> PolylineSurface:
    x1, x2 = sorted((x_entry, x_exit))
    xs = np.linspace(x1, x2, n_ctrl + 2)
    pts = []
    for i, x in enumerate(xs):
        z_ground = interp_y_on_polyline(ground_xy, float(x))
        if i == 0 or i == len(xs)-1:
            z = z_ground
        else:
            z = z_ground - depth_frac * abs(z_ground) - random.uniform(0.2, 1.0)*depth_frac*max(1.0, abs(z_ground))
        pts.append((float(x), float(z)))
    pts = sorted(pts, key=lambda p: p[0])
    return PolylineSurface(pts)

def compute_F(model: Model, method: str, surface_type: str, surface, n_slices: int, fshape='half-sine'):
    if surface_type == 'Circular':
        slices = build_slices_circle(model, surface, n_slices)
    else:
        slices = build_slices_polyline(model, surface, n_slices)

    if method == "Bishop":
        return bishop_simplified_with_slices(slices, model.layers)
    if method == "Spencer":
        return gle_with_slices(slices, model.layers, fshape='constant', spencer=True)
    if method == "Morgenstern–Price (GLE)":
        return gle_with_slices(slices, model.layers, fshape=fshape, spencer=False)
    if method == "Fellenius":
        if len(slices) < 3:
            return float('nan'), {"slices": slices}
        num = den = 0.0
        for s in slices:
            a = 0.5*(s.base_alpha_left + s.base_alpha_right)
            b = base_len_from_slice(s)
            zc = 0.5 * ((s.z_top_left + s.z_top_right) + (s.z_base_left + s.z_base_right)) * 0.5
            mat = material_at_z(model.layers, zc)
            c = mat.c
            tanphi = math.tan(phi_rad(mat.phi_deg))
            num += c*b + (s.W - s.u_bar*b) * tanphi
            den += s.W * math.sin(a)
        return num / max(1e-9, den), {"slices": slices}
    return compute_F(model, "Fellenius", surface_type, surface, n_slices)

st.set_page_config(page_title="Slide2-style Slope Stability (v3)", layout="wide")
st.title("Slide2-style Slope Stability – Streamlit v3")
st.caption("Circular & non-circular (polyline) slip surfaces; Bishop, Spencer, GLE. Educational MVP.")

with st.sidebar:
    st.header("Geometry")
    L = st.number_input("Horizontal extent L (m)", 40.0)
    H = st.number_input("Slope height H (m)", 10.0)
    beta = st.slider("Slope angle β (deg)", 10.0, 60.0, 30.0)
    crest = st.number_input("Crest length (m)", 10.0)
    toe = st.number_input("Toe length (m)", 10.0)
    x0 = 0.0
    x1 = crest
    x2 = crest + H / math.tan(math.radians(beta))
    x3 = x2 + toe
    ground_xy = [(x0, H), (x1, H), (x2, 0.0), (x3, 0.0)]

    st.header("Soils (horizontal layers)")
    nlay = st.number_input("Number of layers", 1, 5, 1)
    layers: List[Layer] = []
    for i in range(int(nlay)):
        st.subheader(f"Layer {i+1}")
        zt = st.number_input(f"Layer {i+1} top z (m)", value=H - i*5.0, key=f"zt{i}")
        zb = st.number_input(f"Layer {i+1} bottom z (m)", value=H - (i+1)*5.0, key=f"zb{i}")
        c = st.number_input(f"c (kPa) L{i+1}", value=5.0, key=f"c{i}")
        phi = st.number_input(f"phi (deg) L{i+1}", value=30.0, key=f"phi{i}")
        gam = st.number_input(f"gamma (kN/m^3) L{i+1}", value=19.0, key=f"gam{i}")
        layers.append(Layer(zt, zb, Material(f"L{i+1}", c, phi, gam)))

    st.header("Groundwater")
    use_ru = st.checkbox("Use ru coefficient instead of water table", value=True)
    water_xy = None
    ru = None
    if use_ru:
        ru = st.number_input("ru coefficient", value=0.0)
    else:
        wz1 = st.number_input("Water x1 (m)", 0.0)
        wy1 = st.number_input("Water z1 (m)", H*0.5)
        wz2 = st.number_input("Water x2 (m)", x3)
        wy2 = st.number_input("Water z2 (m)", H*0.2)
        water_xy = [(wz1, wy1), (wz2, wy2)]

    st.header("Loads")
    line_q = st.number_input("Crest line load q (kN/m)", 0.0)
    line_x = st.number_input("Line load acts at x ≤", crest)
    sur_q = st.number_input("Surcharge q (kPa)", 0.0)
    sur_x1 = st.number_input("Surcharge x1 (m)", 0.0)
    sur_x2 = st.number_input("Surcharge x2 (m)", x3)

    st.header("Surface & Method")
    surface_type = st.selectbox("Slip surface type", ["Circular", "Polyline (non-circular)"])
    method = st.selectbox("Analysis method", ["Bishop", "Spencer", "Morgenstern–Price (GLE)", "Fellenius", "Janbu"], index=1)
    fshape = 'half-sine'
    if method == "Morgenstern–Price (GLE)":
        fshape = st.selectbox("Interslice function f(x)", ["half-sine", "parabolic", "constant"], index=0)

    n_slices = st.slider("Number of slices", 10, 160, 60)

    st.subheader("Search settings")
    if surface_type == 'Circular':
        search_type = st.selectbox("Search strategy", ["Grid", "Monte Carlo"], index=1)
        xmin = st.number_input("Cx min", -0.5 * L, value=-10.0)
        xmax = st.number_input("Cx max", 1.5 * L, value=x3 + 20.0)
        ymin = st.number_input("Cy min", -L, value=-30.0)
        ymax = st.number_input("Cy max", 2 * H, value=H)
        Rmin = st.number_input("R min", 5.0)
        Rmax = st.number_input("R max", 200.0)
        if search_type == "Grid":
            nx = st.slider("grid nx", 5, 60, 20)
            ny = st.slider("grid ny", 3, 40, 10)
            nR = st.slider("grid nR", 5, 60, 15)
            n_trials = None
        else:
            n_trials = st.slider("Monte Carlo trials", 100, 8000, 2000)
            nx = ny = nR = None
    else:
        x_entry = st.slider("Entry x on ground", float(0.0), float(x3), float(0.25*x3))
        x_exit  = st.slider("Exit x on ground",  float(0.0), float(x3), float(0.9*x3))
        n_ctrl = st.slider("# interior control points", 2, 8, 4)
        depth_frac = st.slider("Depth band (fraction of ground z)", 0.05, 1.0, 0.3)
        n_trials = st.slider("Monte Carlo trials", 100, 10000, 3000)
        xmin = xmax = ymin = ymax = Rmin = Rmax = nx = ny = nR = None

    compute = st.button("Compute critical surface")

model = Model(
    ground_xy=ground_xy,
    toe_x=crest + H / math.tan(math.radians(beta)),
    layers=sorted(layers, key=lambda L: -L.z_top),
    water_xy=water_xy,
    ru=ru,
    line_load_q=line_q,
    line_load_x=line_x,
    surcharge_q=sur_q,
    surcharge_x1=sur_x1,
    surcharge_x2=sur_x2,
)

col1, col2 = st.columns([2, 1])

with col1:
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [p[0] for p in model.ground_xy]
    ys = [p[1] for p in model.ground_xy]
    ax.plot(xs, ys, lw=2, label="Ground")
    if model.water_xy:
        ax.plot([p[0] for p in model.water_xy], [p[1] for p in model.water_xy], linestyle='--', label="Water table")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.grid(True, alpha=0.2)

    best_F = None
    best_surface = None

    if compute:
        if surface_type == 'Circular':
            if n_trials is None:
                best = (float('inf'), None)
                for xi in np.linspace(xmin, xmax, nx):
                    for yi in np.linspace(ymin, ymax, ny):
                        for Ri in np.linspace(Rmin, Rmax, nR):
                            c = Circle(xi, yi, Ri)
                            F = compute_F(model, method, 'Circular', c, n_slices, fshape)[0]
                            if not math.isnan(F) and F < best[0]:
                                best = (F, c)
                best_F, best_surface = best
            else:
                best = (float('inf'), None)
                for _ in range(n_trials):
                    c = Circle(random.uniform(xmin, xmax), random.uniform(ymin, ymax), random.uniform(Rmin, Rmax))
                    F = compute_F(model, method, 'Circular', c, n_slices, fshape)[0]
                    if not math.isnan(F) and F < best[0]:
                        best = (F, c)
                best_F, best_surface = best

            if best_surface is not None:
                c = best_surface
                th = np.linspace(0, 2*math.pi, 400)
                X = c.x + c.R * np.cos(th)
                Y = c.y + c.R * np.sin(th)
                ax.plot(X, Y, alpha=0.7, label=f"Critical (F={best_F:.3f})")
        else:
            best = (float('inf'), None)
            for _ in range(n_trials):
                surf = random_polyline_between(model.ground_xy, x_entry, x_exit, n_ctrl, depth_frac)
                F = compute_F(model, method, 'Polyline', surf, n_slices, fshape)[0]
                if not math.isnan(F) and F < best[0]:
                    best = (F, surf)
            best_F, best_surface = best

            if best_surface is not None:
                ax.plot([p[0] for p in best_surface.pts], [p[1] for p in best_surface.pts], '-.', label=f"Critical (F={best_F:.3f})")

    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("Results")
    if compute and best_surface is not None:
        F_val, data = compute_F(model, method, 'Circular' if surface_type=='Circular' else 'Polyline', best_surface, n_slices, fshape)
        st.metric("Critical Factor of Safety", f"{F_val:.3f}")
        if surface_type == 'Circular':
            st.write({"type": "circle", "Cx": best_surface.x, "Cy": best_surface.y, "R": best_surface.R})
        else:
            st.write({"type": "polyline", "n_pts": len(best_surface.pts)})
        slices = data.get("slices", [])
        if slices:
            if st.button("Export CSV report"):
                import csv
                path = "slope_report.csv"
                with open(path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(["method", method])
                    w.writerow(["surface", surface_type])
                    w.writerow(["FOS", f"{F_val:.6f}"])
                    if surface_type == 'Circular':
                        w.writerow(["Cx", best_surface.x, "Cy", best_surface.y, "R", best_surface.R])
                    else:
                        w.writerow(["polyline_pts"] + [val for pt in best_surface.pts for val in pt])
                    w.writerow([])
                    w.writerow(["slice", "x_left", "x_right", "W(kN/m)", "u_bar(kPa)"])
                    for i, s in enumerate(slices, 1):
                        w.writerow([i, s.x_left, s.x_right, s.W, s.u_bar])
                st.success("Saved slope_report.csv in the working directory.")
    else:
        st.info("Set parameters on the left and click 'Compute critical surface'.")
