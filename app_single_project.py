# app_single_project.py
# Single-project cutting optimization UI (no Excel needed)
import math
import pandas as pd
import streamlit as st
import pulp

st.set_page_config(page_title="Door Frame Cutting Optimizer", page_icon="🪚", layout="centered")

# ---------------------------- Helpers ----------------------------
def round_piece_len(x: float) -> float:
    # If you want whole-inch pieces, change to: return math.ceil(x)
    return x

def optimize_single_project(frame_w, frame_l, qty, stock_lengths, oversize=1.20, kerf=0.0,
                            time_limit=60, frac_gap=0.001):
    """
    Minimize waste using a bin-packing MILP:
      - Each door needs 1 width piece + 2 length pieces (oversized by factor).
      - Each required piece must come from a single stock stick (no gluing).
      - Capacity per bar: sum(piece_lengths) + kerf * (cuts_on_bar - 1) <= stock_length.
    """
    width_piece  = round_piece_len(frame_w * oversize)
    length_piece = round_piece_len(frame_l * oversize)

    pieces = [width_piece]*int(qty) + [length_piece]*int(2*qty)
    n = len(pieces)

    # Feasibility checks
    if n == 0:
        return {"status":"No pieces","counts":{L:0 for L in stock_lengths},"cut_plan":{L:[] for L in stock_lengths},
                "total_required_length":0.0,"total_used_length":0.0,"total_waste":0.0,"utilization":1.0}

    if not stock_lengths:
        return {"status":"No inventory selected","counts":{},"cut_plan":{},
                "total_required_length":sum(pieces),"total_used_length":0.0,"total_waste":float('inf'),"utilization":0.0}

    if max(pieces) > max(stock_lengths):
        return {"status":"Infeasible (a piece is longer than max stock length)",
                "counts":{L:0 for L in stock_lengths},"cut_plan":{L:[] for L in stock_lengths},
                "total_required_length":sum(pieces),"total_used_length":0.0,"total_waste":float('inf'),"utilization":0.0}

    # Speed-ups
    pieces.sort(reverse=True)  # helps solver branching

    total_required_len = sum(pieces)
    UB_per_len = {Ls: min(n, math.ceil(total_required_len / Ls) + 2) for Ls in stock_lengths}  # tight cap per length

    S = list(range(len(stock_lengths)))
    I = list(range(n))
    def bars_for_s(s):  # allowed bar indices for stock length s
        return range(UB_per_len[stock_lengths[s]])

    # MILP
    prob = pulp.LpProblem("SingleProjectCutting", pulp.LpMinimize)

    y, z = {}, {}
    for s in S:
        for b in bars_for_s(s):
            y[(s, b)] = pulp.LpVariable(f"y_s{s}_b{b}", cat="Binary")
            for i in I:
                z[(i, s, b)] = pulp.LpVariable(f"z_i{i}_s{s}_b{b}", cat="Binary")

    # Symmetry-breaking: use earlier bars first
    for s in S:
        bs = list(bars_for_s(s))
        for k in range(len(bs) - 1):
            prob += y[(s, bs[k])] >= y[(s, bs[k + 1])]

    # Capacity constraints with kerf
    for s in S:
        Ls = stock_lengths[s]
        for b in bars_for_s(s):
            sum_len = pulp.lpSum(pieces[i] * z[(i, s, b)] for i in I)
            sum_items = pulp.lpSum(z[(i, s, b)] for i in I)
            prob += sum_len + kerf * (sum_items - y[(s, b)]) <= Ls * y[(s, b)]

    # Each piece assigned exactly once
    for i in I:
        prob += pulp.lpSum(z[(i, s, b)] for s in S for b in bars_for_s(s)) == 1

    # Objective: minimize total waste
    total_stock_used  = pulp.lpSum(stock_lengths[s] * y[(s, b)] for s in S for b in bars_for_s(s))
    total_pieces_used = pulp.lpSum(pieces[i] * z[(i, s, b)] for i in I for s in S for b in bars_for_s(s))
    prob += total_stock_used - total_pieces_used






    # Solve with HiGHS via Python API (highspy). Falls back to CBC if available.
    solver_ok = False
    err_msgs = []

    # 1) Try HiGHS (Python API) — does not need a system binary
    try:
        from pulp import HiGHS
        prob.solve(HiGHS(msg=False, timeLimit=int(time_limit)))
        solver_ok = True
    except Exception as e:
        err_msgs.append(f"HiGHS(py) failed: {e}")

    # 2) Fallback to CBC if the host provides it
    if not solver_ok:
        try:
            prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=int(time_limit), fracGap=frac_gap))
            solver_ok = True
        except Exception as e:
            err_msgs.append(f"CBC failed: {e}")

    # 3) Final fallback to command-line HiGHS only if available (usually not on Streamlit Cloud)
    if not solver_ok:
        try:
            from pulp import HiGHS_CMD
            prob.solve(HiGHS_CMD(msg=False, timeLimit=int(time_limit)))
            solver_ok = True
        except Exception as e:
            err_msgs.append(f"HiGHS(cmd) failed: {e}")

    if not solver_ok:
        raise RuntimeError(
            "No MILP solver available on this host. "
            "Install one of: highspy (preferred) or system CBC. "
            + " | ".join(err_msgs)
        )

                                
    # Solve with CBC, fall back to HiGHS if needed
#    try:
 #       prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=int(time_limit), fracGap=frac_gap))
  #  except Exception:
   #     from pulp import HiGHS_CMD
    #    prob.solve(HiGHS_CMD(msg=False, timeLimit=int(time_limit)))

    status = pulp.LpStatus.get(prob.status, str(prob.status))

    counts = {L: 0 for L in stock_lengths}
    cut_plan = {L: [] for L in stock_lengths}
    for s in S:
        Ls = stock_lengths[s]
        for b in bars_for_s(s):
            if pulp.value(y[(s, b)]) > 0.5:
                counts[Ls] += 1
                assigned = [pieces[i] for i in I if pulp.value(z[(i, s, b)]) > 0.5]
                cut_plan[Ls].append(sorted(assigned, reverse=True))

    total_required = float(sum(pieces))
    total_used = float(sum(L * counts[L] for L in counts))
    waste = float(total_used - total_required)
    util = 0.0 if total_used == 0 else total_required / total_used

    return {
        "status": status,
        "counts": counts,
        "cut_plan": cut_plan,
        "total_required_length": total_required,
        "total_used_length": total_used,
        "total_waste": waste,
        "utilization": util,
        "width_piece": width_piece,
        "length_piece": length_piece,
        "n_width": int(qty),
        "n_length": int(2 * qty),
    }

# ---------------------------- UI ----------------------------
st.title("Single Project — Door Frame Cutting Optimization")

with st.expander("Inputs", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        frame_w = st.number_input("Frame_W (inches)", min_value=1.0, step=0.1, value=37.5)
        qty = st.number_input("LINE_QTY (# of doors)", min_value=1, step=1, value=6)
        oversize = st.number_input("Oversize factor", value=1.20, min_value=1.0, max_value=1.5, step=0.01)
    with col2:
        frame_l = st.number_input("Frame_L (inches)", min_value=1.0, step=0.1, value=97.5)
        kerf = st.number_input("Saw kerf per cut (inches)", value=0.0, min_value=0.0, max_value=0.25, step=0.01)
        time_limit = st.slider("Solver time limit (seconds)", min_value=10, max_value=300, value=60, step=10)

with st.expander("Inventory selection", expanded=True):
    units = st.radio("Inventory units", ["feet", "inches"], horizontal=True, index=0)
    if units == "feet":
        options = [6,8,9,10,11,12,13,14]
        default = [6,8,9,10,11,12,13,14]
        selected = st.multiselect("Choose stock sizes (feet)", options=options, default=default)
        stock_lengths = sorted({int(ft * 12) for ft in selected})
        st.caption(f"Using inventory (inches): {stock_lengths}")
    else:
        options = [72,96,108,120,132,144,156,168]
        default = [72,96,108,120,132,144,156,168]
        selected = st.multiselect("Choose stock sizes (inches)", options=options, default=default)
        stock_lengths = sorted({int(x) for x in selected})

run = st.button("Run optimization", type="primary")

if run:
    res = optimize_single_project(
        frame_w=frame_w, frame_l=frame_l, qty=qty,
        stock_lengths=stock_lengths, oversize=oversize,
        kerf=kerf, time_limit=time_limit, frac_gap=0.001
    )

    st.subheader("Results")
    st.write(f"**Status:** {res['status']}")
    st.write(f"**Width piece (after oversize):** {res['width_piece']:.3f}\" × {res['n_width']} pcs")
    st.write(f"**Length piece (after oversize):** {res['length_piece']:.3f}\" × {res['n_length']} pcs")

    # Summary metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Required (in)", f"{res['total_required_length']:.2f}")
    m2.metric("Used (in)", f"{res['total_used_length']:.2f}")
    m3.metric("Waste (in)", f"{res['total_waste']:.2f}")
    util_pct = 0.0 if res["total_used_length"] == 0 else 100.0 * res["utilization"]
    m4.metric("Utilization", f"{util_pct:.2f}%")

    # Counts by stock size
    counts_df = pd.DataFrame(
        [{"Stock (in)": L, "Count": res["counts"][L]} for L in sorted(res["counts"].keys())]
    )
    st.table(counts_df)

    # Optional: show cut plan per used bar
    with st.expander("Cut plan (pieces on each used stick)"):
        for L in sorted(res["cut_plan"].keys()):
            bars = [b for b in res["cut_plan"][L] if b]  # only used
            if not bars:
                continue
            st.markdown(f"**Stock {L}\"** — {len(bars)} used")
            for idx, pieces_on_bar in enumerate(bars, start=1):
                st.code(f"Bar {idx}: {pieces_on_bar}  | sum={sum(pieces_on_bar):.2f}\"")

    # CSV export (single row)
    out = {
        "Frame_W_in": frame_w, "Frame_L_in": frame_l, "LINE_QTY": qty,
        "Oversize": oversize, "Kerf_in": kerf, "Status": res["status"],
        "Total_Required_in": res["total_required_length"],
        "Total_Used_in": res["total_used_length"],
        "Total_Waste_in": res["total_waste"],
        "Utilization": res["utilization"],
    }
    for L in sorted(stock_lengths):
        out[f"Count_{L}in"] = res["counts"][L]
    csv_df = pd.DataFrame([out])
    st.download_button(
        "Download result as CSV",
        data=csv_df.to_csv(index=False).encode("utf-8"),
        file_name="single_project_optimization_result.csv",
        mime="text/csv",
    )

else:
    st.info("Enter inputs and click **Run optimization**.")
