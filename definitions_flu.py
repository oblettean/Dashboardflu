import re
import os
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import jinja2
import pathlib
import plotly.express as px
import streamlit.components.v1 as components
import hashlib
palette = px.colors.qualitative.Plotly
import numpy as np
from streamlit_plotly_events import plotly_events
from pathlib import Path
from io import StringIO
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

#PARTIE 1 : Aide a la confirmation
HISTO_FILE = "historique_fichiers.csv"
# Mapping global ID EPIISL → souche lisible
reference_map = {
    "EPIISL129744": "H3N2",
    "EPIISL200780": "H1N1",
    "EPIISL219327": "BVic",
}
# --- Témoins (global, insensible à la casse) ---
TEMOINS = ["TposH1", "TposH3", "TposB"]

def load_static_css(file_path: str):
    """Load a CSS file and inject into Streamlit."""
    css_file = pathlib.Path(file_path)
    if css_file.exists():
        css_content = css_file.read_text()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"CSS file not found: {file_path}")
        
def answered(val):
    if isinstance(val, bool):
        return True
    if val is None:
        return False
    if isinstance(val, str):
        s = val.strip().lower()
        return s != "" and s != "sélectionnez"
    return bool(val)
    
def page_aide():
    return None

# ✅ Fonction d'extraction de plaque
def extraire_plaque(sample_id):
    if not isinstance(sample_id, str):
        return None
    match = re.search(r'\d{2}P\d{3}GR[A-Z]', sample_id)
    if match:
        return match.group(0)
    if "-" in sample_id:
        return sample_id.split("-")[0]
    return None

def is_yes(v):
    if isinstance(v, bool):
        return v is True
    return str(v).strip().lower() in {"oui", "yes", "true", "1"}

def is_no(v):
    if isinstance(v, bool):
        return v is False
    return str(v).strip().lower() in {"non", "no", "false", "0"}

def _wrap_label(s: str, width: int = 14) -> str:
    """Insère des <br> pour éviter que les libellés trop longs débordent."""
    s = str(s)
    out, line = [], []
    for word in s.split():
        if sum(len(w) for w in line) + len(line) + len(word) > width:
            out.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        out.append(" ".join(line))
    return "<br>".join(out)

# ── Helpers robustes lecture/écriture d'historique ─────────────────────────────
import os

def safe_read_historique_data(path: str) -> pd.DataFrame:
    """Lecture 'sûre' de l'historique : tout en str, pas de low_memory."""
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)

def persist_full_dataset_atomic(df: pd.DataFrame, path: str) -> None:
    """Écrit df -> path de façon atomique (tmp + replace)."""
    tmp_path = path + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, path)

def filter_gra_group(group):
    target_refs = ["EPIISL200780", "EPIISL129744"]
    target_rows = group[group["summary_reference_id"].isin(target_refs)]
    if target_rows.empty:
        return pd.DataFrame()
    counts = target_rows["summary_bam_readcount"]
    if (counts > 0).any():
        return target_rows[counts > 0]
    else:
        row = target_rows.iloc[[0]].copy()
        row["summary_reference_id"] = "NA"
        return row

def count_double_pop(df, epiisl_reference, seuil=6):
    df2 = df.copy()
    df2["summary_vcf_dpcount"] = pd.to_numeric(df2["summary_vcf_dpcount"], errors="coerce")
    # On retourne une vraie copie du filtrage
    return df2.loc[
        (df2["summary_reference_id"] == epiisl_reference) &
        (df2["summary_vcf_dpcount"] >= seuil)
    ].copy()

def make_counts(df):
    return {
        "H3N2": len(df[df["summary_reference_id"] == "EPIISL129744"]),
        "H1N1": len(df[df["summary_reference_id"] == "EPIISL200780"]),
        "BVic": len(df[df["summary_reference_id"] == "EPIISL219327"])
    }

def render_table(df, title):
    if not df.empty:
        st.markdown(f"#### {title}")
        styled = (
            df[["sample_id", "plaque_id", "summary_run_id", "souche_EPIISL", "clade", "similarity_score"]]
            .style.set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}])
            .set_properties(**{"text-align": "center"})
            .set_properties(**{"font-weight": "bold"}, subset=["similarity_score"]))
        st.markdown(
            f"<div style='display: flex; justify-content: center;'>"
            f"<div style='width: 90%;'>{styled.to_html(escape=False, index=False)}</div></div>",
            unsafe_allow_html=True)

def render_table_cmap(
    df,
    label: str,
    cmap: str = "Reds",
    columns: list = None
):
    if df is None or df.empty:
        return

    # Colonnes à afficher
    default_cols = [
        "sample_id", "plaque_id", "summary_run_id",
        "souche_EPIISL", "clade", "similarity_score"
    ]
    cols = columns or [c for c in default_cols if c in df.columns]
    if not cols:
        cols = list(df.columns)

    df_disp = df[cols].copy()

    # Conserver numérique pour le gradient, formater à l’affichage
    if "similarity_score" in df_disp.columns:
        df_disp["similarity_score"] = pd.to_numeric(df_disp["similarity_score"], errors="coerce")

    styled = (
        df_disp.style
            .set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}])
            .set_properties(**{"text-align": "center"})
    )
    if "similarity_score" in df_disp.columns:
        styled = styled.background_gradient(subset=["similarity_score"], cmap=cmap)\
                       .format({"similarity_score": "{:.2f}"})

    # Titre + tableau (pas d’expander)
    st.markdown(f"#### {label} – {len(df_disp)} échantillon(s)")
    st.markdown(
        f"<div style='width:95%;margin:auto;'>{styled.to_html(index=False)}</div>",
        unsafe_allow_html=True
    )

def register_section_html(title: str, html: str, counts: int = None, plaque: str = None, sections: list = None):
    section = {"title": title, "html": html, "counts": counts, "plaque": plaque}
    sections.append(section)

def register_section(title, df, cols, empty_msg, counts=None, plaque=None, sections=None):
    # mapping des sous-types
    subtype_map = {
        "EPIISL129744": "H3N2",
        "EPIISL200780": "H1N1",
        "EPIISL219327": "BVic"
    }

    if df is None or df.empty:
        html = f"<div class='message'>{empty_msg}</div>"
        sections.append({
            "title": title,
            "html": html,
            "counts": make_counts(df),
            "plaque": plaque
        })
        return

    # Si la colonne summary_reference_id n'existe pas → pas de split par souche
    if "summary_reference_id" not in df.columns:
        html = add_table(df, cols, empty_msg, counts=counts, plaque=plaque)
        sections.append({
            "title": title,
            "html": html,
            "counts": counts,
            "plaque": plaque
        })
        return

    # ✅ Split par sous-type
    html_blocks = []
    for ref, label in subtype_map.items():
        df_sub = df[df["summary_reference_id"] == ref]
        if df_sub.empty:
            continue
        html_blocks.append(f"<h4>{label} – {len(df_sub)} échantillons</h4>")
        html_blocks.append(add_table(df_sub, cols, empty_msg))

    # Ajout au rapport
    sections.append({
        "title": title,
        "html": "\n".join(html_blocks),
        "counts": make_counts(df),
        "plaque": plaque
    })

def build_pie_div(df_plaque: pd.DataFrame) -> str:
    col = "summary_vcf_coinf01match"
    if col not in df_plaque.columns:
        return ""

    data = df_plaque[col].value_counts()
    if data.empty:
        return ""

    labels = data.index.astype(str).tolist()
    values = data.values.tolist()

    total = float(sum(values)) if sum(values) else 1.0
    percents = [v / total for v in values]
    SMALL = 0.06
    textpos = ["outside" if p < SMALL else "inside" for p in percents]

    labels_wrapped = [_wrap_label(lbl, width=14) for lbl in labels]

    palette = px.colors.qualitative.Plotly
    fig = go.Figure(
        data=[go.Pie(
            labels=labels_wrapped,
            values=values,
            hole=0,
            marker=dict(colors=palette[:len(labels_wrapped)]),
            textinfo="none",
            texttemplate="%{label}<br>%{percent}",
            textposition=textpos,
            insidetextorientation="radial",
            hovertemplate="<b>%{label}</b><br>%{percent} (%{value})<extra></extra>",
            textfont_size=12,
            showlegend=False
        )]
    )

    fig.update_layout(
        margin=dict(t=10, r=40, b=120, l=40),
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig.to_html(full_html=False, include_plotlyjs="cdn", config={"responsive": True})

def detect_souches(ids, filtered_sim, base_df, threshold=0.1):
    """
    ids: iterable d'identifiants échantillons (ex: failed_ids / warning_ids)
    filtered_sim: DataFrame de similarité (index: références [MultiIndex ou str], colonnes: sample_id)
    base_df: non utilisé ici sauf pour compat future (on ne renomme pas vos DataFrames)
    threshold: score minimal pour retenir une détection
    """
    out = []

    for s in ids:
        if s not in filtered_sim.columns:
            continue

        # Colonne des similarités pour l'échantillon s
        col = filtered_sim[s].dropna()
        if col.empty:
            continue

        # Meilleure référence (score max)
        ref_idx = col.idxmax()
        score = float(col.loc[ref_idx])

        if score <= threshold:
            continue

        # Déduire EPI/segment selon structure d'index
        epi, seg = None, None
        if isinstance(filtered_sim.index, pd.MultiIndex) and filtered_sim.index.nlevels >= 2:
            try:
                epi, seg = ref_idx[0], ref_idx[1]
            except Exception:
                epi, seg = str(ref_idx), None
        else:
            ref_str = str(ref_idx)
            if isinstance(ref_idx, tuple) and len(ref_idx) >= 2:
                epi, seg = ref_idx[0], ref_idx[1]
            elif "_" in ref_str:
                epi, seg = ref_str.split("_", 1)
            else:
                epi, seg = ref_str, None

        out.append({
            "sample_id": s,
            "souche_EPIISL": epi,
            "segment": seg,
            "clade": reference_map.get(epi, epi),
            "similarity_score": score
        })

    return pd.DataFrame(out, columns=["sample_id", "souche_EPIISL", "segment", "clade", "similarity_score"])

def make_report_html(
    controls: dict,
    sections: list,
    plot_div: str,
    plaques: list,
    plaque_selected: str,
    counts: dict = None,
    run_id=None
) -> str:
    # 1) On construit le chemin absolu vers le dossier templates
    template_dir = Path(__file__).parent / "templates"
    # 2) On l’utilise pour le loader Jinja
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        autoescape=True
    )
    # 3) On charge le bon fichier
    tpl = env.get_template("rapport_template.html")
    generation_date = datetime.now().strftime("%d %B %Y")
    current_year   = datetime.now().year
    return tpl.render(
        controls=controls,
        sections=sections,
        plaques=plaques,
        plaque_selected=plaque_selected,
        generation_date=generation_date,
        current_year=current_year,
        counts=counts or {},
        run_id=run_id
    )

def build_ininterpretable_html(df_plaque: pd.DataFrame) -> str:
    # Filtrer les ininterprétables
    df_plaque["summary_consensus_perccoverage_S4"] = pd.to_numeric(
        df_plaque["summary_consensus_perccoverage_S4"], errors="coerce"
    )
    df_plaque["summary_consensus_perccoverage_S6"] = pd.to_numeric(
        df_plaque["summary_consensus_perccoverage_S6"], errors="coerce"
    )
    mask = (
        (df_plaque["summary_consensus_perccoverage_S4"] < 90) |
        (df_plaque["summary_consensus_perccoverage_S6"] < 90)
    )
    df_int = df_plaque.loc[mask].drop_duplicates(subset="sample_id")
    if df_int.empty:
        return ""  # on retournera une section vide

    # Construire pour chaque souche
    souche_map = {
        "H3N2": "EPIISL129744",
        "H1N1": "EPIISL200780",
        "Bvic": "EPIISL219327"
    }
    blocks = []
    for label, ref in souche_map.items():
        sub = df_int[df_int["summary_reference_id"] == ref]
        if sub.empty:
            continue
        blocks.append(f"<h4>{label} – {len(sub)} échantillons</h4>")
        styled = (
            sub[[
                "sample_id","plaque_id","summary_run_id",
                "summary_consensus_perccoverage_S4","summary_consensus_perccoverage_S6"
            ]]
            .style
            .set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}])
            .set_properties(**{"text-align": "center"})
            .set_properties(**{"font-weight": "bold"},
                            subset=["summary_consensus_perccoverage_S4",
                                    "summary_consensus_perccoverage_S6"])
        )
        blocks.append(styled.to_html(escape=False, index=False))
    return "\n".join(blocks)

def build_secondary_html(df_plaque: pd.DataFrame) -> str:
    """
    Génère le HTML des souches secondaires (score > 0.4) pour une plaque donnée.
    """
    df_plaque["similarity_score"] = pd.to_numeric(df_plaque["similarity_score"], errors="coerce")
    df_sec = df_plaque[df_plaque["similarity_score"] > 0.4]
    if df_sec.empty:
        return ""

    subtype_map = {
        "EPIISL129744": "H3N2",
        "EPIISL200780": "H1N1",
        "EPIISL219327": "BVic"
    }

    blocks = []
    for ref, label in subtype_map.items():
        sub = df_sec[df_sec["summary_reference_id"] == ref]
        if sub.empty:
            continue
        blocks.append(f"<h4>{label} – {len(sub)} échantillons</h4>")
        styled = (
            sub[["sample_id", "plaque_id", "summary_run_id", "souche_EPIISL", "clade", "similarity_score"]]
            .style
            .set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}])
            .set_properties(**{"text-align": "center"})
            .set_properties(**{"font-weight": "bold"}, subset=["similarity_score"])
        )
        blocks.append(styled.to_html(escape=False, index=False))
    return "\n".join(blocks)

def build_interclade_html(df_plaque: pd.DataFrame) -> str:
    """
    Génère le HTML de la section coinfection inter-clade (IQR>0) pour une plaque donnée,
    avec découpage par sous-type.
    """
    df_plaque["summary_vcf_coinf02iqr"] = pd.to_numeric(
        df_plaque.get("summary_vcf_coinf02iqr", 0),
        errors="coerce"
    )
    df_int = df_plaque[df_plaque["summary_vcf_coinf02iqr"] > 0]
    if df_int.empty:
        return ""

    subtype_map = {
        "EPIISL129744": "H3N2",
        "EPIISL200780": "H1N1",
        "EPIISL219327": "BVic"
    }

    blocks = []
    for ref, label in subtype_map.items():
        sub = df_int[df_int["summary_reference_id"] == ref]
        if sub.empty:
            continue
        blocks.append(f"<h4>{label} – {len(sub)} échantillons</h4>")
        styled = (
            sub[["sample_id", "plaque_id", "summary_vcf_coinf02iqr"]]
            .style
            .set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}])
            .set_properties(**{"text-align": "center"})
            .background_gradient(subset=["summary_vcf_coinf02iqr"], cmap="Reds")
            .set_properties(**{"font-weight": "bold"}, subset=["summary_vcf_coinf02iqr"])
        )
        blocks.append(styled.to_html(escape=False, index=False))
    return "\n".join(blocks)

def add_table(df: pd.DataFrame,
              cols: list,
              empty_msg: str,
              counts: int = None,
              plaque: str = None) -> str:
    buf = StringIO()
    buf.write('<div class="table-section">')
    if df is None or df.empty:
        buf.write(f"<div class='message'>{empty_msg}</div>")
    else:
        df2 = df.copy()[cols]

        for c in df2.select_dtypes(include=["float", "int"]):
            df2[c] = df2[c].map(lambda x: f"{x:.2f}")

        # ✅ Afficher le nombre uniquement si c’est un int
        if isinstance(counts, int):
            buf.write(f'<p><strong>Nombre d’échantillons : {counts}</strong></p>')

        buf.write(df2.to_html(index=False,
                              classes="table table-bordered data-table mb-0"))

    buf.write('</div>')
    return buf.getvalue()

def append_history(nom_fichier: str, taille_ko: float, run_id: str = "", operateur: str = ""):
    """Ajoute une ligne à l'historique persistant (CSV)."""
    new_row = {
        "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "run_tsv",
        "nom_fichier": nom_fichier,
        "taille_Ko": round(taille_ko, 2),
        "run_id": run_id,
        "operateur": operateur,
    }
    if os.path.exists(HISTO_FILE):
        df_h = pd.read_csv(HISTO_FILE)
        # assure les colonnes
        for col in new_row:
            if col not in df_h.columns:
                df_h[col] = ""
        df_h = pd.concat([df_h, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df_h = pd.DataFrame([new_row])
    df_h.to_csv(HISTO_FILE, index=False)

#PARTIE 2 : Plan de plaque

def page_plaque():
    return None

# --- constantes plaques ---
ROWS96  = list("ABCDEFGH")            # 8 lignes
COLS96  = list(range(1, 13))          # 12 colonnes

ROWS384 = list("ABCDEFGHIJKLMNOP")    # 16 lignes
COLS384 = list(range(1, 25))          # 24 colonnes

# pour la découpe R1/R2 (haut/bas) dans la 384
ROWS384_TOP    = list("ABCDEFGH")     # R1 = haut
ROWS384_BOTTOM = list("IJKLMNOP")     # R2 = bas

def _norm(s):
    """Normalise un en-tete/valeur Excel (strip + lower)."""
    s = "" if s is None else str(s)
    return s.replace("\xa0", " ").strip().lower()

def _have_all(cols):
    """Verifie la presence des colonnes attendues (normalisees)."""
    cols_norm = {_norm(c) for c in cols}
    return {"positionpl384", "samplesheet", "sampleproject"}.issubset(cols_norm)

def _well_positions(rows, cols):
    return [(c, r) for r in rows for c in cols]  # (col, row)

def _occupied_from_sample_ids(df_pl, rows):
    """
    Deduit les puits occupes (set of tuples (col, row)) a partir des 'S#' trouves
    dans sample_id, selon le mapping colonne-major:
    S1->A1, S2->B1, ..., S8->H1, S9->A2, etc.
    """
    occupied = set()
    if df_pl is None or df_pl.empty or "sample_id" not in df_pl.columns:
        return occupied

    n_rows = len(rows)
    for _, r in df_pl.iterrows():
        sid = str(r.get("sample_id", ""))
        m = re.search(r"S(\d+)", sid, flags=re.IGNORECASE)
        if not m:
            continue
        s_idx = int(m.group(1))
        if s_idx <= 0:
            continue
        rr = (s_idx - 1) % n_rows          # index ligne
        cc = (s_idx - 1) // n_rows         # index colonne (0-based)
        occupied.add((cc + 1, rows[rr]))   # (col_number, row_letter)
    return occupied
    
def _finish_axes96(fig, rows, cols, height, showlegend=False):
    """Applique la mise en forme standard pour une plaque 96."""
    fig.update_xaxes(
        side="top",
        tickmode="array", tickvals=list(cols), ticktext=[str(c) for c in cols],
        showgrid=False, zeroline=False, showline=False, ticks=""
    )
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array", tickvals=list(rows), ticktext=list(rows),
        showgrid=False, zeroline=False, showline=False, ticks=""
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        plot_bgcolor="white",
        showlegend=showlegend
    )
    return fig


def _make_plate_fig_96(df_pl, marker_size=40, height=520, *,
                       grid_color="#EEEEEE", occupied_color="#5B9BD5",
                       ctrl_neg_color="#2ECC71", ctrl_pos_color="#E74C3C",
                       empty_color="#BDBDBD"):
    """
    Affiche une plaque 96 :
      - fond gris clair pour tous les puits,
      - puits 'normaux' en bleu,
      - NT1/NT2/NT3 en vert avec label,
      - TposH1/H3/B en rouge avec label,
      - TVIDE en gris foncé,
      - hover : 'A1 — sample_id'.
    """
    rows = ROWS96
    cols = COLS96
    wells = _well_positions(rows, cols)

    # Grille (fond)
    x_all = [c for c, r in wells]
    y_all = [r for c, r in wells]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_all, y=y_all,
        mode="markers",
        marker=dict(symbol="circle", size=marker_size,
                    color=grid_color, line=dict(color="black", width=1)),
        hoverinfo="text",
        text=[f"{r}{c}" for c, r in wells],
        name=""
    ))

    if df_pl is None or df_pl.empty or "sample_id" not in df_pl.columns:
        return _finish_axes96(fig, rows, cols, height, showlegend=False)

    # Mapping sample_id -> puits
    n_rows = len(rows)
    occ = {}
    for _, rec in df_pl.iterrows():
        sid = str(rec.get("sample_id", "") or "")
        m = re.search(r"S(\d+)", sid, flags=re.IGNORECASE)
        if not m:
            continue
        s_idx = int(m.group(1))
        if s_idx <= 0:
            continue

        # Normalise en modulo 96 → de 1 à 96
        pos96 = ((s_idx - 1) % 96) + 1

        rr = (pos96 - 1) % n_rows   # ligne
        cc = (pos96 - 1) // n_rows  # colonne
        row_letter = rows[rr]
        col_num = cc + 1
        key = (row_letter, col_num)
        if key not in occ:
            occ[key] = sid

    if not occ:
        return _finish_axes96(fig, rows, cols, height, showlegend=False)

    occ_items = [(r, c, occ[(r, c)]) for (r, c) in sorted(occ.keys(), key=lambda k: (k[1], rows.index(k[0])))]
    rows_v = [r for r, c, s in occ_items]
    cols_v = [c for r, c, s in occ_items]
    sids_v = [s for r, c, s in occ_items]
    name_u = pd.Series(sids_v, dtype=str).str.upper()

    # Détections
    m_tvide = name_u.str.contains("TVIDE", na=False)
    nt_digit = name_u.str.extract(r"NT\s*[-_]*\s*([1-3])", flags=re.IGNORECASE, expand=False)
    m_nt = nt_digit.notna()
    pos_match = name_u.str.extract(r"TPOS\s*(H1|H3|B)", flags=re.IGNORECASE, expand=False)
    m_pos = pos_match.notna() & (~m_nt)

    # Labels
    nt_labels = pd.Series("", index=range(len(sids_v)))
    nt_labels.loc[m_nt] = "NT" + nt_digit[m_nt]
    pos_labels = pd.Series("", index=range(len(sids_v)))
    pos_labels.loc[m_pos] = pos_match[m_pos].str.upper()

    # Normaux = pas de ctrl, pas de TVIDE
    m_norm = (~m_nt) & (~m_pos) & (~m_tvide)

    # Puits normaux
    if m_norm.any():
        idx = m_norm[m_norm].index.tolist()
        fig.add_trace(go.Scatter(
            x=[cols_v[i] for i in idx], y=[rows_v[i] for i in idx],
            mode="markers",
            marker=dict(symbol="circle", size=marker_size,
                        color=occupied_color, line=dict(color="black", width=1)),
            hovertemplate="%{customdata}",
            customdata=[f"{rows_v[i]}{int(cols_v[i])} — {sids_v[i]}" for i in idx],
            name="Echantillon",
            showlegend=False
        ))

    # TVIDE
    if m_tvide.any():
        idx = m_tvide[m_tvide].index.tolist()
        fig.add_trace(go.Scatter(
            x=[cols_v[i] for i in idx], y=[rows_v[i] for i in idx],
            mode="markers",
            marker=dict(symbol="circle", size=marker_size,
                        color=empty_color, line=dict(color="black", width=1)),
            hovertemplate="%{customdata}",
            customdata=[f"{rows_v[i]}{int(cols_v[i])} — {sids_v[i]}" for i in idx],
            name="Tvide",
            showlegend=False
        ))

    # NT
    if m_nt.any():
        idx = m_nt[m_nt].index.tolist()
        fig.add_trace(go.Scatter(
            x=[cols_v[i] for i in idx], y=[rows_v[i] for i in idx],
            mode="markers+text",
            marker=dict(symbol="circle", size=marker_size,
                        color=ctrl_neg_color, line=dict(color="black", width=1)),
            text=[nt_labels[i] for i in idx],
            textposition="middle center",
            textfont=dict(color="white", size=max(8, int(marker_size*0.35))),
            hovertemplate="%{customdata}",
            customdata=[f"{rows_v[i]}{int(cols_v[i])} — {sids_v[i]}" for i in idx],
            name="Tneg",
            showlegend=False
        ))

    # Tpos
    if m_pos.any():
        idx = m_pos[m_pos].index.tolist()
        fig.add_trace(go.Scatter(
            x=[cols_v[i] for i in idx], y=[rows_v[i] for i in idx],
            mode="markers+text",
            marker=dict(symbol="circle", size=marker_size,
                        color=ctrl_pos_color, line=dict(color="black", width=1)),
            text=[pos_labels[i] for i in idx],
            textposition="middle center",
            textfont=dict(color="white", size=max(8, int(marker_size*0.35))),
            hovertemplate="%{customdata}",
            customdata=[f"{rows_v[i]}{int(cols_v[i])} — {sids_v[i]}" for i in idx],
            name="Tpos",
            showlegend=False
        ))

    return _finish_axes96(fig, rows, cols, height, showlegend=False)


# =============================
# Excel -> positions 384
# =============================
_pos_re = re.compile(r"^\s*C\s*(\d{1,2})\s*-\s*R\s*([12])\s*$", re.IGNORECASE)

def _map_excel_to_384_positions(xdf: pd.DataFrame) -> pd.DataFrame:
    """
    Attend un DataFrame contenant les colonnes (insensibles à la casse/espaces) :
      - PositionPl384 (ex: 'C1-R1', 'C3-R2')
      - Samplesheet
      - Sampleproject

    Retourne un DF avec colonnes:
      row (A..P), col (1..24), samplesheet, sampleproject
    """
    # Détection souple des colonnes
    cols_map = { _norm(c): c for c in xdf.columns }
    needed = ["positionpl384", "samplesheet", "sampleproject"]
    if not all(k in cols_map for k in needed):
        missing = [k for k in needed if k not in cols_map]
        raise ValueError(f"Colonnes manquantes dans l'Excel : {missing}")

    df = xdf[[cols_map["positionpl384"], cols_map["samplesheet"], cols_map["sampleproject"]]].copy()
    df.columns = ["PositionPl384", "Samplesheet", "Sampleproject"]

    # Nettoyage soft (on conserve la casse d'origine pour l'affichage pas upper ici)
    df["PositionPl384"] = df["PositionPl384"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    df["Samplesheet"]   = df["Samplesheet"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()
    df["Sampleproject"] = df["Sampleproject"].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

    # Parse 'Ck-Rx'
    parsed = df["PositionPl384"].str.extract(_pos_re)
    df["col_num"] = pd.to_numeric(parsed[0], errors="coerce")
    df["half"]    = parsed[1].astype(str)

    # Filtrer positions invalides
    df = df[df["col_num"].between(1, 24) & df["half"].isin(["1", "2"])].copy()

    # Groupement par position 384 (8 puits haut/bas)
    out = []
    for (col_num, half), g in df.groupby(["col_num", "half"], sort=False):
        g = g.reset_index(drop=True)
        rows_block = ROWS384_TOP if half == "1" else ROWS384_BOTTOM

        # Si >8, tronque et avertit cote appelant si besoin (ici: on tronque juste)
        g = g.iloc[:8].copy()

        for i, row in enumerate(g.itertuples(index=False)):
            out.append({
                "row": rows_block[i],          # A..H (R1) ou I..P (R2)
                "col": int(col_num),
                "samplesheet": row.Samplesheet,
                "sampleproject": row.Sampleproject
            })

    return pd.DataFrame(out, columns=["row", "col", "samplesheet", "sampleproject"])

# =============================
# Plot plaque 384 (ronds + tooltip)
# =============================
def _make_plate384_fig_from_map(
    df_map: pd.DataFrame,
    *,
    row_col=("row", "col"),
    samplesheet_col="samplesheet",
    project_col="sampleproject",
    tvide_token="TVIDE",
    viro_project="VIRO-GRIPPE",
    marker_size=30,
    width=640,
    height=820,
    showlegend=False,            # on forcerá False dans le layout
    highlight_project_not=None,  # compat inutile
):
    if df_map is None or df_map.empty:
        return go.Figure()

    ROWS_384 = list("ABCDEFGHIJKLMNOP")
    COLS_384 = list(range(1, 25))

    # --- Couleurs fixes
    GRID_COLOR   = "#EEEEEE"
    EMPTY_COLOR  = "#BDBDBD"
    NEUTRAL_PALE = "#F2F2F2"
    VIRO_BASE    = "#1f77b4"   # bleu franc
    VIRO_GRA_COL = "#1a5f8f"   # plus soutenu
    VIRO_GRB_COL = "#56a3d8"   # plus clair
    CTRL_NEG_COLOR = "#2ecc71" # vert
    CTRL_POS_COLOR = "#e74c3c" # rouge

    # --- Helpers couleur
    def _rgb_to_hex(rgb):
        r,g,b = [max(0, min(255, int(round(x)))) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"

    def _hsl_to_rgb(h, s, l):
        # h ∈ [0,1], s ∈ [0,1], l ∈ [0,1]
        def hue2rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        if s == 0:
            r = g = b = l
        else:
            q = l * (1 + s) if l < 0.5 else l + s - l * s
            p = 2 * l - q
            r = hue2rgb(p, q, h + 1/3)
            g = hue2rgb(p, q, h)
            b = hue2rgb(p, q, h - 1/3)
        return (r*255, g*255, b*255)

    def _distinct_color(i, n, s=0.55, l=0.88):
        # palette sans collision : n couleurs espacées sur le cercle (golden ratio)
        # i ∈ [0, n-1]
        golden = 0.61803398875
        h = (i * golden) % 1.0
        return _rgb_to_hex(_hsl_to_rgb(h, s, l))

    # --- Normalisation
    row_key, col_key = row_col
    for c in (row_key, col_key, samplesheet_col, project_col):
        if c not in df_map.columns:
            raise ValueError(f"Colonnes manquantes : {', '.join([x for x in (row_key, col_key, samplesheet_col, project_col) if x not in df_map.columns])}")

    df = df_map.copy()
    df[row_key] = df[row_key].astype(str).str.strip().str.upper()
    df[col_key] = pd.to_numeric(df[col_key], errors="coerce").astype("Int64")
    for c in (samplesheet_col, project_col):
        df[c] = df[c].astype(str).replace({"<NA>": ""}).fillna("").str.strip()

    bad_rows = ~df[row_key].isin(ROWS_384)
    bad_cols = ~df[col_key].isin(COLS_384)
    if bad_rows.any() or bad_cols.any():
        bad = df.loc[bad_rows | bad_cols, [row_key, col_key]].head(10)
        raise ValueError(
            "Valeurs de puits invalides. Attendu: rows A..P, cols 1..24.\n"
            f"Exemples invalides (max 10):\n{bad.to_string(index=False)}"
        )

    df[row_key] = pd.Categorical(df[row_key], categories=ROWS_384, ordered=True)

    # --- Flags
    name_u = df[samplesheet_col].fillna("").astype(str).str.upper()
    proj    = df[project_col].fillna("").astype(str).str.strip()
    proj_u  = proj.str.upper()

    is_empty = name_u.str.contains(str(tvide_token).upper(), na=False)
    is_viro  = proj_u.eq(str(viro_project).upper())
    has_proj = proj.str.len() > 0

    # GRA / GRB (pour VIRO hors contrôles)
    viro_gra = name_u.str.contains(r"\bGRA\b", na=False)
    viro_grb = name_u.str.contains(r"\bGRB\b", na=False)

    # CONTRÔLES (uniquement VIRO)
    nt_digit = name_u.str.extract(r"NT\s*[-_]*\s*([1-3])", flags=re.IGNORECASE, expand=False)
    m_viro_nt = is_viro & nt_digit.notna()
    nt_labels = pd.Series("", index=df.index)
    nt_labels.loc[m_viro_nt] = "NT" + nt_digit[m_viro_nt]

    pos_match = name_u.str.extract(r"TPOS\s*(H1|H3|B)", flags=re.IGNORECASE, expand=False)
    m_viro_pos = is_viro & pos_match.notna() & (~m_viro_nt)  # NT prioritaire
    pos_labels = pd.Series("", index=df.index)
    pos_labels.loc[m_viro_pos] = pos_match[m_viro_pos].str.upper()

    # --- Couleurs par projet (unicité garantie)
    # on construit la liste ordonnée de projets ≠ VIRO et ≠ vide
    projects = sorted(p for p in proj.unique() if p and p.upper() != viro_project.upper())
    n = len(projects)
    project_colors = {}
    for i, p in enumerate(projects):
        # couleur pâle unique par projet
        project_colors[p] = _distinct_color(i, max(n, 1), s=0.50, l=0.92)

    # couleur utilisée pour VIRO dans la légende externe
    project_colors_viro = {viro_project: VIRO_BASE}

    # tableau des couleurs pour chaque puits
    colors = np.full(len(df), NEUTRAL_PALE, dtype=object)
    mask_other = has_proj & (~is_viro)
    if mask_other.any():
        colors_other = [ project_colors.get(p, NEUTRAL_PALE) for p in proj[mask_other] ]
        colors[mask_other.to_numpy()] = np.array(colors_other, dtype=object)

    # VIRO hors contrôles : base + nuances GRA/GRB
    m_viro_no_ctrl = is_viro & (~m_viro_nt) & (~m_viro_pos)
    if m_viro_no_ctrl.any():
        colors[m_viro_no_ctrl.to_numpy()] = VIRO_BASE
        mgra = (m_viro_no_ctrl & viro_gra).to_numpy()
        mgrb = (m_viro_no_ctrl & viro_grb).to_numpy()
        colors[mgra] = VIRO_GRA_COL
        colors[mgrb] = VIRO_GRB_COL

    # TVIDE override
    if is_empty.any():
        colors[is_empty.to_numpy()] = EMPTY_COLOR

    # --- Hover
    hover_text = (
        df[row_key].astype(str)
        + df[col_key].astype(int).astype(str)
        + " — "
        + df[samplesheet_col].replace("", "(sans nom)")
        + np.where(df[project_col] != "", " | " + df[project_col], "")
    )

    # --- Grille (fond)
    x_grid = np.repeat(COLS_384, len(ROWS_384))
    y_grid = np.tile(ROWS_384, len(COLS_384))

    fig = go.Figure()

    # Fond
    fig.add_trace(go.Scatter(
        x=x_grid, y=y_grid,
        mode="markers",
        marker=dict(symbol="circle", size=marker_size, line=dict(width=1, color="black"), color=GRID_COLOR),
        hoverinfo="skip",
        showlegend=False
    ))

    # ===== Traces par projet (hors contrôles) =====
    # 1) VIRO-GRIPPE (hors NT/H1/H3/B) — une seule trace, avec couleurs point-par-point (GRA/GRB)
    m_viro_no_ctrl = is_viro & (~m_viro_nt) & (~m_viro_pos)
    if m_viro_no_ctrl.any():
        # couleurs VIRO point-par-point (base/GRA/GRB)
        viro_colors = np.where(
            viro_gra[m_viro_no_ctrl], VIRO_GRA_COL,
            np.where(viro_grb[m_viro_no_ctrl], VIRO_GRB_COL, VIRO_BASE)
        ).tolist()

        fig.add_trace(go.Scatter(
            x=df.loc[m_viro_no_ctrl, col_key].astype(int),
            y=df.loc[m_viro_no_ctrl, row_key].astype(str),
            mode="markers",
            marker=dict(symbol="circle", size=marker_size, line=dict(width=1, color="black"), color=viro_colors),
            hovertemplate="%{customdata}",
            customdata=hover_text[m_viro_no_ctrl],
            showlegend=False,
            name=viro_project  # IMPORTANT : le nom de trace = nom du projet pour le toggle
        ))

    # 2) Autres projets : une trace par projet, couleur pâle unique
    projects_other = sorted(p for p in proj.unique() if p and p.upper() != viro_project.upper())
    for p in projects_other:
        m_p = (proj == p) & (~m_viro_nt) & (~m_viro_pos)  # “hors contrôles” ne s’applique qu’à VIRO, ok
        if not m_p.any():
            continue
        fig.add_trace(go.Scatter(
            x=df.loc[m_p, col_key].astype(int),
            y=df.loc[m_p, row_key].astype(str),
            mode="markers",
            marker=dict(symbol="circle", size=marker_size, line=dict(width=1, color="black"),
                        color=project_colors.get(p, NEUTRAL_PALE)),
            hovertemplate="%{customdata}",
            customdata=hover_text[m_p],
            showlegend=False,
            name=p  # IMPORTANT : nom de trace = nom du projet
        ))


    # TÉMOIN NEG (NT*)
    if m_viro_nt.any():
        fig.add_trace(go.Scatter(
            x=df.loc[m_viro_nt, col_key].astype(int),
            y=df.loc[m_viro_nt, row_key].astype(str),
            mode="markers+text",
            marker=dict(symbol="circle", size=marker_size, line=dict(width=1, color="black"), color=CTRL_NEG_COLOR),
            text=nt_labels.loc[m_viro_nt].tolist(),
            textposition="middle center",
            textfont=dict(color="white", size=max(8, int(marker_size*0.35))),
            hovertemplate="%{customdata}",
            customdata=hover_text[m_viro_nt],
            showlegend=False,
            name=""
        ))

    # TÉMOIN POS (TposH1/H3/B)
    if m_viro_pos.any():
        fig.add_trace(go.Scatter(
            x=df.loc[m_viro_pos, col_key].astype(int),
            y=df.loc[m_viro_pos, row_key].astype(str),
            mode="markers+text",
            marker=dict(symbol="circle", size=marker_size, line=dict(width=1, color="black"), color=CTRL_POS_COLOR),
            text=pos_labels.loc[m_viro_pos].tolist(),
            textposition="middle center",
            textfont=dict(color="white", size=max(8, int(marker_size*0.35))),
            hovertemplate="%{customdata}",
            customdata=hover_text[m_viro_pos],
            showlegend=False,
            name=""
        ))

    # Axes + layout
    fig.update_xaxes(
        side="top",
        tickmode="array", tickvals=COLS_384, ticktext=[str(c) for c in COLS_384],
        ticklabelposition="inside top",
        showgrid=False, zeroline=False, showline=False, ticks="", ticklen=0,
        automargin=True
    )
    fig.update_yaxes(
        autorange="reversed",
        tickmode="array", tickvals=ROWS_384, ticktext=ROWS_384,
        showgrid=False, zeroline=False, showline=False, ticks="",
        automargin=True
    )
    fig.update_layout(
        margin=dict(l=8, r=8, t=10, b=8),
        width=width,
        height=height,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        plot_bgcolor="white",
        showlegend=False
    )

    # ✅ meta : on expose les couleurs pour fabriquer la légende externe
    meta = {"project_colors": project_colors, "viro_color": VIRO_BASE, "viro_project": viro_project}
    fig.update_layout(meta=meta)

    return fig

#PARTIE 3 : Suivie de performance

def page_suivi():
    return None
# === Helpers UI/plots pour l'onglet 3 (aucun rename de DF) ===
def _parse_flags_from_comment(txt: str) -> str:
    s = str(txt or "").lower()
    flags = []
    if ("nouveau" in s or "nouvelle" in s) and "lot" in s: flags.append("lot")
    if "nc" in s: flags.append("nc")
    if "pb" in s or "probl" in s: flags.append("pb")
    return ",".join(flags)

def add_lot_labels_on_top(fig, temoin_df, temoin: str, x_col: str = "plaque_id"):
    """
    Ajoute de petites étiquettes 'Lot XXX' en haut du graphe aux changements de lot.
    Ne modifie pas les DF persistant; utilise extract_lot_for_temoin() déjà existante.
    """
    import numpy as np
    from definitions_flu import extract_lot_for_temoin

    sub = temoin_df[temoin_df["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty: 
        return fig

    if x_col in sub.columns:
        xorder = x_col
    elif "date_heure" in sub.columns:
        xorder = "date_heure"
    elif "summary_run_id" in sub.columns:
        xorder = "summary_run_id"
    else:
        xorder = "plaque_id" if "plaque_id" in sub.columns else "sample_id"

    sub = sub.sort_values([xorder, "sample_id"]).reset_index(drop=True)
    sub["__lot__"] = extract_lot_for_temoin(sub, temoin)  # déjà dans definitions_flu.py

    # Détecte les *points de changement* de lot (première occurrence de chaque lot)
    change_mask = sub["__lot__"].ne(sub["__lot__"].shift(1)) & sub["__lot__"].notna()
    marks = sub.loc[change_mask, [xorder, "__lot__"]]

    # Ajoute des annotations discrètes en haut (sans vline pour éviter les catégories)
    for _, r in marks.iterrows():
        fig.add_annotation(
            x=r[xorder], y=105, xref="x", yref="y",
            text=f"Lot {r['__lot__']}", showarrow=False,
            bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.15)", borderwidth=1,
            font=dict(size=11), align="center"
        )
    return fig

def assign_lots_from_commentaires(df):
    df_sorted = df.sort_values(by=["plaque_id", "sample_id"])
    lot_actuel = None
    lot_colonne = pd.Series(index=df_sorted.index, dtype=object)
    for plaque, group in df_sorted.groupby("plaque_id"):
        group_index = group.index
        new_lots = {}
        for i, idx in enumerate(group_index):
            commentaire = str(group.loc[idx, 'commentaire']).lower()
            match = re.search(r"lot\s*(\d+)", commentaire)
            if ('nouveau' in commentaire or 'nouvelle' in commentaire) and match:
                new_lots[i] = match.group(1)
        group_lots = [None] * len(group_index)
        for i, idx in enumerate(group_index):
            if i in new_lots:
                lot_actuel = new_lots[i]
            group_lots[i] = lot_actuel
        lot_colonne.loc[group_index] = group_lots
    df_sorted["lot_assigné"] = lot_colonne
    return df_sorted

def has_special_comment(comment):
    if pd.isna(comment):
        return False
    comment = comment.lower()
    return ('nouveau' in comment or 'nouvelle' in comment) and ('lot' in comment)

def _x_order_col(df: pd.DataFrame) -> str:
    """
    Retourne le nom de la colonne à utiliser comme 'ordre X':
    priorité: 'date_heure' > 'summary_run_id' > 'plaque_id'.
    Crée une colonne '__x_order__' triable si besoin.
    """
    if "date_heure" in df.columns:
        # On garde tel quel (texte) – l’ordre lexical YYYY-MM-DD HH:MM:SS marche
        df["__x_order__"] = df["date_heure"].astype(str)
    elif "summary_run_id" in df.columns:
        df["__x_order__"] = df["summary_run_id"].astype(str)
    else:
        df["__x_order__"] = df.get("plaque_id", "").astype(str)
    return "__x_order__"

def has_cols(df: pd.DataFrame, cols: list[str]) -> bool:
    """Vérifie que toutes les colonnes sont présentes, sinon loggue proprement."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes: {missing}")
        return False
    return True
def update_comment_and_persist(base_df: pd.DataFrame, plaque_id: str, sample_id: str, new_comment: str, data_file: str):
    """Met à jour base_df en place (commentaire) et persiste le *dataset complet* dans data_file."""
    if not has_cols(base_df, ["plaque_id", "sample_id", "commentaire"]):
        return base_df
    mask = (base_df["plaque_id"] == plaque_id) & (base_df["sample_id"] == sample_id)
    if mask.any():
        base_df.loc[mask, "commentaire"] = new_comment
        # Sauvegarde *complète* et sûre
        tmp_path = data_file + ".tmp"
        base_df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, data_file)
        st.success("✅ Commentaire mis à jour (dataset complet sauvegardé).")
    else:
        st.warning("ℹ️ Échantillon non retrouvé dans base_df (aucune sauvegarde).")
    return base_df

def extract_lot_corrected(df_sorted: pd.DataFrame) -> list:
    """
    Pour chaque ligne, renvoie le lot actif (H1/H3/B indépendants) en lisant 'commentaire'.
    La logique est pilotée par l'ordre X canonique.
    """
    if df_sorted is None or df_sorted.empty:
        return []

    df = df_sorted.copy()
    xcol = _x_order_col(df)
    df.sort_values([xcol, "sample_id"], inplace=True)  # ordre stable

    lot_by_type = {"h1": None, "h3": None, "b": None}
    lots_out = []

    for _, row in df.iterrows():
        commentaire = str(row.get("commentaire", "")).lower()
        sid = str(row.get("sample_id", "")).lower()

        t = "h1" if "tposh1" in sid else "h3" if "tposh3" in sid else "b" if "tposb" in sid else None

        # Mise à jour du lot pour le type détecté si on voit "nouveau/nouvelle ... lot XXX"
        m = re.search(r"(nouveau|nouvelle).*?lot\s*([a-z0-9_\-]+)", commentaire, flags=re.IGNORECASE)
        if t and m:
            lot_by_type[t] = m.group(2)

        lots_out.append(lot_by_type.get(t) if t else None)

    return lots_out

def extract_lot_for_temoin(df_temoin: pd.DataFrame, temoin_code: str) -> list:
    """
    Calcule le lot actif UNIQUEMENT pour le témoin demandé (TposH1 / TposH3 / TposB).
    - On parcourt df_temoin dans l'ordre (date_heure > run_id > plaque_id > sample_id).
    - On met à jour le lot quand on voit "nouveau/nouvelle ... lot XXX" dans 'commentaire'.
    - Les lignes qui ne correspondent pas au temoin_code reçoivent None.
    """
    if df_temoin is None or df_temoin.empty:
        return []

    df = df_temoin.copy()

    # Ordre robuste
    if "date_heure" in df.columns:
        xorder = "date_heure"
    elif "summary_run_id" in df.columns:
        xorder = "summary_run_id"
    elif "plaque_id" in df.columns:
        xorder = "plaque_id"
    else:
        xorder = "sample_id"

    df = df.sort_values([xorder, "sample_id"])

    # Détecteur du type
    t = temoin_code.lower()
    def _is_this_temoin(sid: str) -> bool:
        sid = str(sid).lower()
        if t == "tposh1": return "tposh1" in sid
        if t == "tposh3": return "tposh3" in sid
        if t == "tposb":  return "tposb"  in sid
        return False

    lot_actuel = None
    lots_out = []
    for _, row in df.iterrows():
        commentaire = str(row.get("commentaire", "")).lower()
        sid = str(row.get("sample_id", ""))
        if _is_this_temoin(sid):
            m = re.search(r"(nouveau|nouvelle).*?lot\s*([a-z0-9_\-]+)", commentaire, flags=re.IGNORECASE)
            if m:
                lot_actuel = m.group(2)
            lots_out.append(lot_actuel)
        else:
            # lignes d’un autre témoin (s’il y en a) -> pas de lot
            lots_out.append(None)

    return lots_out
    
# --- simplification : pas de périodes, juste un segment par lot -------------
def add_lot_segments(df_sorted: pd.DataFrame) -> pd.DataFrame:
    """
    Compatibilité pour l'appelant : ne segmente plus.
    Se contente de garantir la présence de 'lot_actif'.
    """
    df = df_sorted.copy()
    if "lot_actif" not in df.columns:
        df["lot_actif"] = extract_lot_corrected(df)
    # colonnes “héritées” pour éviter toute référence brisée
    df["lot_segment"] = 1
    df["period_idx"] = 1
    df["period_tag"] = "A"
    return df

def compute_x_grouped_id(sample_id):
    sample_id = str(sample_id) if not pd.isna(sample_id) else ""
    base = sample_id[:9]
    suffix = ''
    sample_id_lower = sample_id.lower()
    if 'tposh1' in sample_id_lower:
        suffix = 'TposH1'
    elif 'tposh3' in sample_id_lower:
        suffix = 'TposH3'
    elif 'tposb' in sample_id_lower:
        suffix = 'TposB'
    return f"{base}-{suffix}" if suffix else sample_id

def _check_cols(df, needed, label=""):
    miss = [c for c in needed if c not in df.columns]
    if miss:
        st.warning(f"Colonnes manquantes {label}: {miss}")
        return False
    return True

def _lot_label(lot):
    return "NA" if (pd.isna(lot) or lot is None or str(lot).strip()=="") else str(lot)
    
def plot_temoin_lots_s4s6_unique(
    temoin_df: pd.DataFrame,
    temoin: str,
    seuil: float = 90,
    x_col: str = "plaque_id",
    show_dropdown: bool = False  # ignoré ici
):
    need = {"sample_id", "summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6", "commentaire"}
    if not _check_cols(temoin_df, need, label="S4/S6"):
        return go.Figure()

    # 1) Isoler le témoin demandé
    sub = temoin_df[temoin_df["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty:
        st.info(f"Aucun {temoin} dans la sélection courante.")
        return go.Figure()

    # 2) Ordre X
    def _x_order_col_local(d):
        if x_col in d.columns: return x_col
        if "date_heure" in d.columns: return "date_heure"
        if "summary_run_id" in d.columns: return "summary_run_id"
        return "plaque_id" if "plaque_id" in d.columns else "sample_id"

    xorder = _x_order_col_local(sub)
    sub = sub.sort_values([xorder, "sample_id"]).reset_index(drop=True)

    # 3) Colonnes numériques
    sub["S4"] = pd.to_numeric(sub["summary_consensus_perccoverage_S4"], errors="coerce")
    sub["S6"] = pd.to_numeric(sub["summary_consensus_perccoverage_S6"], errors="coerce")

    # 4) Lot PAR TÉMOIN
    sub["lot_actif"] = extract_lot_for_temoin(sub, temoin)

    # 5) Flag “commentaire spécial” → losange
    #    (has_special_comment renvoie True si "nouveau/nouvelle ... lot XXX" est présent)
    sub["special_comment"] = sub["commentaire"].apply(has_special_comment)

    # 6) Libellés & couleurs par lot
    def _lot_lbl(v): 
        return "Sans lot" if (pd.isna(v) or str(v).strip()=="") else str(v)
    sub["lot_lbl"] = sub["lot_actif"].map(_lot_lbl)
    lots = sub["lot_lbl"].drop_duplicates().tolist()
    base = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    color_map = {lot: base[i % len(base)] for i, lot in enumerate(lots)}

    # 7) Tracé : 1 segment par lot (pas de périodes)
    fig = go.Figure()
    for lot, g in sub.groupby("lot_lbl", dropna=False):
        g = g.sort_values([xorder, "sample_id"])
        color = color_map.get(lot, "#7f7f7f")
        name_core = f"Lot {lot}"

        # symboles : cercle/square par défaut, ♦ si commentaire spécial
        s4_symbols = ["diamond" if v else "circle" for v in g["special_comment"]]
        s6_symbols = ["diamond" if v else "square" for v in g["special_comment"]]

        mode_s4 = "lines+markers" if g["S4"].notna().sum() >= 2 else "markers"
        fig.add_trace(go.Scatter(
            x=g[xorder], y=g["S4"],
            mode=mode_s4,
            name=name_core,
            legendgroup=f"{lot}-S4",
            showlegend=True,
            line=dict(color=color, width=3, dash="solid"),
            marker=dict(symbol=s4_symbols, size=9),
            hovertemplate="Sample: %{customdata[0]}<br>"+xorder+": %{x}<br>S4: %{y:.1f}%<br>"+name_core,
            customdata=g[["sample_id"]].values
        ))

        mode_s6 = "lines+markers" if g["S6"].notna().sum() >= 2 else "markers"
        fig.add_trace(go.Scatter(
            x=g[xorder], y=g["S6"],
            mode=mode_s6,
            name=f"{name_core} — S6",
            legendgroup=f"{lot}-S4",
            showlegend=False,
            line=dict(color=color, width=3, dash="dot"),
            marker=dict(symbol=s6_symbols, size=9),
            hovertemplate="Sample: %{customdata[0]}<br>"+xorder+": %{x}<br>S6: %{y:.1f}%<br>"+name_core,
            customdata=g[["sample_id"]].values
        ))

    # Seuil horizontal
    fig.add_hline(y=seuil, line=dict(color="red", dash="dash"),
                  annotation_text=f"Seuil {seuil}%", annotation_position="top left")

    fig.update_layout(
        title=f"{temoin} — S4 (plein) & S6 (pointillé) · 1 segment/lot (commentaire spécial en ♦)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(255,255,255,0.85)", bordercolor="rgba(0,0,0,0.1)", borderwidth=1),
        hoverlabel=dict(namelength=-1),
        xaxis_title=xorder, yaxis_title="% Couverture",
        yaxis=dict(range=[0,110]),
        xaxis=dict(tickangle=-45),
        margin=dict(l=20, r=20, t=80, b=60),
    )
    return fig


    
def plot_temoin_autres_segments_curves(temoin_df: pd.DataFrame, temoin: str, x_col: str = "plaque_id"):
    present = [c for c in [f"summary_consensus_perccoverage_S{i}" for i in range(1,9)] if c in temoin_df.columns]
    others = [c for c in present if c not in ["summary_consensus_perccoverage_S4","summary_consensus_perccoverage_S6"]]
    need = {"sample_id","commentaire"} | set(others)
    if not _check_cols(temoin_df, need, label="Autres segments"):
        return go.Figure()

    # 1) Filtrer le bon témoin d'abord
    sub = temoin_df[temoin_df["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty:
        return go.Figure()

    # 2) Ordre X
    def _x_order_col_local(d):
        if x_col in d.columns: return x_col
        if "date_heure" in d.columns: return "date_heure"
        if "summary_run_id" in d.columns: return "summary_run_id"
        return "plaque_id" if "plaque_id" in d.columns else "sample_id"

    xorder = _x_order_col_local(sub)
    sub = sub.sort_values([xorder, "sample_id"]).reset_index(drop=True)

    # 3) Lot PAR TÉMOIN
    sub["lot_actif"] = extract_lot_for_temoin(sub, temoin)

    # 4) Numérisation
    for c in others:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    def _lot_lbl(v):
        return "Sans lot" if (pd.isna(v) or str(v).strip()=="") else str(v)
    sub["lot_lbl"] = sub["lot_actif"].map(_lot_lbl)

    seg_names = [f"S{c[-1]}" for c in others]
    dash_by_seg = {s: ["solid","dot","dash","longdash","dashdot","longdashdot"][i % 6] for i, s in enumerate(seg_names)}

    fig = go.Figure()
    # 1 segment par lot
    for lot, g in sub.groupby("lot_lbl", dropna=False):
        g = g.sort_values([xorder, "sample_id"])
        for c in others:
            sname = f"S{c[-1]}"
            y = g[c]
            mode = "lines+markers" if y.notna().sum() >= 2 else "markers"
            fig.add_trace(go.Scatter(
                x=g[xorder], y=y,
                mode=mode,
                name=f"{sname} · Lot {lot}",
                line=dict(width=1.5, dash=dash_by_seg[sname]),
                marker=dict(size=5),
                hovertext=[f"{r['sample_id']}<br>{xorder}:{r[xorder]}<br>{sname}: {r[c]:.1f}%<br>Lot:{lot}"
                           for _, r in g.iterrows()],
                hoverinfo="text",
                showlegend=False
            ))

    fig.update_layout(
        title=f"{temoin} — Autres segments (1 segment/lot, isolation par témoin)",
        xaxis_title=xorder, yaxis_title="% Couverture",
        yaxis=dict(range=[0,110]),
        xaxis=dict(tickangle=-45),
        margin=dict(l=20,r=20,t=50,b=60)
    )
    return fig


    
def plot_histogram_with_export(df, titre, filename):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["plaque_id"],
        y=df["pct_ininterpretable"],
        text=[f"{pct:.1f}%" for pct in df["pct_ininterpretable"]],
        textposition='outside',
        marker_color='indianred',
        name="% Ininterprétable"
    ))
    fig.update_layout(
        title=titre,
        xaxis_title="Plaque ID",
        yaxis_title="% Ininterprétable",
        yaxis=dict(range=[0, 100]),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=100),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    # Préparer CSV à télécharger
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()
    st.download_button(
        label="Télécharger les données CSV",
        data=csv_data,
        file_name=filename,
        mime='text/csv')

# --- Fonction pour tracer avec % au-dessus ---
def plot_varcount(df, titre):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["plaque_id"],
        y=df["% varcount >= 13"],
        text=[f"{pct:.1f}%" for pct in df["% varcount >= 13"]],
        textposition='outside',
        marker_color='seagreen',
        name="% varcount >= 13"
    ))
    fig.update_layout(
        title=titre,
        xaxis_title="Plaque ID",
        yaxis_title="% avec varcount >= 13",
        yaxis=dict(range=[0, 100]),
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=50, b=100),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
def contains_any(df, col, keywords):
    """True si col contient au moins un mot-clé (regex-safe)."""
    safe = [re.escape(str(k)) for k in keywords if pd.notna(k) and str(k) != ""]
    if not safe:
        return pd.Series(False, index=df.index)
    pat = "|".join(safe)
    return df[col].astype(str).str.contains(pat, case=False, na=False, regex=True)

def display_grippe_temoins_complet(df, seuil: float, col=st):
    """
    Affiche un récap rapide des témoins, stats S4/S6, et un tableau des lignes concernées.
    Prérequis: colonnes summary_consensus_perccoverage_S4/S6, sample_id, commentaire (optionnelle).
    """
    if df is None or df.empty:
        col.info("⚠️ Aucun témoin : DataFrame vide.")
        return

    # Filtrage témoins robuste
    mask_temoins = contains_any(df, "sample_id", TEMOINS)
    df_t = df.loc[mask_temoins].copy()
    if df_t.empty:
        col.warning("⚠️ Aucun témoin détecté dans sample_id.")
        # Panneau debug utile
        with col.expander("🔍 Debug témoins"):
            col.write({"TEMOINS": TEMOINS, "nb_lignes_df": int(len(df))})
            col.dataframe(df[["sample_id"]].head(50), use_container_width=True)
        return

    # Types sûrs
    for c in ["summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"]:
        if c in df_t.columns:
            df_t[c] = pd.to_numeric(df_t[c], errors="coerce")

    # Stats globales par témoin
    col.markdown("### 🧪 Récap témoins")
    for t in TEMOINS:
        tmask = df_t["sample_id"].astype(str).str.contains(re.escape(t), case=False, na=False)
        tdf = df_t[tmask]
        if tdf.empty:
            col.info(f"— {t}: 0 échantillon")
            continue
        s4 = tdf["summary_consensus_perccoverage_S4"].mean()
        s6 = tdf["summary_consensus_perccoverage_S6"].mean()
        n_below = (
            tdf[["summary_consensus_perccoverage_S4","summary_consensus_perccoverage_S6"]]
            .lt(seuil).any(axis=1).sum()
        )
        col.write(f"**{t}** — n={len(tdf)} | sous seuil ({seuil}%) : **{n_below}** | S4 moy: **{s4:.2f}%** | S6 moy: **{s6:.2f}%**")

    # Tableau compact
    show_cols = [c for c in ["sample_id","plaque_id","summary_run_id","summary_consensus_perccoverage_S4","summary_consensus_perccoverage_S6","commentaire","lot_actif"] if c in df_t.columns]
    col.dataframe(df_t[show_cols].sort_values("plaque_id"), use_container_width=True)

#PARTIE 4 : Historique : 

def page_historique():
    return None