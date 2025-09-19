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
from functools import lru_cache
palette = px.colors.qualitative.Plotly
import numpy as np
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
# Lien nom de tuile -> type
TEMOIN_TO_TYPE = {
    "TposH3": "H3N2",
    "TposH1": "H1N1",
    "TposB":  "BVic",
}
# Inverse le mapping EPIISL -> type
REF_BY_TYPE = {v: k for k, v in reference_map.items()}

COMMENT_NEWLOT_IMPLIES_CURRENT = True

NEW_TPOS_PROMOTE_RE = re.compile(
    r"(?:nouveau\s*lot\s*t[ée]moin)\s*[:\-]?\s*([A-Za-z0-9][\w\-\/]*)",
    re.IGNORECASE
)


def expected_ref_for_temoin(temoin: str):
    """
    Retourne l'ID EPIISL attendu pour un témoin positif (TposH3/H1/B),
    ou None si non applicable (NT1/NT2/NT3).
    """
    t = TEMOIN_TO_TYPE.get(temoin)
    return REF_BY_TYPE.get(t) if t else None


# --- Témoins (global, insensible à la casse) ---
TEMOINS = ["TposH1", "TposH3", "TposB"]

def load_static_css(file_path: str):
    """Load a CSS file and inject into Streamlit, robust to Windows encodings."""
    css_file = pathlib.Path(file_path)
    if not css_file.exists():
        st.warning(f"CSS file not found: {file_path}")
        return
    try:
        # recommandé : fichiers CSS en UTF-8
        css_content = css_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            # fallback Windows si le fichier a été enregistré en ANSI
            css_content = css_file.read_text(encoding="cp1252")
        except UnicodeDecodeError:
            # dernier recours : on ignore les octets illisibles (évite le crash)
            css_content = css_file.read_bytes().decode("utf-8", errors="ignore")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

        
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
# === Nouveau : traçabilité des lots témoins ===
from datetime import datetime

TPOS_LOTS_LOG_FILE = "temoin_lots_log.csv"  # journal dédié aux changements de lot Tpos

def log_temoin_lot_event(
    temoin: str, lot_number: str, scope: str,
    run_id: str = "", plaque_id: str = "", note: str = "", operateur: str = "",
    action: str = "", old_lot: str = ""
):
    row = {
        "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event": "temoin_lot",
        "temoin": str(temoin),
        "lot": str(lot_number).strip(),
        "old_lot": str(old_lot).strip(),
        "action": str(action).strip(),   # "nouveau" | "en_cours" | "promotion" | "clos"
        "scope": scope,
        "run_id": run_id or "",
        "plaque_id": plaque_id or "",
        "note": note or "",
        "operateur": operateur or "",
    }

    if os.path.exists(TPOS_LOTS_LOG_FILE):
        df = pd.read_csv(TPOS_LOTS_LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(TPOS_LOTS_LOG_FILE, index=False)

    # 2) Historique global (résumé) — s’affiche déjà dans l’onglet “Historique”
    # On remplit les colonnes existantes (nom_fichier sert ici d’intitulé)
    hist_row = {
        "date_heure": row["date_heure"],
        "type": "temoin_lot",
        "nom_fichier": f"{temoin}:{row['lot']}",
        "run_id": row["run_id"],
        "operateur": operateur or "",
    }
    if os.path.exists(HISTO_FILE):
        df_h = pd.read_csv(HISTO_FILE)
        # Assure les colonnes attendues (robuste si schéma évolue)
        for col in ["date_heure", "type", "nom_fichier", "run_id", "operateur"]:
            if col not in df_h.columns:
                df_h[col] = ""
        df_h = pd.concat([df_h, pd.DataFrame([hist_row])], ignore_index=True)
    else:
        df_h = pd.DataFrame([hist_row])
    df_h.to_csv(HISTO_FILE, index=False)
    
def undo_last_temoin_action(base_df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(TPOS_LOTS_LOG_FILE):
        return base_df
    df_log = pd.read_csv(TPOS_LOTS_LOG_FILE)
    if df_log.empty:
        return base_df
    last = df_log.iloc[-1].to_dict()
    temoin = last.get("temoin","")
    lot    = last.get("lot","")
    scope  = last.get("scope","Plaque")
    run_id = last.get("run_id","")
    plaque = last.get("plaque_id","")

    tags = [
        f"nouveau lot {lot}",
        f"lot en cours {lot}",
        f"lot clos {lot}",
    ]
    def _rm_tags(s: str) -> str:
        parts = [p.strip() for p in str(s or "").split(";") if p and p.strip()!=""]
        keep  = [p for p in parts if not any(p.lower()==t.lower() for t in tags)]
        return " ; ".join(keep)

    df = base_df.copy()
    m = df["sample_id"].astype(str).str.contains(str(temoin), case=False, na=False)
    if scope == "Run" and "summary_run_id" in df.columns and run_id:
        m = m & df["summary_run_id"].astype(str).eq(str(run_id))
        # ≥ plaque de départ
        def _idx(pid):
            m2 = re.search(r"(\d+)", str(pid)); return int(m2.group(1)) if m2 else None
        start = _idx(plaque)
        m = m & df["plaque_id"].map(_idx).apply(lambda i: i is not None and start is not None and i>=start)
    elif scope == "Plaque" and plaque:
        m = m & df["plaque_id"].astype(str).eq(str(plaque))

    df.loc[m, "commentaire"] = df.loc[m, "commentaire"].apply(_rm_tags)

    # trace l'undo
    hist_row = {
        "date_heure": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":"undo", "nom_fichier": f"{temoin}:{lot}", "run_id": str(run_id), "operateur": ""
    }
    try:
        df_h = pd.read_csv(HISTO_FILE) if os.path.exists(HISTO_FILE) else pd.DataFrame()
        for c in ["date_heure","type","nom_fichier","run_id","operateur"]:
            if c not in df_h.columns: df_h[c] = ""
        df_h = pd.concat([df_h, pd.DataFrame([hist_row])], ignore_index=True)
        df_h.to_csv(HISTO_FILE, index=False)
    except Exception:
        pass

    return df

# === Tab1: construire le commentaire à partir des réponses
def _make_tab1_comment(nouvelle_dilution, details_dilution,
                       problemes_tech, problemes_tech_ex,
                       non_conf, num_nc, justification_nc) -> str:
    parts = []
    if is_yes(nouvelle_dilution):
        txt = "Nouvelle dilution amorces"
        if answered(details_dilution):
            txt += f": {details_dilution}"
        parts.append(txt)

    if is_yes(problemes_tech):
        txt = "PB technique"
        if answered(problemes_tech_ex):
            txt += f": {problemes_tech_ex}"
        parts.append(txt)

    if is_yes(non_conf) and answered(num_nc):
        parts.append(f"NC {num_nc}")
    elif is_no(non_conf) and answered(justification_nc):
        parts.append(f"Justif NC: {justification_nc}")

    return " | ".join([p for p in parts if p])

# --- TUILES TÉMOINS POUR LE RAPPORT HTML ------------------------------------
def build_temoin_tiles_html(df_plaque: pd.DataFrame, df_history: pd.DataFrame = None) -> str:
    """
    Rend les tuiles témoins pour le rapport HTML.
    - TposH3/H1/B : gère duplicats (2 tuiles empilées), 'Lot en cours' vs 'Période probatoire'
      calculés via l'HISTORIQUE (df_history) pour coller au comportement de l'app.
    - NT1/NT2/NT3 : une seule tuile, comme dans l'app.
    """

    import pandas as pd

    if df_plaque is None or df_plaque.empty:
        return "<p><em>Aucun témoin détecté pour cette plaque.</em></p>"

    df = df_plaque.copy()
    for c in ["summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    witness_specs = [
        ("TposH3", "TposH3", "3C.2a1b.2a.2a"),
        ("TposH1", "TposH1", "6B.1A.5a.2"),
        ("TposB",  "TposB",  "B"),
        ("NT1",    "NT1",    None),
        ("NT2",    "NT2",    None),
        ("NT3",    "NT3",    None),
    ]

    def _fmt_pct(v):
        v = pd.to_numeric(v, errors="coerce")
        if pd.isna(v):
            return "—"
        return f"{float(v):.1f}%"

    def _fmt_txt(v):
        if v is None:
            return "—"
        s = str(v).strip()
        return "—" if (s == "" or s.lower() == "nan") else s

    # CSS plus compact + grille responsive
    tiles = []
    tiles.append("""
      <style>
        .tile{border-radius:10px;padding:10px;margin-bottom:10px}
        .tile-head{display:flex;justify-content:space-between;align-items:center}
        .tile-title{font-size:18px;font-weight:700}
        .tile-ribbon{font-size:11px;padding:2px 8px;border-radius:999px;background:#fff;border:1px solid}
        .tile-ok{background:#e6ffed;border:2px solid #2ecc71}
        .tile-grey{background:#f6f7fb;border:2px solid #bdc3c7}
        .tile-warn{background:#ffe6e6;border:2px solid #e74c3c}
        .grid-tiles{display:grid;grid-template-columns:repeat(1,minmax(0,1fr));gap:12px}
        @media (min-width: 900px){.grid-tiles{grid-template-columns:repeat(3,minmax(0,1fr));}}
        @media (min-width: 1400px){.grid-tiles{grid-template-columns:repeat(4,minmax(0,1fr));}}
        .tile-body{font-size:16px;line-height:1.5}
        .tile-meta{font-size:13px;color:#555;margin-top:4px}
      </style>
      <div class="grid-tiles">
    """)

    # historique global pour calculer lot & stage
    df_hist = df_history if (df_history is not None and not df_history.empty) else df

    for (title, pattern, attendu) in witness_specs:
        base_mask = df["sample_id"].astype(str).str.contains(pattern, case=False, na=False)

        # --- Tpos* (filtrage + duplicats) ---
        if pattern in ("TposH3", "TposH1", "TposB"):
            ref_attendue = expected_ref_for_temoin(pattern)
            mask = base_mask
            if ref_attendue and "summary_reference_id" in df.columns:
                mask = mask & df["summary_reference_id"].astype(str).eq(ref_attendue)

            df_t = (
                df.loc[mask]
                  .sort_values("sample_id")
                  .drop_duplicates(subset=["sample_id"], keep="first")
                  .reset_index(drop=True)
            )

            if df_t.empty:
                tiles.append(f"""
                  <div class="tile tile-grey">
                    <div class="tile-title">❓ {title} non trouvé</div>
                  </div>
                """)
                continue

            # Mapping lot/stage à partir de l'HISTORIQUE (base_df / new_data_filtered)
            try:
                sub_hist = df_hist[df_hist["sample_id"].astype(str).str.contains(pattern, case=False, na=False)].copy()
                lots_hist, stages_hist = extract_lot_and_stage_for_temoin(sub_hist, pattern)
                lot_map   = dict(zip(sub_hist["sample_id"].astype(str), lots_hist))
                stage_map = dict(zip(sub_hist["sample_id"].astype(str), stages_hist))
            except Exception:
                lot_map, stage_map = {}, {}

            # Fallback si historique muet : 1er = courant, 2e = probatoire
            if all(stage_map.get(str(sid)) in (None, "", "None") for sid in df_t["sample_id"].astype(str)):
                for j, sid in enumerate(df_t["sample_id"].astype(str).tolist()):
                    stage_map[sid] = "courant" if j == 0 else ("probatoire" if len(df_t) > 1 else "courant")

            for j, row in df_t.iterrows():
                sid = str(row.get("sample_id", ""))
                clade = _fmt_txt(row.get("summary_vcf_coinf01match", "—"))
                s4 = _fmt_pct(row.get("summary_consensus_perccoverage_S4", None))
                s6 = _fmt_pct(row.get("summary_consensus_perccoverage_S6", None))

                lot_here = lot_map.get(sid, None)
                lot_lbl  = _fmt_txt(lot_here)
                stage    = str(stage_map.get(sid, "") or "").lower()
                is_current = ("cours" in stage) or (stage == "courant")

                ribbon   = "Lot en cours" if is_current else "Période probatoire"
                tile_cls = "tile-ok" if is_current else "tile-grey"
                icon     = "🧪" if (attendu is None or (attendu in clade)) else "⚠️"
                border   = "#2ecc71" if is_current else "#bdc3c7"

                tiles.append(f"""
                  <div class="tile {tile_cls}">
                    <div class="tile-head">
                      <div class="tile-title">{icon} {title}</div>
                      <span class="tile-ribbon" style="border-color:{border}">{ribbon}</span>
                    </div>
                    <div class="tile-meta">ID : <b>{sid}</b></div>
                    <div class="tile-meta">Lot associé : <b>{lot_lbl}</b></div>
                    <div class="tile-body">
                      S4 : <b>{s4}</b><br>
                      S6 : <b>{s6}</b><br>
                      <span style="font-weight:700;color:#0066cc">{clade}</span>
                    </div>
                  </div>
                """)

        # --- NT* : UNE seule tuile, comme en app ---
        else:
            df_nt = df.loc[base_mask]
            if df_nt.empty:
                tiles.append(f"""
                  <div class="tile tile-grey">
                    <div class="tile-title">❓ {title} non trouvé</div>
                  </div>
                """)
                continue

            r0 = df_nt.iloc[0]
            clade = _fmt_txt(r0.get("summary_vcf_coinf01match", "—"))
            s4 = _fmt_pct(r0.get("summary_consensus_perccoverage_S4", None))
            s6 = _fmt_pct(r0.get("summary_consensus_perccoverage_S6", None))
            ok = (attendu is None) or (attendu in clade)
            icon = "🧪" if ok else "⚠️"
            tile_cls = "tile-ok" if ok else "tile-warn"

            tiles.append(f"""
              <div class="tile {tile_cls}">
                <div class="tile-title">{icon} {title}</div>
                <div class="tile-body">
                  S4 couverture : <b>{s4}</b><br>
                  S6 couverture : <b>{s6}</b><br>
                  <span style="font-weight:700;color:#0066cc">{clade}</span>
                </div>
              </div>
            """)

    tiles.append("</div>")
    return "\n".join(tiles)

    
def apply_new_lot_for_temoin(base_df: pd.DataFrame, df_filtered: pd.DataFrame, temoin: str,
                             lot_number: str, scope: str,
                             data_file: str,
                             run_id: str = "", plaque_id: str = "", note: str = "") -> tuple[pd.DataFrame, int]:
    """
    Applique le marquage 'nouveau lot <lot_number>' au *premier point d’ancrage pertinent*
    pour le témoin demandé, selon la portée choisie (Selon filtres / Plaque / Run).

    - Ne renomme aucun DataFrame.
    - Met à jour base_df + persist (via update_comment_and_persist).
    - Retourne (base_df_mis_a_jour, nb_modifs)
    """
    if base_df is None or base_df.empty or df_filtered is None:
        return base_df, 0
    need = {"sample_id", "plaque_id", "commentaire"}
    if not has_cols(base_df, list(need)) or not has_cols(df_filtered, list(need)):
        return base_df, 0

    # 1) Sous-ensemble du témoin
    sub = df_filtered[df_filtered["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty:
        return base_df, 0

    # 2) Portée
    if scope == "Plaque" and plaque_id:
        sub = sub[sub["plaque_id"].astype(str) == str(plaque_id)]
    elif scope == "Run" and run_id and "summary_run_id" in sub.columns:
        sub = sub[sub["summary_run_id"].astype(str) == str(run_id)]

    if sub.empty:
        return base_df, 0

    # 3) Choix d’un *ancrage unique* (évite d’inonder les commentaires partout)
    xcol = _x_order_col(sub)
    sub = sub.sort_values([xcol, "sample_id"], ascending=[True, True])
    anchor = sub.iloc[0]
    pl = str(anchor["plaque_id"])
    sid = str(anchor["sample_id"])

    # 4) Construire le nouveau commentaire (append sémantique + dédup)
    cur = base_df.loc[(base_df["plaque_id"] == pl) & (base_df["sample_id"] == sid), "commentaire"]
    current_comment = cur.iloc[0] if len(cur) else ""
    parts = [p.strip() for p in str(current_comment).split(";") if p is not None and str(p).strip() != ""]
    tag = f"nouveau lot {str(lot_number).strip()}"
    if not any(tag.lower() == p.lower() for p in parts):
        parts.append(tag)
    if note and str(note).strip():
        if not any(str(note).strip().lower() == p.lower() for p in parts):
            parts.append(str(note).strip())
    new_comment = " ; ".join(parts)

    # 5) Mise à jour + persistance atomique
    base_df = update_comment_and_persist(base_df, pl, sid, new_comment, data_file)
    return base_df, 1

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

def _ensure_headroom(fig, data_peak=None, y_min=0.0, y_target_min=125.0):
    try:
        current_top = float(fig.layout.yaxis.range[1]) if fig.layout.yaxis and fig.layout.yaxis.range else None
    except Exception:
        current_top = None
    data_peak = float(data_peak) if (data_peak is not None) else 100.0
    # marge visuelle pour laisser respirer les badges
    y_top = max(y_target_min, (current_top or 0.0), data_peak + 28.0)
    fig.update_yaxes(range=[y_min, y_top])
    fig.update_layout(
        margin=dict(
            l=getattr(fig.layout.margin, "l", 40),
            r=getattr(fig.layout.margin, "r", 30),
            t=max(95, getattr(fig.layout.margin, "t", 80)),
            b=max(100, getattr(fig.layout.margin, "b", 90)),
        ),
        hovermode="closest",
        hoverdistance=10,
    )
    return y_top


# === Badges & helpers commentaires (PARTIE 3) =================================
def comment_badges(run_id: str = "", plaque_id: str = "") -> str:
    """Construit un petit bandeau HTML avec un badge Run et/ou Plaque s'ils ont un commentaire simple."""
    try:
        run_txt = get_run_comment(run_id) if run_id else ""
    except Exception:
        run_txt = ""
    try:
        pl_txt = get_plaque_comment(plaque_id) if plaque_id else ""
    except Exception:
        pl_txt = ""
    if not run_txt and not pl_txt:
        return ""
    def _chip(lbl, txt):
        txt = (str(txt) or "").strip()
        if len(txt) > 140: txt = txt[:140] + "…"
        return f"""
        <div style="display:inline-flex;align-items:center;gap:8px;padding:6px 10px;
                    border:1px solid #e5e7eb;border-radius:999px;background:#fafafa;margin-right:8px;">
          <span style="font-weight:700">{lbl}</span>
          <span style="opacity:.9">{txt if txt else '—'}</span>
        </div>"""
    chips = []
    if run_id:
        chips.append(_chip(f"Run {run_id}", run_txt))
    if plaque_id:
        chips.append(_chip(f"Plaque {plaque_id}", pl_txt))
    return f"""<div style="margin:6px 0 10px;">{''.join(chips)}</div>"""

def apply_comment_presets(df_subset: pd.DataFrame, selected_ids: list[str], preset_text: str) -> pd.DataFrame:
    """
    Ajoute le texte 'preset_text' à la colonne 'commentaire' des lignes dont sample_id est dans selected_ids.
    Ne renomme rien. Retourne une *copie* de df_subset modifiée.
    """
    if df_subset is None or df_subset.empty or not selected_ids:
        return df_subset
    out = df_subset.copy()
    mask = out["sample_id"].astype(str).isin([str(s) for s in selected_ids])
    base = out.loc[mask, "commentaire"].astype(str).fillna("")
    sep = np.where(base.str.strip().eq(""), "", " | ")
    out.loc[mask, "commentaire"] = base + sep + preset_text
    return out
def _percent_series(s):
    s = s.astype(str).str.replace("%","",regex=False).str.replace(",",".",regex=False).str.strip()
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() and v.quantile(0.95) <= 1.05:
        v = v*100.0
    return v

def compute_temoin_stats_cards(df_temoin: pd.DataFrame, temoin: str, seuil: float = 90.0, x_col: str = "plaque_id"):
    sub = df_temoin[df_temoin["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty:
        return None, None

    sub["S4"] = _percent_series(sub["summary_consensus_perccoverage_S4"])
    sub["S6"] = _percent_series(sub["summary_consensus_perccoverage_S6"])
    sub["lot_actif"] = extract_lot_for_temoin(sub, temoin)
    sub["lot_lbl"] = sub["lot_actif"].apply(lambda v: "Sans lot" if (pd.isna(v) or str(v).strip()=="") else str(v))

    def kpis(d):
        n = len(d)
        below = d[["S4","S6"]].lt(seuil).any(axis=1).sum()
        return {
            "n": n,
            "s4": float(d["S4"].mean()) if n else float("nan"),
            "s6": float(d["S6"].mean()) if n else float("nan"),
            "below": int(below),
            "below_pct": round(100.0*below/n,1) if n else 0.0
        }

    kpi_global = kpis(sub)
    kpi_lots = (sub.groupby("lot_lbl", dropna=False)[["S4","S6"]]
                  .agg(["mean","count"]))
    # petit DF prêt à afficher
    cards = []
    for lot, d in sub.groupby("lot_lbl", dropna=False):
        m = kpis(d)
        cards.append({"lot": lot, **m})

    return kpi_global, cards
def render_temoin_stats_compact(kpi_global: dict, cards: list, temoin: str, seuil: float = 90.0, max_inline_lots: int = 6, visible_lots: int = 2):
    """
    Affichage 'style exemple':
      - Ligne globale (n, # sous seuil, seuil, moyennes S4/S6)
      - Liste par lot (n, # sous seuil, moyennes)
      - Seuls les 'visible_lots' premiers sont visibles; le reste dans un expander.
    """
    import math
    import pandas as pd
    import streamlit as st

    if not isinstance(kpi_global, dict) or not kpi_global:
        return

    # --- 1) LIGNE GLOBALE ---
    n      = int(kpi_global.get("n") or 0)
    below  = int(kpi_global.get("below") or 0)
    s4_moy = kpi_global.get("s4", float("nan"))
    s6_moy = kpi_global.get("s6", float("nan"))
    try:  s4_moy = float(s4_moy) if s4_moy is not None else float("nan")
    except: s4_moy = float("nan")
    try:  s6_moy = float(s6_moy) if s6_moy is not None else float("nan")
    except: s6_moy = float("nan")

    st.write(
        f"🧮 **Statistiques globales** pour {temoin} : "
        f"📊 {n} échantillons | "
        f"📉 {below} sous seuil (< {seuil}%) | "
        f"📈 Moyennes — S4 : {s4_moy:.1f}% · S6 : {s6_moy:.1f}%"
    )

    # --- 2) LISTE PAR LOT ---
    if not cards:
        st.write("📋 **Statistiques par lot actif** : _aucune donnée_.")
        return

    df = pd.DataFrame(cards)

    # Normalisation douce des colonnes attendues
    ren = {}
    if "lot" not in df.columns:
        for c in ["Lot", "lot_actif", "batch", "Batch"]:
            if c in df.columns: ren[c] = "lot"; break
    if "n" not in df.columns:
        for c in ["count", "N", "total_samples"]:
            if c in df.columns: ren[c] = "n"; break
    if "below" not in df.columns:
        for c in ["below_threshold", "under", "nb_sous_seuil"]:
            if c in df.columns: ren[c] = "below"; break
    if "s4" not in df.columns:
        for c in ["S4", "s4_mean", "s4_moy"]:
            if c in df.columns: ren[c] = "s4"; break
    if "s6" not in df.columns:
        for c in ["S6", "s6_mean", "s6_moy"]:
            if c in df.columns: ren[c] = "s6"; break
    if ren:
        df = df.rename(columns=ren)

    for c in ["n", "below", "s4", "s6", "below_pct"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Tri : on respecte un éventuel ordre fourni; sinon “pire en premier”
    order_col = next((c for c in ["order","ord","first_idx","first_seen","first_seen_idx"] if c in df.columns), None)
    if order_col:
        df = df.sort_values(by=order_col, ascending=False, kind="mergesort")
    elif "below_pct" in df.columns:
        df = df.sort_values(by=["below_pct","n"], ascending=[False, False], kind="mergesort")
    else:
        df = df.sort_values(by=["n"], ascending=False, kind="mergesort")

    # Construit les lignes Markdown une fois, pour réutiliser en visible + expander
    lines = []
    for _, r in df.iterrows():
        lot  = r.get("lot", None)
        nlt  = r.get("n", float("nan"))
        blt  = r.get("below", float("nan"))
        s4l  = r.get("s4", float("nan"))
        s6l  = r.get("s6", float("nan"))

        lot_lbl = "Sans lot" if (pd.isna(lot) or str(lot).strip()=="") else f"**{lot}**"
        n_txt   = "?" if pd.isna(nlt) else str(int(nlt))
        b_txt   = "?" if pd.isna(blt) else str(int(blt))
        try:  s4_txt = f"{float(s4l):.1f}%"
        except: s4_txt = "NA"
        try:  s6_txt = f"{float(s6l):.1f}%"
        except: s6_txt = "NA"

        lines.append(f"- Lot {lot_lbl} : {n_txt} échantillons | {b_txt} sous seuil | Moyenne S4 : {s4_txt} · S6 : {s6_txt}")

    st.write("📋 **Statistiques par lot actif** :")

    # Affiche les 'visible_lots' premiers
    head = lines[:max(0, int(visible_lots))]
    if head:
        st.markdown("\n".join(head))

    # Le reste dans un expander
    tail = lines[len(head):]
    if tail:
        with st.expander(f"Voir tous les lots (+{len(tail)})"):
            st.markdown("\n".join(tail))

def render_temoin_stats_header(kpi_global: dict, cards: list, temoin: str, seuil: float = 90.0):
    if not kpi_global: 
        return
    import streamlit as st
    st.markdown(f"#### 🧪 Statistiques témoins — {temoin}")

    # ligne globale
    g = kpi_global
    st.markdown(
        f"""
        <div style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:8px">
            <div style="background:#f5f7fa;border:1px solid #e5e7eb;border-radius:12px;padding:10px 14px">
              <b>Global</b> — n={g['n']} • Sous seuil({seuil}%): <b>{g['below']}</b> ({g['below_pct']}%) • 
              S4 moy: <b>{g['s4']:.1f}%</b> • S6 moy: <b>{g['s6']:.1f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # chips par lot
    if cards:
        chips = []
        for c in cards:
            chips.append(
                f"""<div style="background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:8px 12px">
                    <b>Lot {c['lot']}</b> — n={c['n']} • Sous seuil: <b>{c['below']}</b> ({c['below_pct']}%) •
                    S4: <b>{c['s4']:.1f}%</b> • S6: <b>{c['s6']:.1f}%</b>
                </div>"""
            )
        st.markdown(f"<div style='display:flex;gap:8px;flex-wrap:wrap'>{''.join(chips)}</div>", unsafe_allow_html=True)


    
def render_comment_feed(df_filtered: pd.DataFrame, limit: int = 20):
    import streamlit as st
    if df_filtered is None or df_filtered.empty or "commentaire" not in df_filtered.columns:
        return
    df = df_filtered.copy()
    df["commentaire"] = df["commentaire"].astype(str).str.strip()
    df = df[df["commentaire"] != ""]
    if df.empty:
        st.info("Aucun commentaire sur la sélection en cours.")
        return
    # tri “temps” si dispo
    order_cols = [c for c in ["date_heure","summary_run_id","plaque_id","sample_id"] if c in df.columns]
    if order_cols:
        df = df.sort_values(order_cols, ascending=True)
    # derniers N
    df = df.tail(limit)

    st.markdown("##### 🗒️ Derniers commentaires (sélection courante)")
    blocks = []
    for _, r in df.iterrows():
        hdr = " • ".join([str(r[c]) for c in ["summary_run_id","plaque_id","sample_id"] if c in r and pd.notna(r[c])])
        blocks.append(
            f"""<div style="border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px;margin-bottom:8px;background:#fff">
                 <div style="font-weight:600;margin-bottom:4px">{hdr}</div>
                 <div style="white-space:pre-wrap">{(str(r['commentaire']))}</div>
               </div>"""
        )
    st.markdown("\n".join(blocks), unsafe_allow_html=True)

def bulk_update_comments(
    base_df: pd.DataFrame,
    *,
    scope: str,                 # "sample" | "plaque" | "run"
    run_id: str | None = None,
    plaque_id: str | None = None,
    sample_ids: list[str] | None = None,
    comment_text: str,
    data_file: str,
    mode: str = "append",       # "append" | "replace"
    include_temoins: bool = True
) -> tuple[pd.DataFrame, int]:
    """
    Met à jour en masse la colonne 'commentaire' selon le 'scope'.
    - scope="sample"  → sample_ids requis (liste)
    - scope="plaque"  → plaque_id requis
    - scope="run"     → run_id   requis
    - mode="append"   → concatène à la fin (avec " | " si déjà présent)
      mode="replace"  → écrase l'existant
    - include_temoins → True pour inclure Tpos/NT/Tvide, False pour les exclure
    Retourne (df_maj, nb_lignes_impactées) et PERSISTE en sûr (atomic).
    """
    if not has_cols(base_df, ["plaque_id", "sample_id", "commentaire"]):
        return base_df, 0

    df = base_df.copy()
    m = pd.Series(True, index=df.index)

    # Filtrage par scope
    if scope == "sample":
        if not sample_ids:
            return base_df, 0
        m &= df["sample_id"].isin(sample_ids)
    elif scope == "plaque":
        if not plaque_id:
            return base_df, 0
        m &= (df["plaque_id"] == plaque_id)
    elif scope == "run":
        if not run_id or "summary_run_id" not in df.columns:
            return base_df, 0
        m &= (df["summary_run_id"] == run_id)
    else:
        return base_df, 0

    # Inclure/exclure témoins
    if not include_temoins:
        pat = r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide"
        m &= ~df["sample_id"].astype(str).str.contains(pat, case=False, na=False)

    idx = df.index[m]
    if len(idx) == 0:
        return base_df, 0

    comment_text = str(comment_text or "").strip()
    if not comment_text:
        return df, 0

    # Persister aussi les commentaires plaque/run pour activer 🏷️/📌 sur les graphes
    if scope == "plaque" and plaque_id:
        set_plaque_comment(plaque_id, comment_text)
    if scope == "run" and run_id:
        set_run_comment(run_id, comment_text)

    if mode == "replace":
        df.loc[idx, "commentaire"] = comment_text
    else:
        # append
        old = df.loc[idx, "commentaire"].astype(str).fillna("")
        sep = np.where(old.str.strip() == "", "", " | ")
        df.loc[idx, "commentaire"] = old + sep + comment_text

    # Persistance atomique
    tmp_path = data_file + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, data_file)

    return df, len(idx)


# --- Bulk comment (par plaque) : applique un commentaire à plusieurs lignes sans renommer les DataFrames
def apply_comment_plate(
    base_df: pd.DataFrame,
    plaque_id: str,
    comment: str,
    scope: str = "all",           # "all" | "tpos_only" | "non_temoin"
    mode: str = "append",         # "append" | "replace"
    data_file: str = "historique_data.csv"
) -> pd.DataFrame:
    """
    Applique 'comment' aux lignes de base_df pour une plaque.
    - scope="all": tous les échantillons de la plaque
    - scope="tpos_only": uniquement TposH1/H3/B
    - scope="non_temoin": tous sauf Tpos/NT/Tvide
    - mode="append": concatène au commentaire existant (séparateur ' | ')
      mode="replace": remplace entièrement
    Sauvegarde le DataFrame complet (atomique) dans data_file.
    Retourne base_df modifié (même objet passé par référence dans Streamlit).
    """
    if base_df is None or base_df.empty:
        st.warning("Aucune donnée en base pour appliquer un commentaire.")
        return base_df
    needed = ["plaque_id", "sample_id", "commentaire"]
    missing = [c for c in needed if c not in base_df.columns]
    if missing:
        st.error(f"Colonnes manquantes pour l'édition bulk: {missing}")
        return base_df

    if not isinstance(plaque_id, str) or plaque_id.strip() == "":
        st.warning("Plaque ID manquant.")
        return base_df
    if not isinstance(comment, str) or comment.strip() == "":
        st.warning("Commentaire vide : rien à appliquer.")
        return base_df

    # masque plaque
    m_pl = (base_df["plaque_id"].astype(str) == plaque_id)

    # périmètre
    name_u = base_df["sample_id"].astype(str).str.upper()
    m_tpos = name_u.str.contains(r"\bTPOSH1\b|\bTPOSH3\b|\bTPOSB\b", regex=True, na=False)
    m_nt   = name_u.str.contains(r"\bNT\s*[-_]*\s*[123]\b", regex=True, na=False)
    m_tvide= name_u.str.contains("TVIDE", na=False)

    if scope == "tpos_only":
        mask_scope = m_pl & m_tpos
    elif scope == "non_temoin":
        mask_scope = m_pl & (~m_tpos) & (~m_nt) & (~m_tvide)
    else:
        mask_scope = m_pl

    idx = base_df.index[mask_scope]
    if len(idx) == 0:
        st.info("Aucune ligne correspondante pour cette plaque et ce périmètre.")
        return base_df

    # appliquer
    if mode == "replace":
        base_df.loc[idx, "commentaire"] = comment
    else:
        # append
        cur = base_df.loc[idx, "commentaire"].astype(str).fillna("")
        sep = np.where(cur.str.strip()=="", "", " | ")
        base_df.loc[idx, "commentaire"] = cur + sep + comment

    # persist (écriture sûre)
    tmp_path = data_file + ".tmp"
    base_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, data_file)

    st.success(f"✅ Commentaire appliqué à {len(idx)} ligne(s) (plaque {plaque_id}, scope='{scope}', mode='{mode}').")
    return base_df

    with st.expander("🧰 Outil commentaires", expanded=False):
       
        # Bandeau scope + options
        c0, c1, c2, c3 = st.columns([1.2, 1.6, 1.2, 1.4])
        with c0:
            scope = st.radio("Portée", ["Échantillon", "Plaque", "Run"], horizontal=True)
        with c1:
            mode = st.selectbox("Mode", ["Ajouter à la fin", "Remplacer le commentaire"])
        with c2:
            include_temoins = st.checkbox("Inclure Tpos/NT/Tvide", value=True)
        with c3:
            show_advanced = st.checkbox("Options avancées", value=False)

        # Sélecteurs selon la portée
        target_sample_ids, target_plaque_id, target_run_id = None, None, None
        s1, s2 = st.columns([2, 2])
        with s1:
            if scope == "Run":
                runs_all = sorted(df_src.get("summary_run_id", pd.Series(dtype=str)).dropna().unique().tolist()) if "summary_run_id" in df_src.columns else []
                target_run_id = st.selectbox("Choisir un run", options=runs_all if runs_all else ["—"])
            elif scope == "Plaque":
                plaques_all2 = sorted(df_src.get("plaque_id", pd.Series(dtype=str)).dropna().unique().tolist())
                target_plaque_id = st.selectbox("Choisir une plaque", options=plaques_all2 if plaques_all2 else ["—"])
            else:
                if filtered_df.empty:
                    st.info("Aucun échantillon dans la sélection courante.")
                else:
                    target_sample_ids = st.multiselect(
                        "Choisir un ou plusieurs échantillons",
                        options=sorted(filtered_df["sample_id"].unique().tolist()),
                        default=[]
                    )
                    apply_all_filtered = st.checkbox(
                        "Appliquer à tous les échantillons filtrés (si aucun sélectionné)",
                        value=False,
                        help="Sécurité : sinon, aucune action si aucun échantillon n’est explicitement choisi."
                    )
        with s2:
            fast_filters = st.multiselect(
                "Filtre rapide (optionnel)",
                options=["Uniquement témoins", "Exclure témoins"],
                help="S’applique uniquement si Portée = Échantillon ou si tu utilises la sélection courante."
            )

        # Presets “chips” + zone libre (compact)
        st.markdown("**Composition du commentaire**")
        p1, p2, p3, p4 = st.columns([1.1, 1.2, 1.0, 1.6])
        with p1:
            preset_pb = st.toggle("PB technique", value=False)
        with p2:
            preset_nc = st.toggle("NC ouverte", value=False)
            num_nc = st.text_input("N° NC", value="", placeholder="ex: NC-2025-123") if preset_nc else ""
        with p3:
            preset_lot = st.toggle("Nouveau lot", value=False)
        with p4:
            lot_txt = st.text_input("Lot n°", value="", placeholder="ex: 2345") if preset_lot else ""

        user_comment = st.text_area("Commentaire libre", placeholder="Ajoutez un contexte…", height=80)

        # Assemble le message final
        parts = []
        if preset_pb: parts.append("PB technique")
        if preset_nc and num_nc.strip(): parts.append(f"NC {num_nc.strip()}")
        if preset_lot and lot_txt.strip(): parts.append(f"Nouveau lot {lot_txt.strip()}")
        if user_comment.strip(): parts.append(user_comment.strip())
        final_comment = " ; ".join(parts).strip()

        # Boutons d’action
        a1, a2, a3 = st.columns([1, 1, 2])
        with a1:
            btn_apply = st.button("✅ Appliquer le commentaire")
        with a2:
            reset_comment = st.button("🧹 Réinitialiser la saisie")

        if reset_comment:
            st.rerun()

        # Application commentaires (FIX)
        if btn_apply:
            if not final_comment:
                st.warning("Veuillez renseigner au moins un preset ou un texte libre.")
                st.stop()

            scope_key = "sample" if scope == "Échantillon" else ("plaque" if scope == "Plaque" else "run")
            mode_key = "append" if mode == "Ajouter à la fin" else "replace"

            resolved_sample_ids = None

            if scope_key == "sample":
                df_target = filtered_df.copy()

                # Filtres rapides sur la sélection courante
                if "Uniquement témoins" in (fast_filters or []):
                    df_target = df_target[
                        df_target["sample_id"].str.contains(
                            r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide", case=False, na=False
                        )
                    ]
                if "Exclure témoins" in (fast_filters or []):
                    df_target = df_target[
                        ~df_target["sample_id"].str.contains(
                            r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide", case=False, na=False
                        )
                    ]

                # 1) Sélection explicite prioritaire
                if target_sample_ids and len(target_sample_ids) > 0:
                    resolved_sample_ids = target_sample_ids

                # 2) Sinon, appliquer à tous les échantillons filtrés si la case est cochée
                elif apply_all_filtered:
                    resolved_sample_ids = df_target["sample_id"].dropna().unique().tolist()

                # 3) Sinon, on bloque proprement (au lieu d’un warning + passage en try)
                else:
                    st.warning("Sélectionnez au moins un échantillon ou cochez « Appliquer à tous les échantillons filtrés ». Aucun changement effectué.")
                    st.stop()

                # Sécurité : si “Nouveau lot”, restreindre aux TposH1/H3/B et bloquer si vide
                if preset_lot and lot_txt.strip():
                    mask_tpos = df_target["sample_id"].astype(str).str.contains(r"TposH1|TposH3|TposB", case=False, na=False)
                    tpos_ids = df_target.loc[mask_tpos, "sample_id"].dropna().unique().tolist()

                    if target_sample_ids and len(target_sample_ids) > 0:
                        resolved_sample_ids = [s for s in resolved_sample_ids if any(t in s for t in ["TposH1", "TposH3", "TposB"])]
                    else:
                        resolved_sample_ids = tpos_ids

                    if not resolved_sample_ids:
                        st.warning("⚠️ « Nouveau lot » demandé mais aucun témoin TposH1/H3/B ciblé. Sélectionne au moins un témoin.")
                        st.stop()

            # Petit récap utile (non bloquant)
            if scope_key == "sample":
                st.info(f"Échantillons ciblés : {len(resolved_sample_ids)}")

            try:
                base_df, nb = definitions_flu.bulk_update_comments(
                    base_df,
                    scope=scope_key,
                    run_id=target_run_id if scope_key == "run" else None,
                    plaque_id=target_plaque_id if scope_key == "plaque" else None,
                    sample_ids=resolved_sample_ids if scope_key == "sample" else None,
                    comment_text=final_comment,
                    data_file=DATA_FILE,
                    mode=mode_key,
                    include_temoins=include_temoins
                )
                st.success(f"✅ Commentaire appliqué à {nb} ligne(s).")
            except Exception as e:
                st.error(f"Erreur mise à jour : {e}")

                # 🔄 rafraîchir la vue filtrée
                filtered_df = base_df.copy()
                if rid != "(tous)" and "summary_run_id" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["summary_run_id"] == rid]
                if pl_choice != "(toutes)" and "plaque_id" in filtered_df.columns:
                    filtered_df = filtered_df[filtered_df["plaque_id"] == pl_choice]

            except Exception as e:
                st.error(f"Erreur mise à jour : {e}")
                
def _lot_badge_points(df_stage: pd.DataFrame, xorder_col: str = "plaque_id"):
    if df_stage is None or df_stage.empty:
        return []
    cats = pd.Categorical(df_stage[xorder_col].astype(str))
    order = list(cats.categories) or df_stage[xorder_col].astype(str).unique().tolist()
    ord_map = {c: i for i, c in enumerate(order)}
    out = []
    for lot, sub in df_stage.groupby("lot_id", dropna=True):
        sub = sub.sort_values(xorder_col, key=lambda s: s.astype(str).map(ord_map.get))
        cur = sub.loc[sub["stage"].astype(str).str.lower().eq("courant")]
        if cur.empty:
            continue
        first_cur = str(cur.iloc[0][xorder_col])
        prob = sub.loc[sub["stage"].astype(str).str.lower().eq("probatoire"), xorder_col].astype(str)
        has_prob_before = any(ord_map.get(p, 10**9) < ord_map.get(first_cur, -1) for p in prob)
        out.append((first_cur, str(lot), bool(has_prob_before)))
    return out


def add_comment_badges_on_fig(fig, df_temoin, temoin, x_col: str = "plaque_id"):
    """
    Overlays (graphiques Tpos) :
      🔁 : changement de lot Témoin (Tpos)
      🧪 : 'Autre nouveau lot (kit, amorces...)'
      🏷️ : commentaire PLAQUE
      📌 : commentaire RUN
      💬 : commentaire ÉCHANTILLON (hors 'nouveau lot Témoin' & 'autre nouveau lot' & miroirs)
    """

    ICON_SAMPLE, ICON_LOT, ICON_PLQ, ICON_RUN, ICON_KIT = "💬", "🔁", "🏷️", "📌", "🧪"
    LOT_REGEX_TPOS = re.compile(
        r"(?:nouveau|nouvelle)\s*lot|lot\s*en\s*cours\s+\S+|lot\s*clos\s+\S+",
        flags=re.I
    )
    KIT_REGEX = re.compile(r"autre\s+nouveau\s+lot", flags=re.I)


    # 1) Filtrer le témoin (TposH1/H3/B)
    sub = df_temoin[df_temoin["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty:
        return fig, None

    # 2) Axe X robuste
    if x_col in sub.columns:              xorder = x_col
    elif "date_heure" in sub.columns:     xorder = "date_heure"
    elif "summary_run_id" in sub.columns: xorder = "summary_run_id"
    else:                                 xorder = "plaque_id" if "plaque_id" in sub.columns else "sample_id"
    sub.sort_values([xorder, "sample_id"], inplace=True)
    cats = sub[xorder].astype(str).drop_duplicates().tolist()

    # 3) Séries & headroom
    sub["S4"] = _percent_series(sub.get("summary_consensus_perccoverage_S4"))
    sub["S6"] = _percent_series(sub.get("summary_consensus_perccoverage_S6"))
    y_max_by_x = sub[["S4","S6"]].max(axis=1).groupby(sub[xorder]).max()

    def _y_for_x(x, off=0.0):
        return float(y_max_by_x.get(x, 100.0)) + float(off)

    try:
        current_top = float(fig.layout.yaxis.range[1]) if fig.layout.yaxis and fig.layout.yaxis.range else None
    except Exception:
        current_top = None
    data_peak = float(max(y_max_by_x.max(), 100.0)) if len(y_max_by_x) else 100.0
    y_top = max(140.0, (current_top or 0.0), data_peak + 28.0)
    fig.update_yaxes(range=[0, y_top])
    fig.update_layout(
        margin=dict(
            l=getattr(fig.layout.margin, "l", 40),
            r=getattr(fig.layout.margin, "r", 30),
            t=max(95, getattr(fig.layout.margin, "t", 80)),
            b=max(90, getattr(fig.layout.margin, "b", 80)),
        ),
        hovermode="closest",
        hoverdistance=10
    )

    # Bandes Y
    Y_RUN, Y_PLQ, Y_LOT, Y_KIT, Y_SAMPLE = y_top-2.0, y_top-8.0, y_top-14.0, y_top-20.0, y_top-26.0

    # 4) Maps plaque/run (texte)
    plaque_map, run_map = {}, {}
    if "plaque_id" in sub.columns:
        for pid in sub["plaque_id"].dropna().astype(str).unique():
            txt = get_plaque_comment(pid)
            if str(txt or "").strip():
                plaque_map[pid] = str(txt).strip()

    if "summary_run_id" in sub.columns:
        for rid in sub["summary_run_id"].dropna().astype(str).unique():
            txt = get_run_comment(rid)
            if str(txt or "").strip():
                run_map[rid] = str(txt).strip()

    # Helpers
    def _contains(a, b):
        a = str(a or "").strip().lower()
        b = str(b or "").strip().lower()
        return (a != "") and (a in b)

    # 5) 🔁 Nouveau lot témoin — calcule les positions des badges
    try:
        df_stage = build_lot_stage_df_for_hist(sub)  # 'sub' = df_temoin filtré au-dessus
        if df_stage is not None and not df_stage.empty:
            df_stage["plaque_id"] = df_stage["plaque_id"].astype(str)
            # Calcule les points candidats (promotion ou début de lot courant)
            pts = _lot_badge_points(df_stage.rename(columns={"plaque_id": xorder}), xorder_col=xorder)

            if pts:
                xs_lot  = [str(p[0]) for p in pts]
                hover_l = [
                    (
                        "Promotion du lot" if p[2] else "Début de lot (courant immédiat)"
                    ) + f" {p[1]}"
                    for p in pts
                ]
                fig.add_trace(go.Scatter(
                    x=xs_lot,
                    y=[min(Y_LOT, _y_for_x(x, 10)) for x in xs_lot],
                    mode="markers+text",
                    marker=dict(size=20, opacity=0),
                    text=["🔁"] * len(xs_lot),
                    textposition="middle center",
                    textfont=dict(size=18),
                    hoverinfo="text",
                    hovertext=hover_l,
                    cliponaxis=False, showlegend=False, name="Promotion lot Témoin"
                ))
    except Exception:
        # on ne casse pas l’affichage du graphe si l’extraction échoue
        pass

    # 6) 💬 / 🧪 ÉCHANTILLONS (exclut miroirs + 'nouveau lot Témoin' + 'autre nouveau lot')
    sub["__c__"] = sub.get("commentaire", "").astype(str).fillna("").str.strip()
    is_new_tpos = sub["__c__"].str.contains(LOT_REGEX_TPOS, na=False)
    is_kit      = sub["__c__"].str.contains(KIT_REGEX, na=False)

    pl_txt_row  = sub.get("plaque_id", pd.Series(index=sub.index, dtype=object)).astype(str).map(plaque_map).fillna("")
    is_pl_mirror  = [_contains(p, c) for p, c in zip(pl_txt_row.tolist(), sub["__c__"].tolist())]
    run_txt_row = sub.get("summary_run_id", pd.Series(index=sub.index, dtype=object)).astype(str).map(run_map).fillna("")
    is_run_mirror = [_contains(r, c) for r, c in zip(run_txt_row.tolist(), sub["__c__"].tolist())]

    # 🧪 — un marqueur par abscisse qui contient le tag 'autre nouveau lot' (sans doublon si déjà au niveau PLAQUE)
    m_kit = (sub["__c__"] != "") & is_kit
    if m_kit.any():
        cols = [xorder, "sample_id", "__c__", "S4", "S6"]
        if "plaque_id" in sub.columns and xorder != "plaque_id":
            cols.append("plaque_id")
        df_kit = sub.loc[m_kit, cols].copy()

        for xval, bloc in df_kit.groupby(xorder):
            if "plaque_id" in bloc.columns:
                pl_ids = bloc["plaque_id"].dropna().astype(str).unique().tolist()
            else:
                pl_ids = [str(xval)]
            has_plaque_kit = any(KIT_REGEX.search(str(plaque_map.get(pid, "")) or "") for pid in pl_ids)
            if has_plaque_kit:
                continue

            hover = "<br>".join([f"{sid}: {txt[:220]}" for sid, txt in bloc[["sample_id","__c__"]].astype(str).values][:8])
            y_here = min(Y_KIT, float(max(bloc["S4"].max(), bloc["S6"].max()) or 0.0) + 6.0)
            fig.add_trace(go.Scatter(
                x=[str(xval)], y=[y_here],
                mode="markers+text",
                marker=dict(size=20, opacity=0),
                text=[ICON_KIT], textposition="middle center",
                textfont=dict(size=18),
                hoverinfo="text", hovertext=[f"Autre nouveau lot (kit/amorces)<br>{hover}"],
                cliponaxis=False, showlegend=False, name="Autre nouveau lot (kit/amorces)"
            ))

    # 💬 — on exclut 'nouveau lot Témoin', 'autre nouveau lot' et les miroirs
    mask_sample = (sub["__c__"]!="") & (~is_new_tpos) & (~is_kit) & (~pd.Series(is_pl_mirror, index=sub.index)) & (~pd.Series(is_run_mirror, index=sub.index))
    if mask_sample.any():
        for xval, bloc in sub.loc[mask_sample, [xorder,"sample_id","__c__","S4","S6"]].groupby(xorder):
            hover = "<br>".join([f"{sid}: {txt[:220]}" for sid, txt in bloc[["sample_id","__c__"]].astype(str).values][:8])
            y_here = min(Y_SAMPLE, float(max(bloc["S4"].max(), bloc["S6"].max()) or 0.0) + 4.0)
            fig.add_trace(go.Scatter(
                x=[str(xval)], y=[y_here],
                mode="markers+text",
                marker=dict(size=18, opacity=0),
                text=[ICON_SAMPLE], textposition="middle center",
                textfont=dict(size=16),
                hoverinfo="text", hovertext=[hover],
                cliponaxis=False, showlegend=False, name="Commentaire échantillon"
            ))

    # 7) 🏷️ / 🧪 Plaque — 🧪 prend le pas
    if "plaque_id" in sub.columns and plaque_map:
        xs_plq, ys_plq, h_plq = [], [], []
        xs_kit_plq, ys_kit_plq, h_kit_plq = [], [], []

        for pid, ptxt in plaque_map.items():
            xs_pid = sub.loc[sub["plaque_id"].astype(str)==pid, xorder].dropna().astype(str)
            if xs_pid.empty:
                continue
            x0 = xs_pid.iloc[0]
            if KIT_REGEX.search(ptxt or ""):
                xs_kit_plq.append(str(x0))
                ys_kit_plq.append(min(Y_KIT, _y_for_x(x0, 12)))
                h_kit_plq.append(f"Plaque {pid}<br>Autre nouveau lot (kit/amorces): {ptxt}")
            else:
                xs_plq.append(str(x0))
                ys_plq.append(min(Y_PLQ, _y_for_x(x0, 14)))
                h_plq.append(f"Plaque {pid}<br>Commentaire (plaque): {ptxt}")

        if xs_plq:
            fig.add_trace(go.Scatter(
                x=xs_plq, y=ys_plq, mode="markers+text",
                marker=dict(size=20, opacity=0),
                text=[ICON_PLQ]*len(xs_plq), textposition="middle center",
                textfont=dict(size=18),
                hoverinfo="text", hovertext=h_plq,
                cliponaxis=False, showlegend=False, name="Commentaire plaque"
            ))
        if xs_kit_plq:
            fig.add_trace(go.Scatter(
                x=xs_kit_plq, y=ys_kit_plq, mode="markers+text",
                marker=dict(size=20, opacity=0),
                text=[ICON_KIT]*len(xs_kit_plq), textposition="middle center",
                textfont=dict(size=18),
                hoverinfo="text", hovertext=h_kit_plq,
                cliponaxis=False, showlegend=False, name="Autre nouveau lot (plaque)"
            ))

    # 8) 📌 / 🧪 RUN — 🧪 prend le pas (si commentaire run contient le tag, pas de 📌)
    if run_map:
        order_xs = cats[:]
        used_pin, used_kit = {}, {}
        for rid, rtxt in sorted(run_map.items(), key=lambda kv: kv[0]):
            xs_run = sub.loc[sub.get("summary_run_id","").astype(str)==rid, xorder].dropna().astype(str).drop_duplicates().tolist()
            x_mid = xs_run[len(xs_run)//2] if xs_run else (order_xs[len(order_xs)//2] if order_xs else None)
            if x_mid is None:
                continue

            if KIT_REGEX.search(rtxt or ""):
                k = str(x_mid); m = used_kit.get(k, 0); used_kit[k] = m + 1
                y_k = max(Y_KIT - m*3.0, y_top - 30.0)
                fig.add_trace(go.Scatter(
                    x=[str(x_mid)], y=[y_k],
                    mode="markers+text",
                    marker=dict(size=20, opacity=0),
                    text=[ICON_KIT], textposition="middle center",
                    textfont=dict(size=18),
                    hoverinfo="text", hovertext=[f"Run {rid}<br>Autre nouveau lot (kit/amorces): {rtxt}"],
                    cliponaxis=False, showlegend=False, name="Autre nouveau lot (run)"
                ))
            else:
                k = str(x_mid); n = used_pin.get(k, 0); used_pin[k] = n + 1
                y_pin = max(Y_RUN - n*3.0, y_top - 30.0)
                fig.add_trace(go.Scatter(
                    x=[str(x_mid)], y=[y_pin],
                    mode="markers+text",
                    marker=dict(size=20, opacity=0),
                    text=[ICON_RUN], textposition="middle center",
                    textfont=dict(size=18),
                    hoverinfo="text", hovertext=[f"Run {rid}<br>Commentaire (run): {rtxt}"],
                    cliponaxis=False, showlegend=False, name="Commentaire run"
                ))

    fig.update_xaxes(categoryorder="array", categoryarray=cats)
    return fig, sub[[xorder,"sample_id","__c__"]].rename(columns={"__c__":"commentaire"})


def add_run_plaque_flags_on_fig(
    fig,
    aggregated_by_plaque_df: pd.DataFrame,
    all_samples_df: pd.DataFrame = None,
    x_col: str = "plaque_id",
    run_id: str | None = None,
    lot_stage_df: pd.DataFrame | None = None
):
    if aggregated_by_plaque_df is None or aggregated_by_plaque_df.empty or x_col not in aggregated_by_plaque_df.columns:
        return fig

    xs = aggregated_by_plaque_df[x_col].astype(str).dropna().tolist()
    if not xs:
        return fig

    ICON_SAMPLE, ICON_LOT, ICON_PLQ, ICON_RUN, ICON_KIT = "💬", "🔁", "🏷️", "📌", "🧪"
    LOT_REGEX_TPOS = re.compile(r"(?:nouveau|nouvelle)\s*lot|lot\s*en\s*cours\s+\S+|lot\s*clos\s+\S+", flags=re.I)
    KIT_REGEX      = re.compile(r"autre\s+nouveau\s+lot", flags=re.I)

    # Pic pour le headroom
    if "pct_ininterpretable" in aggregated_by_plaque_df.columns:
        peak = float(aggregated_by_plaque_df["pct_ininterpretable"].max())
    elif "% varcount >= 13" in aggregated_by_plaque_df.columns:
        peak = float(aggregated_by_plaque_df["% varcount >= 13"].max())
    else:
        peak = 100.0
    y_top = _ensure_headroom(fig, peak)
    Y_RUN, Y_PLQ, Y_LOT, Y_KIT, Y_SAMPLE = y_top-2, y_top-8, y_top-14, y_top-20, y_top-26

    # 🏷️/🧪—PLAQUE
    xs_plq, hover_plq, xs_kit_plq, hover_kit_plq, plaque_map = [], [], [], [], {}
    for pid in xs:
        txt = get_plaque_comment(pid)
        if str(txt or "").strip():
            plaque_map[pid] = str(txt).strip()
            if KIT_REGEX.search(plaque_map[pid] or ""):
                xs_kit_plq.append(pid); hover_kit_plq.append(f"Plaque {pid}<br>Autre nouveau lot (kit/amorces): {plaque_map[pid]}")
            else:
                xs_plq.append(pid);     hover_plq.append(f"Plaque {pid}<br>Commentaire (plaque): {plaque_map[pid]}")

    # 📌/🧪—RUN (1 badge par run présent sur l’intervalle affiché)
    pins_run, pins_kit = [], []
    if (all_samples_df is not None) and {"summary_run_id","plaque_id"}.issubset(all_samples_df.columns):
        seen = all_samples_df[all_samples_df["plaque_id"].astype(str).isin(xs)]
        run_comments = {}
        for rid in seen["summary_run_id"].dropna().astype(str).unique():
            txt = get_run_comment(rid)
            if str(txt or "").strip():
                run_comments[rid] = str(txt).strip()
        if run_id:
            rid_norm = str(run_id).strip()
            if rid_norm and (rid_norm not in run_comments):
                txt = get_run_comment(rid_norm)
                if str(txt or "").strip():
                    run_comments[rid_norm] = str(txt).strip()
        usedR, usedK = {}, {}
        for rid, rtxt in sorted(run_comments.items(), key=lambda kv: kv[0]):
            pls = seen.loc[seen["summary_run_id"].astype(str)==rid, "plaque_id"].astype(str)
            pls = [p for p in xs if p in set(pls)]
            if not pls: 
                continue
            x_mid = pls[len(pls)//2]
            if KIT_REGEX.search(rtxt or ""):
                n = usedK.get(x_mid, 0); usedK[x_mid] = n+1
                pins_kit.append((x_mid, max(Y_KIT - n*3.0, y_top - 32.0), f"Run {rid}<br>Autre nouveau lot (kit/amorces): {rtxt}"))
            else:
                n = usedR.get(x_mid, 0); usedR[x_mid] = n+1
                pins_run.append((x_mid, max(Y_RUN - n*3.0, y_top - 32.0), f"Run {rid}<br>Commentaire (run): {rtxt}"))

    # 🔁/🧪/💬—ÉCHANTILLON (par plaque)
    if all_samples_df is not None and {"plaque_id","sample_id","commentaire"}.issubset(all_samples_df.columns):
        df_all = all_samples_df.copy()
        df_all["__x__"] = df_all["plaque_id"].astype(str)
        df_all["__c__"] = df_all["commentaire"].astype(str).fillna("").str.strip()

        # miroirs pour éviter les doublons de badges
        df_all["__is_plq_mirror__"] = df_all.apply(lambda r: (str(plaque_map.get(str(r["__x__"]), "")).strip().lower() in str(r["__c__"]).strip().lower()) if str(plaque_map.get(str(r["__x__"]), "")).strip()!="" else False, axis=1)
        rid2txt = {rid: get_run_comment(rid) for rid in df_all.get("summary_run_id","").astype(str).unique() if str(get_run_comment(rid) or "").strip()}
        df_all["__rid__"] = df_all.get("summary_run_id","").astype(str)
        df_all["__is_run_mirror__"] = df_all.apply(lambda r: (str(rid2txt.get(r["__rid__"], "")).strip().lower() in str(r["__c__"]).strip().lower()) if str(rid2txt.get(r["__rid__"], "")).strip()!="" else False, axis=1)

        is_tpos = df_all["sample_id"].astype(str).str.contains(r"Tpos(H1|H3|B)", case=False, regex=True, na=False)
        is_new_generic = df_all["__c__"].str.contains(LOT_REGEX_TPOS, na=False)
        is_kitS = df_all["__c__"].str.contains(KIT_REGEX, na=False)
        is_new_tpos = is_new_generic & (~is_kitS)

        # 💬 échantillons (on exclut nouveau lot/kit + miroirs)
        m_sample = (df_all["__c__"]!="") & (~is_new_generic) & (~is_kitS) & (~df_all["__is_plq_mirror__"]) & (~df_all["__is_run_mirror__"]) & df_all["__x__"].isin(xs)
        if m_sample.any():
            xs_s, hover_s = [], []
            for pid, g in df_all[m_sample].groupby("__x__"):
                rows = [f"{str(r.get('sample_id',''))}: {str(r['__c__'])[:220]}" for _, r in g.sort_values(["__x__","sample_id"]).head(6).iterrows()]
                if len(g) > 6: rows.append(f"… (+{len(g)-6} autres)")
                xs_s.append(pid); hover_s.append("<br>".join(rows) if rows else "≥1 échantillon commenté")
            fig.add_trace(go.Scatter(x=xs_s, y=[Y_SAMPLE]*len(xs_s), mode="markers+text",
                                     marker=dict(size=18, opacity=0), text=["💬"]*len(xs_s),
                                     textposition="middle center", textfont=dict(size=16),
                                     hoverinfo="text", hovertext=hover_s, cliponaxis=False, showlegend=False, name="Commentaires échantillons"))

    # 🔁 issu de l’état des lots (source de vérité)
    if lot_stage_df is not None and not lot_stage_df.empty:
        pts = _lot_badge_points(lot_stage_df.rename(columns={"plaque_id": x_col}), xorder_col=x_col)
        if pts:
            x_badges = [str(p[0]) for p in pts if str(p[0]) in set(xs)]
            y_badges = [min(Y_LOT, y_top - 16.0)] * len(x_badges)
            hover = [f"Nouveau lot Témoin (lot {p[1]})" + (" — promotion depuis probatoire" if p[2] else "") for p in pts if str(p[0]) in set(xs)]
            if x_badges:
                fig.add_trace(go.Scatter(x=x_badges, y=y_badges, mode="markers+text",
                                         marker=dict(size=20, opacity=0), text=["🔁"]*len(x_badges),
                                         textposition="middle center", textfont=dict(size=18),
                                         hoverinfo="text", hovertext=hover, cliponaxis=False, showlegend=False, name="Nouveau lot Témoin"))

    # 🏷️/🧪 plaque
    if xs_plq:
        fig.add_trace(go.Scatter(x=xs_plq, y=[Y_PLQ]*len(xs_plq), mode="markers+text",
                                 marker=dict(size=20, opacity=0), text=["🏷️"]*len(xs_plq),
                                 textposition="middle center", textfont=dict(size=18),
                                 hoverinfo="text", hovertext=hover_plq, cliponaxis=False, showlegend=False, name="Commentaire plaque"))
    if xs_kit_plq:
        fig.add_trace(go.Scatter(x=xs_kit_plq, y=[Y_KIT]*len(xs_kit_plq), mode="markers+text",
                                 marker=dict(size=20, opacity=0), text=["🧪"]*len(xs_kit_plq),
                                 textposition="middle center", textfont=dict(size=18),
                                 hoverinfo="text", hovertext=hover_kit_plq, cliponaxis=False, showlegend=False, name="Autre nouveau lot (plaque)"))
    # 📌/🧪 run
    for x_mid, y_pin, h in pins_run:
        fig.add_trace(go.Scatter(x=[x_mid], y=[y_pin], mode="markers+text",
                                 marker=dict(size=20, opacity=0), text=["📌"], textposition="middle center",
                                 textfont=dict(size=18), hoverinfo="text", hovertext=[h], cliponaxis=False, showlegend=False, name="Commentaire run"))
    for x_mid, y_kit, h in pins_kit:
        fig.add_trace(go.Scatter(x=[x_mid], y=[y_kit], mode="markers+text",
                                 marker=dict(size=20, opacity=0), text=["🧪"], textposition="middle center",
                                 textfont=dict(size=18), hoverinfo="text", hovertext=[h], cliponaxis=False, showlegend=False, name="Autre nouveau lot (run)"))

    fig.update_xaxes(categoryorder="array", categoryarray=xs)
    return fig



# === UI minimaliste : cartes & chips pour Tpos =================================
def inject_local_css():
    st.markdown("""
    <style>
      .kpi-card{display:flex;gap:14px;align-items:center;padding:10px 12px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;}
      .kpi-pill{display:inline-flex;gap:8px;align-items:center;padding:6px 10px;border-radius:999px;border:1px solid #e5e7eb;background:#fff;}
      .kpi-dot{width:10px;height:10px;border-radius:999px;background:#10b981;display:inline-block}
      .kpi-dot.warn{background:#ef4444;}
      .tpos-ribbon{border:1px solid #e5e7eb;border-radius:12px;padding:10px 12px;margin:6px 0 10px;background:#fff;}
      .chip{display:inline-flex;align-items:center;gap:10px;padding:8px 12px;border-radius:999px;margin:6px 8px 0 0;border:2px solid #10b981;background:#ecfdf5;}
      .chip.warn{border-color:#e11d48;background:#ffecec;}
      .chip b{font-weight:700}
    </style>
    """, unsafe_allow_html=True)

def _pct_fmt(v):
    try: return f"{float(v):.1f}%"
    except: return "—"

def tpos_stats_ribbon(df_temoin: pd.DataFrame, temoin: str, seuil: float=90) -> str:
    """Construit un ruban HTML : 1 ligne globale + chips par lot."""
    if df_temoin is None or df_temoin.empty: return ""
    sub = df_temoin[df_temoin["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
    if sub.empty: return ""
    def _to_pct(s):
        s = s.astype(str).str.replace("%","",regex=False).str.replace(",",".",regex=False)
        x = pd.to_numeric(s, errors="coerce")
        if x.notna().sum() and x.quantile(0.95) <= 1.05: x = x*100
        return x
    sub["S4"] = _to_pct(sub.get("summary_consensus_perccoverage_S4"))
    sub["S6"] = _to_pct(sub.get("summary_consensus_perccoverage_S6"))
    sub["lot_actif"] = extract_lot_for_temoin(sub, temoin)
    sub["lot_lbl"] = sub["lot_actif"].apply(lambda v: "Sans lot" if (pd.isna(v) or str(v).strip()=="") else str(v))
    sub["sous_seuil"] = sub[["S4","S6"]].lt(seuil).any(axis=1)

    n_all = len(sub); s4m = sub["S4"].mean(); s6m = sub["S6"].mean()
    pct_bad = 100.0 * sub["sous_seuil"].sum() / max(1,n_all)
    chips = []
    by_lot = sub.groupby("lot_lbl", dropna=False).agg(n=("sample_id","count"), s4=("S4","mean"), s6=("S6","mean"), bad=("sous_seuil","sum")).reset_index()
    by_lot["pct_bad"] = (100.0*by_lot["bad"]/by_lot["n"]).round(1)
    for _, r in by_lot.sort_values("lot_lbl").iterrows():
        warn = (r["pct_bad"] > 0)
        chips.append(
            f"<div class='chip{' warn' if warn else ''}'><div><b>Lot {r['lot_lbl']}</b></div>"
            f"<div>n={int(r['n'])}</div><div>S4: <b>{_pct_fmt(r['s4'])}</b></div>"
            f"<div>S6: <b>{_pct_fmt(r['s6'])}</b></div><div>{'⚠️' if warn else '✅'} {r['pct_bad']:.1f}% sous seuil</div></div>"
        )
    return (
        f"<div class='tpos-ribbon'>"
        f"<div style='font-weight:700;margin-bottom:6px;'>Résumé {temoin}</div>"
        f"<div class='kpi-card' style='margin-bottom:6px;'>"
        f"<div class='kpi-pill'><span class='kpi-dot{' warn' if pct_bad>0 else ''}'></span>"
        f"<span>Global · n={n_all} · S4: <b>{_pct_fmt(s4m)}</b> · S6: <b>{_pct_fmt(s6m)}</b> · sous seuil: <b>{pct_bad:.1f}%</b></span></div>"
        f"</div>"
        f"<div style='display:flex;flex-wrap:wrap;'>{''.join(chips)}</div>"
        f"</div>"
    )
# === Commentaires simples RUN/PLAQUE ==========================================
RUN_COMMENTS_FILE = "comments_runs.csv"
PLAQUE_COMMENTS_FILE = "comments_plaques.csv"

def _read_comments_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["key","comment","updated_at"])
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    for c in ["key","comment","updated_at"]:
        if c not in df.columns: df[c] = ""
    return df[["key","comment","updated_at"]]

def _write_comments_csv(df: pd.DataFrame, path: str) -> None:
    tmp = path + ".tmp"; df.to_csv(tmp, index=False); os.replace(tmp, path)

def get_run_comment(run_id: str) -> str:
    if not run_id: return ""
    df = _read_comments_csv(RUN_COMMENTS_FILE)
    m = df["key"] == str(run_id)
    return df.loc[m,"comment"].iloc[0] if m.any() else ""

def set_run_comment(run_id: str, comment: str) -> None:
    if not run_id: return
    df = _read_comments_csv(RUN_COMMENTS_FILE)
    m = df["key"] == str(run_id); now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if m.any():
        df.loc[m,"comment"] = str(comment or ""); df.loc[m,"updated_at"] = now
    else:
        df = pd.concat([df, pd.DataFrame([{"key":str(run_id), "comment":str(comment or ""), "updated_at":now}])], ignore_index=True)
    _write_comments_csv(df, RUN_COMMENTS_FILE)

def get_plaque_comment(plaque_id: str) -> str:
    if not plaque_id: return ""
    df = _read_comments_csv(PLAQUE_COMMENTS_FILE)
    m = df["key"] == str(plaque_id)
    return df.loc[m,"comment"].iloc[0] if m.any() else ""

def set_plaque_comment(plaque_id: str, comment: str) -> None:
    if not plaque_id: return
    df = _read_comments_csv(PLAQUE_COMMENTS_FILE)
    m = df["key"] == str(plaque_id); now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if m.any():
        df.loc[m,"comment"] = str(comment or ""); df.loc[m,"updated_at"] = now
    else:
        df = pd.concat([df, pd.DataFrame([{"key":str(plaque_id), "comment":str(comment or ""), "updated_at":now}])], ignore_index=True)
    _write_comments_csv(df, PLAQUE_COMMENTS_FILE)

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
            raw_comment = str(group.loc[idx, 'commentaire'])
            commentaire = raw_comment.lower()
            match = re.search(r"lot\s*([A-Za-z0-9][A-Za-z0-9_/-]*)", raw_comment, flags=re.IGNORECASE)
            if ('nouveau' in commentaire or 'nouvelle' in commentaire) and match:
                new_lots[i] = match.group(1).strip()
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
    - Parcourt df_temoin dans l'ordre (date_heure > summary_run_id > plaque_id > sample_id).
    - Met à jour le lot quand on voit l'un des formats suivants dans 'commentaire' :
        • "Nouveau lot Témoin : <ID>"
        • "nouveau lot <ID>"
      (et ignore explicitement "Autre nouveau lot ...", qui n'est pas un lot Témoin)
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

    # Sélection du témoin
    t = str(temoin_code or "").lower()
    def _is_this_temoin(sid: str) -> bool:
        sid = str(sid).lower()
        if t == "tposh1": return "tposh1" in sid
        if t == "tposh3": return "tposh3" in sid
        if t == "tposb":  return "tposb"  in sid
        # fallback : contient la chaîne fournie
        return (t != "") and (t in sid)

    # Regex:
    # - ignore 'Autre nouveau lot ...'
    re_autre = re.compile(r"\bautre\b.*\bnouveau\w*\s+lot", flags=re.IGNORECASE)
    # - capture:
    #     "Nouveau lot Témoin : <ID>"  OU  "nouveau lot <ID>"
    re_lot1  = re.compile(r"\b(?:nouveau|nouvelle)\s+lot\s*t[ée]moin\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9_\-\/]*)",
                          flags=re.IGNORECASE)
    re_lot2  = re.compile(r"\b(?:nouveau|nouvelle)\s+lot\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9_\-\/]*)",
                          flags=re.IGNORECASE)

    lot_actuel = None
    lots_out = []
    for _, row in df.iterrows():
        sid = row.get("sample_id", "")
        if not _is_this_temoin(sid):
            lots_out.append(None)
            continue

        txt = str(row.get("commentaire", "") or "")

        # Ne pas confondre avec "Autre nouveau lot ..."
        if re_autre.search(txt):
            # on garde le lot courant, pas de changement
            lots_out.append(lot_actuel)
            continue

        m = re_lot1.search(txt) or re_lot2.search(txt)
        if m:
            lot_actuel = m.group(1).strip()

        lots_out.append(lot_actuel)

    return lots_out
def build_lot_stage_df_for_hist(df_in: pd.DataFrame, plaques: list[str] | None = None) -> pd.DataFrame:
    if df_in is None or df_in.empty:
        return pd.DataFrame(columns=["plaque_id","sample_id","lot_id","stage"])
    out = []
    for temoin in ["TposH1","TposH3","TposB"]:
        sub = df_in[df_in["sample_id"].astype(str).str.contains(temoin, case=False, na=False)].copy()
        if sub.empty:
            continue
        lots, stages = extract_lot_and_stage_for_temoin(sub, temoin)
        sub["lot_id"] = lots
        sub["stage"] = pd.Series(stages, index=sub.index).astype(str).str.lower()
        sub = sub[sub["lot_id"].notna() & sub["stage"].isin(["probatoire","courant"])]
        out.append(sub[["plaque_id","sample_id","lot_id","stage"]])
    df_stage = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["plaque_id","sample_id","lot_id","stage"])
    df_stage["plaque_id"] = df_stage["plaque_id"].astype(str)
    if plaques is not None:
        df_stage = df_stage[df_stage["plaque_id"].isin(set(map(str, plaques)))]
    return df_stage

# === Nouveau : extraction "lot + état" pour un témoin (courant/probatoire)
def extract_lot_and_stage_for_temoin(df_temoin: pd.DataFrame, temoin_code: str):
    """
    Renvoie (lot_actif, lot_stage) alignés à df_temoin pour un témoin.
      lot_stage ∈ {'courant','probatoire', None}

    Règles:
      • 'lot en cours <ID>' sur une plaque = promotion du <ID> à partir de CETTE plaque.
        ➜ On trace à cette plaque: (1) l'ancien lot (dernier point) + (2) le lot promu (premier point).
      • 'nouveau lot <ID>': si pas de courant encore → <ID> devient courant (pas de probatoire);
        sinon <ID> est probatoire.
      • Le probatoire se propage automatiquement plaque→plaque tant qu'il n'est pas promu.
      • Pas de propagation intra-plaque (on respecte les 2 lignes distinctes).
    """
    if df_temoin is None or df_temoin.empty:
        return [None] * len(df_temoin), [None] * len(df_temoin)

    import re
    df = df_temoin.copy()

    # Axe X pour l'ordre
    if   "date_heure"     in df.columns: xorder = "date_heure"
    elif "summary_run_id" in df.columns: xorder = "summary_run_id"
    elif "plaque_id"      in df.columns: xorder = "plaque_id"
    else:                                 xorder = "sample_id"

    df.sort_values([xorder, "sample_id"], inplace=True)

    idx = df.index
    lot_per_row   = pd.Series([None] * len(df), index=idx, dtype=object)
    stage_per_row = pd.Series([None] * len(df), index=idx, dtype=object)

    TEMOIN_RE  = re.compile(re.escape(str(temoin_code)), re.I)
    NEW_RE = re.compile(r"(?:nouveau\s*lot(?:\s*t[ée]moin)?)\s*[:\-]?\s*([A-Za-z0-9][\w\-\/]*)",re.I)
    CUR_RE = re.compile(r"(?:lot\s*en\s*cours(?:\s*t[ée]moin)?)\s*[:\-]?\s*([A-Za-z0-9][\w\-\/]*)",re.I)
    CLOSE_RE   = re.compile(r"(?:lot\s*clos)[:\s\-]*([A-Za-z0-9][\w\-\/]*)", re.I)
    NEW_TPOS_RE = re.compile(r"(?:nouveau\s*lot\s*t[ée]moin)\s*[:\-]?\s*([A-Za-z0-9][\w\-\/]*)", re.I)


    current_lot    = None     # lot "officiel" pour la continuité
    probatoire_lot = None     # lot probatoire en cours (propagation)
    last_prob_sid  = None     # sample_id préféré pour recoller le probatoire
    closed_lots = set()

    for _, grp in df.groupby(xorder, sort=False):
        g_tpos = grp[grp["sample_id"].astype(str).str.contains(TEMOIN_RE, na=False)]

        # Détections dans la plaque
        new_ids = {}            # index -> lot 'nouveau lot'
        promote_to = None       # <ID> de 'lot en cours <ID>'
        promoted_index = None   # index de la ligne qui porte la promotion

        for i, r in g_tpos.iterrows():
            c = str(r.get("commentaire", "") or "")
            mcur = CUR_RE.search(c)
            if mcur:
                lot_id = mcur.group(1).strip()
                if promote_to is None:
                    promote_to = lot_id
                if lot_id == promote_to and promoted_index is None:
                    promoted_index = i
            if COMMENT_NEWLOT_IMPLIES_CURRENT:
                mnew_tpos = NEW_TPOS_RE.search(c)
                if (mnew_tpos is not None) and (promote_to is None):
                    lot_id = mnew_tpos.group(1).strip()
                    promote_to = lot_id
                    if promoted_index is None:
                        promoted_index = i
            mnew_tpos = NEW_TPOS_PROMOTE_RE.search(c)
            if mnew_tpos and (promote_to is None):
                # Mode "correction" : un "Nouveau lot Témoin : X" depuis l’outil commentaire
                # fait devenir X "courant" immédiatement sur CETTE plaque,
                # puis X se propage en courant sur les suivantes.
                lot_id = mnew_tpos.group(1).strip()
                promote_to = lot_id
                if promoted_index is None:
                    promoted_index = i

            mnew = NEW_RE.search(c)
            if mnew:
                new_ids[i] = mnew.group(1).strip()
            mclo = CLOSE_RE.search(c)
            if mclo:
                closed_lots.add(mclo.group(1).strip())

        # Ajout sans probatoire: 1er 'nouveau lot' devient courant s'il n'y a encore aucun courant
        if promote_to is None and current_lot is None and len(new_ids) >= 1:
            promote_to = list(new_ids.values())[0]
            # si on sait quelle ligne porte ce 'nouveau lot', on la note comme promue
            for i, lot_id in new_ids.items():
                if lot_id == promote_to:
                    promoted_index = i
                    break
            probatoire_lot = None

        prev_current = current_lot                  # ancien lot courant (avant éventuelle promotion)
        curr_target  = promote_to or current_lot    # lot courant à utiliser pour cette plaque

        # probatoire pour cette plaque (si pas de promotion explicite)
        prob_target = None
        if promote_to is None:
            if len(new_ids) >= 1:
                first_new = list(new_ids.values())[0]
                if first_new != curr_target:
                    prob_target = first_new
            if (prob_target is None) and (probatoire_lot is not None) and (probatoire_lot != curr_target):
                prob_target = probatoire_lot

        # Choix déterministe de la ligne probatoire si besoin
        prob_index = None
        if prob_target is not None:
            tpos_rows = list(g_tpos.sort_values("sample_id").index)
            # priorité à la ligne portant "nouveau lot <prob_target>"
            for i in tpos_rows:
                c = str(grp.loc[i].get("commentaire", "") or "")
                m = NEW_RE.search(c)
                if m and m.group(1).strip() == prob_target:
                    prob_index = i
                    break
            # sinon, reprends le même sample_id que la plaque précédente si possible
            if prob_index is None and last_prob_sid is not None:
                cand = [i for i in tpos_rows if str(grp.loc[i, "sample_id"]) == str(last_prob_sid)]
                if cand:
                    prob_index = cand[0]
            # sinon 2e ligne si duplicat, sinon l'unique
            if prob_index is None and tpos_rows:
                prob_index = tpos_rows[1] if len(tpos_rows) >= 2 else tpos_rows[0]

        # --- Assignation des rôles ligne par ligne ---
        for i in grp.index:
            sid = str(grp.loc[i].get("sample_id", ""))
            if not TEMOIN_RE.search(sid):
                lot_per_row.at[i] = None; stage_per_row.at[i] = None
                continue

            if promote_to:
                # PLAQUE DE PROMOTION : on trace les DEUX points
                if i == promoted_index:
                    # la ligne promue -> nouveau lot, courant
                    lot_per_row.at[i]   = promote_to
                    stage_per_row.at[i] = "courant"
                else:
                    # l'autre ligne (ancien lot) -> dernier point de l'ancien lot
                    if prev_current:
                        lot_per_row.at[i]   = prev_current
                        stage_per_row.at[i] = "courant"
                    else:
                        lot_per_row.at[i]   = None
                        stage_per_row.at[i] = None
            else:
                if i == prob_index and prob_target is not None:
                    lot_per_row.at[i]   = prob_target
                    stage_per_row.at[i] = "probatoire"
                else:
                    lot_per_row.at[i]   = curr_target
                    stage_per_row.at[i] = "courant" if curr_target else None

        # Fin de plaque : appliquer la promotion pour la suite
        if promote_to:
            current_lot    = promote_to
            probatoire_lot = None
            last_prob_sid  = None
        else:
            if prob_target and prob_target != current_lot:
                probatoire_lot = prob_target
                if prob_index is not None:
                    last_prob_sid = str(grp.loc[prob_index, "sample_id"])
            else:
                last_prob_sid = None
        # Clôtures effectives après cette plaque
        if current_lot in closed_lots:
            current_lot = None
        if probatoire_lot in closed_lots:
            probatoire_lot = None


    return lot_per_row.reindex(df_temoin.index).tolist(), stage_per_row.reindex(df_temoin.index).tolist()
    
def known_lots(df: pd.DataFrame, temoin_code: str) -> list[str]:
    """
    Retourne la liste triée des lots déjà vus pour un témoin (dans 'commentaire'),
    en parcourant tout le DataFrame (ex: base_df).
    """
    import re
    if df is None or df.empty:
        return []
    mask = df["sample_id"].astype(str).str.contains(str(temoin_code), case=False, na=False)
    sub = df.loc[mask, "commentaire"].astype(str).fillna("")

    NEW_RE = re.compile(r"(?:nouveau)\s*lot[:\s\-]*([A-Za-z0-9][\w\-\/]*)", re.I)
    CUR_RE = re.compile(r"(?:lot\s*en\s*cours)[:\s\-]*([A-Za-z0-9][\w\-\/]*)", re.I)

    lots: set[str] = set()
    for c in sub:
        for m in NEW_RE.findall(c):
            lots.add(m.strip())
        for m in CUR_RE.findall(c):
            lots.add(m.strip())
    # tri simple (alpha), tu peux adapter
    return sorted(lots)
    
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
    
LOT_REGEX = re.compile(
    r"\b(?:nouveau\s*lot|lot\s*en\s*cours|autre\s*nouveau\s*lot|lot\s*clos)[:\s\-]*([A-Za-z0-9][A-Za-z0-9_\-\/]*)",
    flags=re.IGNORECASE
)

def extract_lot_from_comment(txt: str):
    """Retourne l'ID de lot (dernier trouvé) ou None si absent."""
    if not isinstance(txt, str):
        return None
    last = None
    for m in LOT_REGEX.finditer(txt):
        last = m.group(1)
    return (last or None)
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
    show_dropdown: bool = False
):
    """
    S4 (plein) & S6 (pointillé) pour un témoin.
    - X = plaque_id uniquement (ordre naturel).
    - Une ligne CONTINUE par lot (probatoire + courant fusionnés).
    - Marqueurs d’état: courant=cercle, probatoire=losange.
    - Gaps uniquement si S4 ou S6 manquent (⚠ posé sur la plaque).
    - Carry-forward: 'lot en cours <ID>' se propage jusqu’à 'lot en cours <J>' ou 'lot clos <ID>'.
    """
    need = {"sample_id","summary_consensus_perccoverage_S4","summary_consensus_perccoverage_S6","commentaire"}
    try:
        _ = temoin_df.columns
    except Exception:
        return go.Figure()
    if not need.issubset(set(temoin_df.columns)):
        return go.Figure()

    # --- Filtre témoin ---
    sub = temoin_df[temoin_df["sample_id"].astype(str).str.contains(str(temoin), case=False, na=False)].copy()
    if sub.empty:
        try: st.info(f"Aucun {temoin} dans la sélection courante.")
        except Exception: pass
        return go.Figure()

    # --- Axe X = plaque_id (ordre naturel) ---
    def _plaque_index(pid: str):
        m = re.search(r"(\d+)", str(pid) or "")
        return int(m.group(1)) if m else 10**9

    if x_col in sub.columns: xorder = x_col
    elif "plaque_id" in sub.columns: xorder = "plaque_id"
    elif "date_heure" in sub.columns: xorder = "date_heure"
    else: xorder = "sample_id"

    sub["plaque_id"] = sub["plaque_id"].astype(str).str.strip()
    cat_order = sorted(sub["plaque_id"].unique().tolist(), key=_plaque_index)
    xorder = "plaque_id"  # on force plaque_id

    # --- Numériques ---
    sub["S4"] = pd.to_numeric(sub["summary_consensus_perccoverage_S4"], errors="coerce")
    sub["S6"] = pd.to_numeric(sub["summary_consensus_perccoverage_S6"], errors="coerce")

    # --- Lot + état ---
    try:
        lots, stages = extract_lot_and_stage_for_temoin(sub, temoin)
    except Exception:
        lots = extract_lot_for_temoin(sub, temoin)
        stages = [None] * len(sub)
    sub["lot_actif"] = lots
    sub["lot_stage"] = stages

    # Carry-forward du "lot en cours"
    sub = sub.sort_values(by=["plaque_id"], key=lambda s: s.astype(str).str.extract(r"(\d+)").fillna("999999").astype(int)[0]).reset_index(drop=True)
    cur_lot = None
    def _has_closure_of(comm: str, lot: str) -> bool:
        return bool(lot) and re.search(rf"(?i)\blot\s*clos\s+{re.escape(str(lot))}\b", str(comm or "")) is not None

    for i, r in sub.iterrows():
        stage = (str(r.get("lot_stage","")) or "").lower()
        lot   = r.get("lot_actif", None)
        if pd.notna(lot) and stage == "courant":
            cur_lot = lot
        elif pd.isna(lot) and cur_lot is not None:
            sub.at[i, "lot_actif"] = cur_lot
            sub.at[i, "lot_stage"] = "courant"
        if _has_closure_of(r.get("commentaire",""), cur_lot):
            cur_lot = None

    sub = sub[sub["lot_actif"].notna()].copy()
    sub["lot_lbl"] = sub["lot_actif"].apply(lambda v: "Sans lot" if (pd.isna(v) or str(v).strip()=="") else str(v))

    # --- Figure ---
    fig = go.Figure()
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=cat_order, title_text="")
    base_cols = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    color_map = {lot: base_cols[i % len(base_cols)] for i, lot in enumerate(sub["lot_lbl"].drop_duplicates())}

    # helper: garder 1 point/plaque (dernier en date si dispo)
    def _agg_one_per_plaque(df):
        if df.empty:
            return pd.DataFrame({"plaque_id": [], "S4": [], "S6": []})
        if "date_heure" in df.columns:
            df = df.sort_values("date_heure")
            pick = df.groupby("plaque_id", as_index=False).tail(1)
        else:
            pick = df.drop_duplicates(subset=["plaque_id"], keep="last")
        return pick[["plaque_id","S4","S6"]].copy()

    # --- Une ligne par lot ---
    for lot, g in sub.groupby("lot_lbl", dropna=False):
        color = color_map.get(lot, "#1f77b4")
        g["lot_stage"] = g["lot_stage"].astype(str).str.lower().fillna("")
        g["_pl_ord"]   = g["plaque_id"].map(_plaque_index)
        g = g.sort_values("_pl_ord")

        g_cur  = g[g["lot_stage"].eq("courant")].copy()
        g_prob = g[g["lot_stage"].eq("probatoire")].copy()
        c = _agg_one_per_plaque(g_cur)
        p = _agg_one_per_plaque(g_prob)

        # Fusion courant+probatoire (priorité au courant)
        all_x = pd.DataFrame({"plaque_id": cat_order})
        m = (all_x
             .merge(c.rename(columns={"S4":"S4_c","S6":"S6_c"}), on="plaque_id", how="left")
             .merge(p.rename(columns={"S4":"S4_p","S6":"S6_p"}), on="plaque_id", how="left"))
        m["S4_all"] = m["S4_c"].where(m["S4_c"].notna(), m["S4_p"])
        m["S6_all"] = m["S6_c"].where(m["S6_c"].notna(), m["S6_p"])
        # Un lot est "présent" sur une plaque si au moins une des deux valeurs existe
        present = m["S4_all"].notna() | m["S6_all"].notna()


        # Lignes continues
        fig.add_trace(go.Scatter(
            x=m["plaque_id"], y=m["S4_all"], mode="lines",
            name=f"S4 · Lot {lot}", line=dict(width=2.2, dash="solid", color=color),
            connectgaps=False, legendgroup=f"{lot}_base", showlegend=True
        ))
        fig.add_trace(go.Scatter(
            x=m["plaque_id"], y=m["S6_all"], mode="lines",
            name=f"S6 · Lot {lot}", line=dict(width=1.7, dash="dash", color=color),
            connectgaps=False, legendgroup=f"{lot}_base", showlegend=True
        ))

        # Marqueurs d’état (sur S4 pour lisibilité)
        if not c.empty:
            fig.add_trace(go.Scatter(
                x=c["plaque_id"], y=c["S4"], mode="markers",
                name=f"Courant · Lot {lot}",
                marker=dict(size=7, symbol="circle", color=color),
                showlegend=False, hoverinfo="skip"
            ))
        if not p.empty:
            fig.add_trace(go.Scatter(
                x=p["plaque_id"], y=p["S4"], mode="markers",
                name=f"Probatoire · Lot {lot}",
                marker=dict(size=7, symbol="diamond", color=color),
                showlegend=False, hoverinfo="skip"
            ))



    # Re-run (×n)
    counts = sub.groupby("plaque_id").size()
    rr = counts[counts > 1]
    if not rr.empty:
        fig.add_trace(go.Scatter(
            x=rr.index.tolist(), y=[seuil]*len(rr), mode="text",
            text=[f"×{int(n)}" for n in rr.values], textposition="bottom center",
            hovertext=[f"{int(n)} runs" for n in rr.values],
            hoverinfo="text", showlegend=False
        ))

    fig.update_layout(
        title=f"{temoin} — S4/S6",
        xaxis_title="plaque_id",
        yaxis_title="% Couverture",
        yaxis=dict(range=[0,110]),
        xaxis=dict(tickangle=-45),
        margin=dict(l=20, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )

    try:
        if seuil is not None:
            fig.add_hline(y=float(seuil), line_dash="dot", line_color="#999", opacity=0.6)
    except Exception:
        pass

    # Badges complémentaires éventuels
    try:
        fig, _ = add_comment_badges_on_fig(fig, temoin_df, temoin, x_col="plaque_id")
    except Exception:
        pass
    try:
        fig = add_lot_labels_on_top(fig, temoin_df, temoin, x_col="plaque_id")
    except Exception:
        pass

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
    # 2bis) Ordre X immuable (catégories fixées pour TOUTES les traces)
    cat_order = list(pd.unique(sub[xorder].astype(str)))

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
    fig.update_xaxes(categoryorder="array", categoryarray=cat_order)

    return fig

def plot_histogram_with_export(df, titre, filename, all_samples_df=None, run_id=None, lot_stage_df=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["plaque_id"].astype(str), y=df["pct_ininterpretable"],
        text=[f"{v:.1f}%" for v in df["pct_ininterpretable"]],
        textposition="outside", marker_color="indianred", name="% Ininterprétable"
    ))
    fig.update_layout(title=titre, xaxis_title="Plaque ID", yaxis_title="% Ininterprétable",
                      yaxis=dict(range=[0,115]), xaxis_tickangle=-45,
                      margin=dict(l=40, r=40, t=110, b=100), showlegend=False)
    if all_samples_df is not None:
        lsdf = lot_stage_df if lot_stage_df is not None else build_lot_stage_df_for_hist(
            all_samples_df, plaques=df["plaque_id"].astype(str).tolist()
        )
        fig = add_run_plaque_flags_on_fig(fig, df, all_samples_df, "plaque_id", run_id, lsdf)
    st.plotly_chart(fig, use_container_width=True)
    csv_buffer = io.StringIO(); df.to_csv(csv_buffer, index=False)
    st.download_button("Télécharger les données CSV", csv_buffer.getvalue(), file_name=filename, mime="text/csv")

def plot_varcount(df, titre, all_samples_df=None, run_id=None, lot_stage_df=None):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["plaque_id"].astype(str), y=df["% varcount >= 13"],
        text=[f"{v:.1f}%" for v in df["% varcount >= 13"]],
        textposition="outside", marker_color="seagreen", name="% varcount >= 13"
    ))
    fig.update_layout(title=titre, xaxis_title="Plaque ID", yaxis_title="% avec varcount >= 13",
                      yaxis=dict(range=[0,115]), xaxis_tickangle=-45,
                      margin=dict(l=40, r=40, t=110, b=100), showlegend=False)
    if all_samples_df is not None:
        lsdf = lot_stage_df if lot_stage_df is not None else build_lot_stage_df_for_hist(
            all_samples_df, plaques=df["plaque_id"].astype(str).tolist()
        )
        fig = add_run_plaque_flags_on_fig(fig, df, all_samples_df, "plaque_id", run_id, lsdf)
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

# --- Déduplication locale "Archives" (conserve la dernière occurrence par clé) ---
def _arch_dedup_view(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    av = df.copy()

    # Clés de dédup possibles (on ne prend que celles présentes)
    keys = []
    if "sample_id" in av.columns:
        av["__k_sample"] = av["sample_id"].astype(str).str.strip().str.upper()
        keys.append("__k_sample")
    if "plaque_id" in av.columns:
        av["__k_plaque"] = av["plaque_id"].astype(str).str.strip().str.upper()
        keys.append("__k_plaque")
    if "summary_run_id" in av.columns:
        av["__k_run"] = av["summary_run_id"].astype(str).str.strip().str.upper()
        keys.append("__k_run")

    if not keys:
        return av  # rien pour dédupliquer

    # Si on a une date, on garde la + récente; sinon on garde la dernière lue
    date_candidates = [c for c in ["date_heure","date","summary_date","run_date","created_at"] if c in av.columns]
    if date_candidates:
        dc = date_candidates[0]
        av["_d"] = pd.to_datetime(av[dc], errors="coerce")
        av = av.sort_values("_d", kind="mergesort")  # stable, de l'ancienne à la plus récente
    # Garder la dernière occurrence par clé
    av = av.drop_duplicates(subset=keys, keep="last").copy()

    # Nettoyage des colonnes techniques
    av.drop(columns=[c for c in ["__k_sample","__k_plaque","__k_run","_d"] if c in av.columns], inplace=True, errors="ignore")
    return av

def sort_ids_by_recency(df: pd.DataFrame, id_col: str) -> list:
        all_ids = [str(x) for x in df.get(id_col, pd.Series(dtype=object)).dropna().unique().tolist()]
        if not all_ids:
            return []

        # 1) Si une vraie colonne date existe, on l'utilise (max par ID, puis décroissant)
        date_candidates = [c for c in ["date_heure", "date", "Date", "summary_date", "run_date", "created_at"] if c in df.columns]
        if date_candidates:
            dc = date_candidates[0]
            dd = pd.to_datetime(df[dc], errors="coerce")
            tmp = pd.DataFrame({id_col: df[id_col].astype(str), "_d": dd})
            order = (tmp.dropna(subset=["_d"])
                       .groupby(id_col)["_d"].max()
                       .sort_values(ascending=False)
                       .index.tolist())
            # compléter avec ceux sans date
            return order + [i for i in all_ids if i not in order]

        # 2) Pas de colonne date → on fabrique une "clé de récence" par ID

        # 2a) YYYYMMDD ou YYYY-MM-DD dans l'ID
        def parse_yyyy_mm_dd(s: str):
            m = re.search(r"(20\d{2})[-_/]?(0[1-9]|1[0-2])[-_/]?([0-2]\d|3[01])", s)
            if not m:
                return None
            y, mth, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            try:
                return datetime(y, mth, d)
            except ValueError:
                return None

        # 2b) YYMMDD (ex: 250717 → 2025-07-17)
        def parse_yymmdd(s: str):
            if not re.fullmatch(r"\d{6}", s):
                return None
            yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
            try:
                return datetime(2000 + yy, mm, dd)
            except ValueError:
                return None

        # 2c) Motif type "25P081" → approx récence par (année, numéro)
        def score_year_seq(s: str):
            m = re.match(r"^(\d{2})[A-Za-z]+0*([0-9]+)$", s)
            if not m:
                return None
            yy = 2000 + int(m.group(1))
            seq = int(m.group(2))
            return yy * 10000 + seq

        # 2d) Natural sort key (dernier recours)
        def natural_key(s: str):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

        def make_key(s: str):
            s = str(s)
            dt = parse_yyyy_mm_dd(s) or parse_yymmdd(s)
            if dt is not None:
                return (3, dt)         # priorité 1 : vraie date dans l'ID
            ys = score_year_seq(s)
            if ys is not None:
                return (2, ys)         # priorité 2 : motif année+numéro
            return (1, natural_key(s)) # priorité 3 : natural key

        # Tri décroissant par (groupe, valeur)
        return sorted(all_ids, key=make_key, reverse=True)
        
# PARTIE 5 Archives

@lru_cache(maxsize=50000)
def _cached_plaque_comment(pid: str) -> str:
    try:
        return str(get_plaque_comment(pid) or "")
    except Exception:
        return ""

@lru_cache(maxsize=50000)
def _cached_run_comment(rid: str) -> str:
    try:
        return str(get_run_comment(rid) or "")
    except Exception:
        return ""

def add_icons_column_for_archives(df_in: pd.DataFrame, col_name: str = "__icons__") -> pd.DataFrame:
    """
    Construit __icons__ (emojis) et __icons___tt (tooltip) de façon robuste.
    - AUCUNE concat NumPy de strings (évite UFuncTypeError sur '<U2').
    - Caches plaque/run pour éviter les I/O répétés.
    Hypothèses: df_in contient déjà 'dup' (bool), 'commentaire', 'plaque_id', 'summary_run_id'.
    """
    if df_in is None or df_in.empty:
        return df_in

    df = df_in.copy()

    # Colonnes de base (robustes)
    s_cmt = df.get("commentaire", pd.Series(index=df.index, dtype=object)).astype(str).fillna("")
    s_pid = df.get("plaque_id",  pd.Series(index=df.index, dtype=object)).astype(str).fillna("")
    s_rid = df.get("summary_run_id", pd.Series(index=df.index, dtype=object)).astype(str).fillna("")
    s_dup = df.get("dup", False)

    # Maps plaque/run (1 appel par id grâce au cache)
    unique_pids = pd.unique(s_pid[s_pid != ""])
    unique_rids = pd.unique(s_rid[s_rid != ""])
    plaque_map = {pid: _cached_plaque_comment(pid) for pid in unique_pids}
    run_map    = {rid: _cached_run_comment(rid)    for rid in unique_rids}

    plaque_txt = s_pid.map(plaque_map).fillna("")
    run_txt    = s_rid.map(run_map).fillna("")

    # Regex compilées
    re_kit = re.compile(r"\bautre\b.*\bnouveau\w*\s+lot", flags=re.I)
    re_lot = re.compile(r"\b(?:nouveau|nouvelle)\s+lot(?:\s*t[ée]moin)?\b\s*[:\-]?\s*([A-Za-z0-9][A-Za-z0-9_\-\/]*)", re.I)

    # Détections principales (vectorisées)
    has_kit_sample = s_cmt.str.contains(re_kit, na=False)
    has_kit_plaque = plaque_txt.str.contains(re_kit, na=False)
    has_kit_run    = run_txt.str.contains(re_kit, na=False)

    # Nouveau lot témoin (ignoré si 'autre nouveau lot' au niveau échantillon)
    lot_id = s_cmt.where(~has_kit_sample).str.extract(re_lot, expand=False).fillna("")

    # Miroirs plaque/run (par ligne, mais O(5k) → rapide)
    is_mirror_plq = pd.Series(
        [(pt.strip() != "") and (pt.lower() in c.lower())
         for pt, c in zip(plaque_txt.tolist(), s_cmt.tolist())],
        index=df.index
    )
    is_mirror_run = pd.Series(
        [(rt.strip() != "") and (rt.lower() in c.lower())
         for rt, c in zip(run_txt.tolist(), s_cmt.tolist())],
        index=df.index
    )

    has_sample_comment = (s_cmt.str.strip() != "") & (~has_kit_sample) & (lot_id == "") & (~is_mirror_plq) & (~is_mirror_run)
    show_plaque = (plaque_txt.str.strip() != "") & (~has_kit_plaque)
    show_run    = (run_txt.str.strip() != "")    & (~has_kit_run)

    # -------------------------
    # Construction SANS NumPy +
    # -------------------------
    icons_list: list[str] = []
    tips_list:  list[str] = []

    for i in df.index:
        ic = ""
        tp = []

        # 🕒 → seulement si Glims_id en double (dup True)
        d = s_dup.loc[i]
        if isinstance(d, str):
            is_dup = d.strip().lower() in ("1", "true", "yes")
        else:
            is_dup = bool(d)
        if is_dup:
            ic += "🕒 "
            tp.append("Échantillon repassé (Glims_id en double)")

        # 🧪 (priorité locale: sample > plaque > run)
        if has_kit_sample.loc[i]:
            ic += "🧪 "
            tp.append("Autre nouveau lot (kit/amorces) — échantillon")
        elif has_kit_plaque.loc[i]:
            ic += "🧪 "
            tp.append(f"Autre nouveau lot (kit/amorces) — plaque {s_pid.loc[i]}")
        elif has_kit_run.loc[i]:
            ic += "🧪 "
            tp.append(f"Autre nouveau lot (kit/amorces) — run {s_rid.loc[i]}")

        # 🔁 (coexiste avec 🧪)
        if lot_id.loc[i] != "":
            ic += "🔁 "
            tp.append(f"Nouveau lot Témoin : {lot_id.loc[i]}")

        # 💬
        if has_sample_comment.loc[i]:
            c = s_cmt.loc[i]
            tp.append(f"Commentaire échantillon : {c[:120]}{'…' if len(c)>120 else ''}")
            ic += "💬 "

        # 🏷️
        if show_plaque.loc[i]:
            p = plaque_txt.loc[i]
            tp.append(f"Commentaire plaque {s_pid.loc[i]} : {p[:120]}{'…' if len(p)>120 else ''}")
            ic += "🏷️ "

        # 📌
        if show_run.loc[i]:
            r = run_txt.loc[i]
            tp.append(f"Commentaire run {s_rid.loc[i]} : {r[:120]}{'…' if len(r)>120 else ''}")
            ic += "📌 "

        icons_list.append(ic.strip())
        tips_list.append(" | ".join(tp))

    df[col_name] = pd.Series(icons_list, index=df.index, dtype=object)
    df[col_name + "_tt"] = pd.Series(tips_list, index=df.index, dtype=object)

    # place __icons__ en première colonne
    cols = [col_name] + [c for c in df.columns if c != col_name]
    return df[cols]



