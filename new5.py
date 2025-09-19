import re
import os
import io
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pathlib
import plotly.express as px
import streamlit.components.v1 as components
import numpy as np
import hashlib
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, GridUpdateMode
import math
import definitions_flu
from pathlib import Path
from io import StringIO
from datetime import datetime

sections = []
# --- Couleurs Plotly
palette = px.colors.qualitative.Plotly

# --- Vos utilitaires maison
# 💡 IMPORTEZ *explicitement* ce que vous utilisez (lisible, évite les surprises)
from definitions_flu import (
    # layout / pages / assets
    load_static_css, page_aide, page_plaque, page_suivi, page_historique,
    # helpers booléens
    answered, is_yes, is_no,
    # parsing / logique métier
    extraire_plaque, filter_gra_group, append_history, count_double_pop, detect_souches,
    build_ininterpretable_html, register_section_html, make_counts, register_section,
    build_secondary_html, build_interclade_html, build_pie_div, make_report_html, _wrap_label,
    assign_lots_from_commentaires, display_grippe_temoins_complet, render_table,
    compute_x_grouped_id, has_special_comment, extract_lot_corrected, add_lot_segments,
    plot_histogram_with_export, plot_varcount, render_table_cmap, reference_map, _norm, _have_all, _well_positions, _occupied_from_sample_ids, _make_plate_fig_96, _map_excel_to_384_positions, _make_plate384_fig_from_map,
    safe_read_historique_data, persist_full_dataset_atomic, update_comment_and_persist, plot_temoin_lots_s4s6_unique, plot_temoin_autres_segments_curves, _check_cols, _lot_label, has_cols, _x_order_col,extract_lot_for_temoin,
    _parse_flags_from_comment,add_lot_labels_on_top, inject_local_css, _pct_fmt, tpos_stats_ribbon, _read_comments_csv, _write_comments_csv, get_run_comment, set_run_comment, get_plaque_comment, set_plaque_comment,
    comment_badges,apply_comment_presets, _percent_series, compute_temoin_stats_cards, render_temoin_stats_header, add_comment_badges_on_fig, render_comment_feed, apply_comment_plate, bulk_update_comments, log_temoin_lot_event, apply_new_lot_for_temoin,
    _make_tab1_comment, sort_ids_by_recency, _arch_dedup_view, add_run_plaque_flags_on_fig, add_icons_column_for_archives, expected_ref_for_temoin, extract_lot_and_stage_for_temoin, known_lots)


# --- Configuration Streamlit (⚠️ exactement UNE fois et AVANT tout output)
st.set_page_config(page_title="Suivi Qualité - Séquençage Grippe", layout="wide")
st.title("🧪 Dashboard des Séquençages Grippe")

# --- Fichiers de persistance
DATA_FILE = "historique_data.csv"
HISTO_FILE = "historique_fichiers.csv"

# --- État de navigation
if "page" not in st.session_state:
    st.session_state.page = "Aide à la confirmation"

# --- Styles
load_static_css("static/style.css")

# --- Tabs
st.markdown("""
<style>
.stTabs [role="tab"][aria-selected="true"]  { background:#2c3e50; color:#fff; border-radius:8px 8px 0 0; }
.stTabs [role="tab"][aria-selected="false"] { background:#ecf0f1; color:#2c3e50; border-radius:8px 8px 0 0; }
.stTabs [role="tab"] { padding:10px 20px; font-weight:bold; font-size:15px; }
.divider { height:1px; background:linear-gradient(90deg,#0000,#bdc3c7,#0000); margin:12px 0 4px; }
.separator-hr { border:none; border-top:1px solid #e3e3e3; margin:16px 0; }
.big-number { font-size:22px; }
.small-number { font-size:18px; }
</style>
""", unsafe_allow_html=True)

# --- Tabs (inchangés) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Aide à la confirmation",
    "🧩 Plan de plaque",
    "📈 Suivi de performance",
    "📜 Historique des chargements",
    "📦 Archives"
])

# --- Données chargées (historique) 
base_df = definitions_flu.safe_read_historique_data(DATA_FILE)

with tab1:
    page_aide()  # si ta page_aide() rend des composants ; sinon garde le code inline
    st.subheader("📝 Questionnaire de confirmation")
    
    # --- INIT (avant les widgets) ---
    nom_prenom = None
    metriques_run = None
    dosage_qbit = None
    changement_lot = None
    details_lot = None
    nouvelle_dilution = None
    details_dilution = None
    problemes_tech = None
    problemes_tech_ex = None
    non_conf = None
    num_nc = None
    justification_nc = None

    if "questionnaire_validated" not in st.session_state:
        st.session_state["questionnaire_validated"] = False

    col1, col2 = st.columns(2)
    with col1:
        nom_prenom = st.text_input("Nom et Prénom")
        date_analyse = st.date_input("Date de l'analyse")

    with col2:
        st.markdown("**Questions de contrôle**")

        # 1. Métriques du run
        metriques_run = st.radio(
            "Avez-vous regardé les métriques du run ?",
            options=["Sélectionnez", "Oui", "Non"]
        )
        if metriques_run == "Oui":
            dosage_qbit = st.text_input(
                "Valeur du dosage Qubit (ng/mL) :",
                placeholder="Indiquer la valeur du dosage"
            )

        # 2. Changement de lot
        changement_lot = st.radio(
            "Y a-t-il eu changement de lot du/des témoins ?",
            options=["Sélectionnez", "Oui", "Non"]
        )
        if changement_lot == "Oui":
            details_lot = st.text_input(
                "Précisez le(s) lot(s) modifié(s) :",
                placeholder="Numéro de lot, date, etc."
            )
        else:
            details_lot = None

        # 3. Dilution / reconstitution d'amorces
        nouvelle_dilution = st.radio(
            "Y a-t-il eu une nouvelle dilution/reconstitution d'amorces ?",
            options=["Sélectionnez", "Oui", "Non"]
        )
        if nouvelle_dilution == "Oui":
            details_dilution = st.text_input(
                "Détaillez la dilution/reconstitution :",
                placeholder="Volume, concentration, protocole, etc."
            )
        else:
            details_dilution = None

        # 4. Problèmes techniques
        problemes_tech = st.radio(
            "Y a-t-il eu des problèmes techniques (Extraction, PCR, librairie, séquençage) ?",
            options=["Sélectionnez", "Oui", "Non"]
        )
        if problemes_tech == "Oui":
            problemes_tech_ex = st.text_input("Expliquez en quelques mots :", placeholder="")
        else:
            problemes_tech_ex = None

        # 5. Non-conformité (AFFICHÉE SEULEMENT SI problèmes techniques = Oui)
        if problemes_tech == "Oui":
            non_conf = st.radio(
                "Avez-vous ouvert une non-conformité Kalilab ?",
                options=["Sélectionnez", "Oui", "Non"]
            )
            if non_conf == "Oui":
                num_nc = st.text_input("Indiquez le numéro de la NC")
                justification_nc = None
            elif non_conf == "Non":
                justification_nc = st.text_input(
                    "Merci de justifier pourquoi vous n'avez pas ouvert de non-conformité :",
                    placeholder="Expliquez ici..."
                )
                num_nc = None
            else:
                num_nc = None
                justification_nc = None
        else:
            non_conf = None
            num_nc = None
            justification_nc = None

    # ---------------- Session state ----------------
    if "questionnaire_validated" not in st.session_state:
        st.session_state["questionnaire_validated"] = False

    # ---------------- Évaluation ----------------
    details_nc = num_nc
    missing = []
    if not answered(nom_prenom):
        missing.append("Nom et prénom")
    if not answered(metriques_run):
        missing.append("Métriques du run")
    if not answered(changement_lot):
        missing.append("Changement de lot (Oui/Non)")
    elif is_yes(changement_lot) and not answered(details_lot):
        missing.append("Détail du changement de lot (si Oui)")
    if not answered(nouvelle_dilution):
        missing.append("Nouvelle dilution (Oui/Non)")
    elif is_yes(nouvelle_dilution) and not answered(details_dilution):
        missing.append("Détail de la nouvelle dilution (si Oui)")
    if not answered(problemes_tech):
        missing.append("Problèmes techniques (Oui/Non)")
    elif is_yes(problemes_tech) and not answered(problemes_tech_ex):
        missing.append("Explication des problèmes techniques (si Oui)")
    if is_yes(problemes_tech):
        if not answered(non_conf):
            missing.append("Non-conformité (Oui/Non)")
        else:
            if is_yes(non_conf) and not answered(details_nc):
                missing.append("Numéro / Détail de la non-conformité (si Oui)")
            if is_no(non_conf) and not answered(justification_nc):
                missing.append("Justification de l'absence de NC (si Non)")

    form_complet = len(missing) == 0

    # ---------------- Bouton : Valider ----------------
    validate_disabled = (not form_complet) or st.session_state["questionnaire_validated"]
    if st.button("Valider le questionnaire", disabled=validate_disabled):
        st.session_state["questionnaire_validated"] = True
        st.session_state["tab1_comment_text"] = _make_tab1_comment(
            nouvelle_dilution, details_dilution,
            problemes_tech, problemes_tech_ex,
            non_conf, num_nc, justification_nc
        )
        st.session_state["tab1_comment_pending"] = bool(st.session_state["tab1_comment_text"])
        st.session_state["tab1_comment_applied"] = False

    if not form_complet:
        st.warning("⚠️ Merci de remplir toutes les réponses avant de valider.")

    # ---------------- Upload TSV après validation ----------------
    if st.session_state.get("questionnaire_validated", False):
        st.success("✅ Questionnaire validé ! Vous pouvez maintenant charger votre fichier TSV.")
        tsv_file = st.file_uploader(
            "🔄 Charger un nouveau fichier XXX_mergedfluvalid",
            type=["tsv"],
            key="uploader_validated"
        )

        if tsv_file is not None:
            # Taille + reset curseur
            taille_ko = len(tsv_file.getbuffer()) / 1024.0
            tsv_file.seek(0)

            # 1) Lecture du TSV 
            try:
                columns_needed = [
                    "sample_id","summary_reference_id","summary_fastq_readcount","summary_bam_readcount",
                    "summary_bam_meandepth_S1","summary_bam_meandepth_S2","summary_bam_meandepth_S3",
                    "summary_bam_meandepth_S4","summary_bam_meandepth_S5","summary_bam_meandepth_S6",
                    "summary_bam_meandepth_S7","summary_bam_meandepth_S8",
                    "summary_consensus_perccoverage_S1","summary_consensus_perccoverage_S2",
                    "summary_consensus_perccoverage_S3","summary_consensus_perccoverage_S4",
                    "summary_consensus_perccoverage_S5","summary_consensus_perccoverage_S6",
                    "summary_consensus_perccoverage_S7","summary_consensus_perccoverage_S8",
                    "summary_vcf_dpcount","summary_qc_seqcontrol","summary_bam_verif",
                    "summary_vcf_coinf01match","summary_vcf_coinf02iqr",
                    "summary_run_id","val_poi","val_varcount","val_avisbio","val_insertions",
                    "val_result","nextclade_qc_overallStatus","nextclade_frameShifts","nextclade_RBD"
                ]
                new_data = pd.read_csv(tsv_file, sep="\t", dtype=str, usecols=columns_needed)
            except Exception as e:
                st.error(f"❌ Erreur lecture TSV : {e}")
                st.stop()

            # 2) Log historique minimal
            run_ids = new_data.get("summary_run_id")
            run_id_for_log = str(run_ids.dropna().unique()[0]).strip() if run_ids is not None and len(run_ids.dropna().unique()) > 0 else ""
            append_history(nom_fichier=tsv_file.name, taille_ko=taille_ko, run_id=run_id_for_log,operateur=nom_prenom or"")

            # 3) Pré-traitements
            new_data["plaque_id"] = new_data["sample_id"].apply(extraire_plaque)
            new_data["summary_bam_readcount"] = pd.to_numeric(new_data["summary_bam_readcount"], errors="coerce").fillna(0)

            # GRA / GRB filtres
            gra_df = new_data[new_data["sample_id"].str.contains("GRA", na=False)].copy()
            gra_df = gra_df[gra_df["summary_reference_id"] != "EPIISL219327"]
            filtered_gra = gra_df.groupby("sample_id", group_keys=False).apply(filter_gra_group)

            mask_gra_tpos = new_data["sample_id"].str.contains("GRA", na=False) & new_data["sample_id"].str.contains("Tpos", na=False)
            gra_tpos_keep = new_data[mask_gra_tpos & (new_data["summary_reference_id"] == "EPIISL129744")]

            grb_df = new_data[new_data["sample_id"].str.contains("GRB", na=False)].copy()
            grb_filtered = grb_df[grb_df["summary_reference_id"] == "EPIISL219327"]

            new_data_no_gra_grb = new_data[~(new_data["sample_id"].str.contains("GRA", na=False) | new_data["sample_id"].str.contains("GRB", na=False))]
            new_data_filtered = pd.concat([new_data_no_gra_grb, filtered_gra, gra_tpos_keep, grb_filtered], ignore_index=True)

            # Colonnes manquantes ?
            missing_cols = [c for c in columns_needed if c not in new_data_filtered.columns]
            if missing_cols:
                st.error(f"❌ Colonnes manquantes : {missing_cols}")
                st.stop()

            # Ajout colonne commentaire si absente
            if "commentaire" not in new_data_filtered.columns:
                new_data_filtered["commentaire"] = ""

            # Mise à jour historique échantillons
            base_df_local = pd.concat([base_df, new_data_filtered], ignore_index=True)
            base_df_local = base_df_local.drop_duplicates(subset=["sample_id", "nextclade_RBD"])
            definitions_flu.persist_full_dataset_atomic(base_df_local, DATA_FILE)
            
            st.markdown("### 📦 Données chargées")
            st.markdown(f"**Nombre total d’échantillons dans le run :** {new_data_filtered.shape[0]}")
            st.markdown("**Répartition par plaque :**")
            for plaque, count in new_data_filtered["plaque_id"].value_counts().sort_index().items():
                st.markdown(f"🔸 **{plaque}** : {count} échantillons")

            # Sélecteur de plaque
            plaques_disponibles = sorted(new_data_filtered["plaque_id"].dropna().unique())
            # ✅ Rendre dispo pour l’onglet "Plan de plaque"
            st.session_state["new_data_filtered"] = new_data_filtered
            st.session_state["plaques_disponibles"] = plaques_disponibles

            plaque_selectionnee = st.selectbox("🔍 Sélectionnez une plaque :", plaques_disponibles)
            df_plaque = new_data_filtered[new_data_filtered["plaque_id"] == plaque_selectionnee].copy()
            # === Appliquer le commentaire en attente (hors témoins) sur la plaque affichée
            _comment_text = st.session_state.get("tab1_comment_text", "")
            if st.session_state.get("tab1_comment_pending", False) and _comment_text:
                try:
                    # 1) Append du commentaire sur tous les échantillons NON témoins de la plaque
                    base_df, nb = bulk_update_comments(
                        base_df=base_df,
                        scope="plaque",
                        plaque_id=plaque_selectionnee,
                        comment_text=_comment_text,
                        data_file=DATA_FILE,
                        mode="append",
                        include_temoins=False   # <- ne touche pas aux Tpos/NT/Tvide
                    )

                    # 2) Journalisation dans l'historique (pour l’onglet Historique)
                    try:
                        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        run_id_pl = ""
                        if "summary_run_id" in df_plaque.columns:
                            ids = df_plaque["summary_run_id"].dropna().unique()
                            run_id_pl = str(ids[0]).strip() if len(ids) else ""
                        hist_row = pd.DataFrame([{
                            "date_heure": now,
                            "type": "tab1_flags",
                            "nom_fichier": "",
                            "run_id": run_id_pl,
                            "operateur": nom_prenom or "",
                            "plaque_id": plaque_selectionnee,
                            "details": _comment_text
                        }])
                        if os.path.exists(HISTO_FILE):
                            df_h = pd.read_csv(HISTO_FILE)
                            for c in hist_row.columns:
                                if c not in df_h.columns:
                                    df_h[c] = ""
                            df_h = pd.concat([df_h, hist_row], ignore_index=True)
                        else:
                            df_h = hist_row
                        df_h.to_csv(HISTO_FILE, index=False)
                    except Exception as e:
                        st.warning(f"Traçabilité (Historique) non écrite : {e}")

                    # 3) Fin : état + feedback
                    st.session_state["tab1_comment_applied"] = True
                    st.session_state["tab1_comment_pending"] = False
                    st.toast(f"🧾 Réponses du questionnaire enregistrées dans les commentaires ({nb} lignes, hors témoins).")
                except Exception as e:
                    st.warning(f"Impossible d'appliquer le commentaire : {e}")

            # --- Témoins : filtrage & affichage (➡️ À L’INTÉRIEUR du chargement TSV)
            temoin_pattern = r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide"
            mask_temoin = df_plaque["sample_id"].str.contains(temoin_pattern, case=False, na=False)
            temoin_df = df_plaque.loc[mask_temoin].copy()

            st.markdown("### 🧪 Statistiques des témoins pour la plaque sélectionnée")
            if not temoin_df.empty:
                # Colonnes numériques
                for col in ["summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"]:
                    temoin_df[col] = pd.to_numeric(temoin_df[col], errors="coerce")

                witness_specs = [
                    ("TposH3", "TposH3", "3C.2a1b.2a.2a"),
                    ("TposH1", "TposH1", "6B.1A.5a.2"),
                    ("TposB",  "TposB",  "B"),
                    ("NT1",    "NT1",    None),
                    ("NT2",    "NT2",    None),
                    ("NT3",    "NT3",    None),]

                col1, col2, col3, col4, col5, col6 = st.columns(6)
                cols = [col1, col2, col3, col4, col5, col6]
                if "tpos_lot_choice" not in st.session_state:
                    st.session_state["tpos_lot_choice"] = {}  # (plaque, pattern) -> sample_id choisi

                for (title, pattern, attendu), col in zip(witness_specs, cols):
                    # Filtrage de base sur le nom
                    base_mask = temoin_df["sample_id"].astype(str).str.contains(pattern, case=False, na=False)

                    # ── Cas Tpos* : filtrage par EPIISL attendu + gestion propre des doublons ──
                    if pattern in ("TposH3", "TposH1", "TposB"):
                        ref_attendue = definitions_flu.expected_ref_for_temoin(pattern)
                        mask = base_mask
                        if ref_attendue:
                            srid = temoin_df["summary_reference_id"].astype(str).str.strip()
                            mask = mask & srid.eq(str(ref_attendue).strip())

                        df_t = temoin_df.loc[mask].copy()

                        # 🔧 éliminer les lignes “parasites” (aucune donnée S4/S6)
                        df_t = df_t[~(df_t["summary_consensus_perccoverage_S4"].isna() &
                                      df_t["summary_consensus_perccoverage_S6"].isna())]

                        # 🧹 sécurité: garder 1 seule ligne par sample_id (au cas où)
                        df_t = df_t.sort_index().drop_duplicates(subset=["sample_id"], keep="last")

                        # Normalisation anti “fantômes” (espaces)
                        df_t["sample_id"] = df_t["sample_id"].astype(str).str.strip()
                        df_t["summary_reference_id"] = df_t["summary_reference_id"].astype(str).str.strip()

                        if df_t.empty:
                            col.info(f"❓ {title} non trouvé")
                        else:
                            # 🛡️ Détection de vrais doublons après filtre par EPIISL
                            dups = df_t["sample_id"].value_counts()
                            bad_ids = dups[dups > 1].index.tolist()

                            if bad_ids:
                                st.error(
                                    f"Doublons détectés pour {title} : {', '.join(map(str, bad_ids))}. "
                                    "Sélectionnez les lignes à retenir."
                                )
                                dup_rows = df_t[df_t["sample_id"].isin(bad_ids)].copy()
                                # libellé unique par ligne pour éviter de tout re-sélectionner
                                dup_rows["_rowid_"] = dup_rows.index.astype(str)
                                dup_rows["_row_label"] = (
                                    dup_rows["sample_id"].astype(str)
                                    + " · " + dup_rows.get("plaque_id", "").astype(str)
                                    + " · #" + dup_rows["_rowid_"]
                                )
                                picked_ids = st.multiselect(
                                    "Lignes à retenir (1 ou 2 possibles)",
                                    options=dup_rows["_rowid_"].tolist(),
                                    default=[dup_rows["_rowid_"].iloc[0]],
                                    format_func=lambda rid: dup_rows.loc[dup_rows["_rowid_"] == rid, "_row_label"].iloc[0]
                                )
                                if not picked_ids:
                                    st.stop()
                                keep_dup = dup_rows[dup_rows["_rowid_"].isin(picked_ids)].drop(columns=["_rowid_", "_row_label"])

                                base_part = df_t[~df_t["sample_id"].isin(bad_ids)]
                                df_t_uniq = pd.concat([base_part, keep_dup], ignore_index=True).reset_index(drop=True)
                            else:
                                # pas de doublon réel → on garde 1 ligne par sample_id (sécurité)
                                df_t_uniq = df_t.drop_duplicates(subset=["sample_id"], keep="first").reset_index(drop=True)

                            # Choix du 'lot en cours' (s’il n’y a qu’un Tpos, une seule option)
                            sids = df_t_uniq["sample_id"].tolist()
                            _slug = re.sub(r"\W+", "_", f"{plaque_selectionnee}_{pattern}").lower()
                            key_choice = (plaque_selectionnee, pattern)
                            default_sid = st.session_state["tpos_lot_choice"].get(key_choice, sids[0])
                            chosen_sid = col.radio(
                                f"Lot en cours — {title}",
                                options=sids,
                                index=sids.index(default_sid) if default_sid in sids else 0,
                                key=f"lot_choice_{_slug}",
                                horizontal=True
                            )
                            st.session_state["tpos_lot_choice"][key_choice] = chosen_sid

                            # Une tuile par Tpos unique (empilées si duplicat réel)
                            # Lot associé à chaque tuile (sur la plaque courante)
                            # On garde l’ordre d’itération de df_t_uniq
                            try:
                                lots_line, stages_line = definitions_flu.extract_lot_and_stage_for_temoin(df_t_uniq, pattern)
                            except Exception:
                                lots_line = [None] * len(df_t_uniq)
                            # Historique global pour ce témoin (base_df)
                            try:
                                sub_hist = base_df[base_df["sample_id"].astype(str).str.contains(pattern, case=False, na=False)].copy()
                                lots_hist, _ = definitions_flu.extract_lot_and_stage_for_temoin(sub_hist, pattern)
                                lot_map = dict(zip(sub_hist["sample_id"].astype(str), lots_hist))
                            except Exception:
                                lot_map = {}

                            for i, (_, row) in enumerate(df_t_uniq.iterrows()):
                                sid_str  = str(row["sample_id"])
                                # priorité à l’historique global ; fallback sur le calcul local (df_t_uniq)
                                lot_here = lot_map.get(sid_str, (lots_line[i] if i < len(lots_line) else None))
                                lot_lbl = "—" if (lot_here is None or str(lot_here).strip() == "") else str(lot_here)
                                clade_val = row.get("summary_vcf_coinf01match", "")
                                clade = str(clade_val).strip() if clade_val and str(clade_val).strip() else "—"
                                s4 = pd.to_numeric(row.get("summary_consensus_perccoverage_S4", None), errors="coerce")
                                s6 = pd.to_numeric(row.get("summary_consensus_perccoverage_S6", None), errors="coerce")
                                s4 = 0.0 if pd.isna(s4) else float(s4)
                                s6 = 0.0 if pd.isna(s6) else float(s6)
                                ok = (attendu is None) or (attendu in clade)
                                icon = "🧪" if ok else "⚠️"
                                is_current = (row["sample_id"] == chosen_sid)
                                ribbon   = "Lot en cours" if is_current else "Période probatoire"
                                bg_color = "#e6ffed" if is_current else "#f6f7fb"
                                border   = "#2ecc71" if is_current else "#bdc3c7"

                                col.markdown(
                                    f"""
                                    <div style="background:{bg_color}; border:2px solid {border};
                                                border-radius:12px; padding:12px; margin-bottom:10px;">
                                      <div style='display:flex; justify-content:space-between; align-items:center;'>
                                        <div style='font-size:22px; font-weight:bold;'>{icon} {title}</div>
                                        <span style="font-size:12px; padding:2px 8px; border-radius:999px; background:#fff; border:1px solid {border};">{ribbon}</span>
                                      </div>
                                      <div style='font-size:14px; color:#555; margin-top:4px;'>ID : <b>{row['sample_id']}</b></div>
                                      <div style='font-size:14px; color:#555; margin-top:2px;'>Lot associé : <b>{lot_lbl}</b></div>
                                      <div style="font-size:18px; line-height:1.6">
                                        S4 : <b>{s4:.1f}%</b><br>
                                        S6 : <b>{s6:.1f}%</b><br>
                                        <span style='font-size:18px; font-weight:bold; color:#0066cc;'>{clade}</span>
                                      </div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )


                            # === ➕ Nouveau lot (Tpos uniquement) ===
                            with col:
                                with st.expander("➕ Nouveau lot", expanded=False):
                                    c1, c2, c3 = st.columns([1.1, 1.1, 1.6])
                                    with c1:
                                        lot_number = st.text_input(
                                            "Numéro du **nouveau** lot",
                                            value="", placeholder="ex: 2345",
                                            key=f"lotnum_{_slug}"
                                        )
                                    with c2:
                                        scope = st.radio(
                                            "Portée",
                                            ["Plaque", "Run"],
                                            key=f"scope_{_slug}",
                                            horizontal=True
                                        )
                                    with c3:
                                        note = st.text_input(
                                            "Note (optionnel)",
                                            value="", placeholder="ex: contrôle qualité OK",
                                            key=f"note_{_slug}"
                                        )

                                    run_sel = None
                                    if scope == "Run" and "summary_run_id" in df_t_uniq.columns:
                                        runs_all = sorted(df_t_uniq["summary_run_id"].dropna().unique().tolist())
                                        run_sel = st.selectbox(
                                            "Run cible",
                                            options=runs_all or ["—"],
                                            index=0,
                                            key=f"run_{_slug}"
                                        )

                                    # Proposer les lots déjà connus pour ce témoin
                                    lots_connus = definitions_flu.known_lots(base_df, pattern)  # base_df = dataframe global
                                    lot_pick = st.selectbox(
                                        "Lots connus",
                                        options=["— (saisir manuellement)"] + lots_connus,
                                        index=0,
                                        key=f"lotpick_{_slug}",
                                    )

                                    # ✅ Toujours afficher "Appliquer à" (Plaque & Run)
                                    apply_to = st.radio(
                                        "Appliquer à",
                                        ["Période probatoire", "Lot en cours"],
                                        index=0,  # défaut = probatoire
                                        key=f"applyto_{_slug}",
                                        horizontal=True
                                    )

                                    # ------------ helpers locaux ------------
                                    def _build_scope_df(scope: str) -> pd.DataFrame:
                                        if scope == "Run" and run_sel and run_sel != "—":
                                            df = base_df.copy()
                                            m = df["sample_id"].astype(str).str.contains(pattern, case=False, na=False)
                                            if "summary_run_id" in df.columns:
                                                m = m & df["summary_run_id"].astype(str).eq(str(run_sel))
                                            sub = df.loc[m].copy()
                                            # ≥ plaque sélectionnée dans ce run
                                            def _idx(x):
                                                m2 = re.search(r"(\d+)", str(x)); return int(m2.group(1)) if m2 else None
                                            start_idx = _idx(plaque_selectionnee)
                                            if start_idx is not None and "plaque_id" in sub.columns:
                                                sub = sub[sub["plaque_id"].map(_idx).apply(lambda i: i is not None and i >= start_idx)]
                                            return sub
                                        else:
                                            return df_t_uniq.copy()

                                    def _mirror_to_current_tsv(pl: str, sid: str, new_comment: str):
                                        """Miroir en mémoire : met à jour aussi new_data_filtered pour le *même TSV*."""
                                        try:
                                            df_live = globals().get("new_data_filtered", None)
                                            if isinstance(df_live, pd.DataFrame):
                                                mm = (
                                                    df_live.get("plaque_id", pd.Series(dtype=str)).astype(str).eq(str(pl)) &
                                                    df_live.get("sample_id", pd.Series(dtype=str)).astype(str).eq(str(sid))
                                                )
                                                if mm.any():
                                                    df_live.loc[mm, "commentaire"] = new_comment
                                        except Exception as e:
                                            st.info(f"Note: mise à jour en mémoire non appliquée ({e}).")
                                    # ----------------------------------------

                                    # === 💾 Enregistrer (nouveau lot) ===
                                    if st.button("💾 Enregistrer", key=f"save_newlot_{_slug}", type="primary", use_container_width=True):
                                        # Choisir l'ID du lot: champ texte prioritaire, sinon le selectbox
                                        lot_number_clean = str(lot_number).strip()
                                        chosen_lot_id = lot_number_clean or (lot_pick if not lot_pick.startswith("—") else "")

                                        if not chosen_lot_id:
                                            st.warning("Merci d’indiquer le numéro du lot (champ texte ou liste).")
                                            st.stop()
                                        else:
                                            sub_scoped = _build_scope_df(scope)
                                            if sub_scoped.empty:
                                                st.warning("Aucune ligne témoin dans cette portée.")
                                            else:
                                                # Déterminer la/les cibles
                                                avail = sub_scoped["sample_id"].astype(str).tolist()
                                                is_duplicate_visible = len(avail) > 1  # pour la plaque courante

                                                if scope == "Run":
                                                    targets_df = sub_scoped  # 🔁 toutes les lignes du run sélectionné
                                                else:
                                                    # portée = Plaque (sélection radio)
                                                    if apply_to == "Lot en cours":
                                                        target_sid = chosen_sid if chosen_sid in avail else (avail[0] if avail else chosen_sid)
                                                    else:
                                                        # Période probatoire = l’autre Tpos s’il existe, sinon fallback = lot en cours
                                                        candidates = [s for s in avail if s != chosen_sid]
                                                        target_sid = candidates[0] if candidates else (avail[0] if avail else chosen_sid)

                                                    xcol2 = _x_order_col(sub_scoped)
                                                    sub_scoped = sub_scoped.sort_values([xcol2, "sample_id"], ascending=[True, True])
                                                    targets_df = sub_scoped.loc[sub_scoped["sample_id"].astype(str) == str(target_sid)]
                                                    if targets_df.empty:
                                                        targets_df = sub_scoped.head(1)
                                                # 🛡 Probatoire unique pour ce témoin dans la portée : retire tout 'nouveau lot <Y>' ≠ choisi
                                                scope_df = _build_scope_df(scope).copy()

                                                def _rm_other_prob(s: str) -> str:
                                                    parts = [p.strip() for p in str(s or "").split(";") if p and p.strip() != ""]
                                                    # supprime tous les 'nouveau lot XYZ' sauf celui qu'on s'apprête à poser
                                                    keep = [p for p in parts if not re.match(rf"(?i)^nouveau\s+lot\s+(?!{re.escape(chosen_lot_id)}$)\S+$", p.strip())]
                                                    return " ; ".join(keep)

                                                mask_scope = (
                                                    base_df["plaque_id"].astype(str).isin(scope_df["plaque_id"].astype(str)) &
                                                    base_df["sample_id"].astype(str).isin(scope_df["sample_id"].astype(str))
                                                )
                                                base_df.loc[mask_scope, "commentaire"] = base_df.loc[mask_scope, "commentaire"].apply(_rm_other_prob)

                                                # Appliquer les tags sur chaque cible
                                                for _, anchor in targets_df.iterrows():
                                                    pl  = str(anchor.get("plaque_id", ""))
                                                    sid = str(anchor.get("sample_id", ""))

                                                    cur_series = base_df.loc[
                                                        (base_df.get("plaque_id","").astype(str) == pl) &
                                                        (base_df.get("sample_id","").astype(str) == sid),
                                                        "commentaire"
                                                    ]
                                                    # --- Construction des tags (sans doublons) ---
                                                    current_comment = cur_series.iloc[0] if len(cur_series) else ""
                                                    parts = [p.strip() for p in str(current_comment).split(";") if p and str(p).strip() != ""]

                                                    lot_tag_new = f"nouveau lot {chosen_lot_id}"
                                                    lot_tag_cur = f"lot en cours {chosen_lot_id}"

                                                    def _ensure(tag: str):
                                                        if not any(tag.lower() == p.lower() for p in parts):
                                                            parts.append(tag)

                                                    # === Règles de tags selon la portée ===
                                                    if scope == "Run":
                                                        if apply_to == "Lot en cours":
                                                            # ➜ uniquement 'lot en cours', et on retire un 'nouveau lot' identique s'il traînait
                                                            parts = [p for p in parts if not re.search(r"(?i)^lot\s*en\s*cours\s+\S+$", p.strip())]
                                                            parts = [p for p in parts if not re.search(rf"(?i)^nouveau\s+lot\s+{re.escape(chosen_lot_id)}$", p.strip())]
                                                            _ensure(lot_tag_cur)

                                                        else:  # "Période probatoire"
                                                            # ➜ uniquement 'nouveau lot', et on retire un 'lot en cours' identique s'il traînait
                                                            parts = [p for p in parts if not re.search(rf"(?i)^lot\s*en\s*cours\s+{re.escape(chosen_lot_id)}$", p.strip())]
                                                            _ensure(lot_tag_new)
                                                    else:
                                                        _ensure(lot_tag_new)
                                                        if (not is_duplicate_visible) or (apply_to == "Lot en cours"):
                                                            # 🧹 retirer tout ancien 'lot en cours XYZ' sur cette ligne avant d'ajouter
                                                            parts = [p for p in parts if not re.search(r"(?i)^lot\s*en\s*cours\s+\S+$", p.strip())]
                                                            _ensure(lot_tag_cur)

                                                    # note éventuelle
                                                    if note and str(note).strip() and not any(str(note).strip().lower() == p.lower() for p in parts):
                                                        parts.append(str(note).strip())

                                                    new_comment = " ; ".join(parts)
                                                    base_df = _write_comment(pl, sid, new_comment)
                                                    _mirror_to_current_tsv(pl, sid, new_comment)
                                                    # 🧾 journal dédié lots témoins
                                                    definitions_flu.log_temoin_lot_event(
                                                        temoin=pattern,
                                                        lot_number=chosen_lot_id,
                                                        scope=scope,
                                                        run_id=str(run_sel or ""),
                                                        plaque_id=str(pl),
                                                        note=str(note or ""),
                                                        operateur=nom_prenom or "",
                                                        action=("en_cours" if apply_to == "Lot en cours" else "nouveau")
                                                    )

                                            # Historique (optionnel)
                                            try:
                                                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                hist_row = pd.DataFrame([{
                                                    "date_heure": now,
                                                    "type": "temoin_lot",
                                                    "nom_fichier": f"{pattern}:{chosen_lot_id}",
                                                    "run_id": str(run_sel or ""),
                                                    "operateur": nom_prenom or ""
                                                }])
                                                if os.path.exists(HISTO_FILE):
                                                    df_h = pd.read_csv(HISTO_FILE)
                                                    df_h = pd.concat([df_h, hist_row], ignore_index=True)
                                                else:
                                                    df_h = hist_row
                                                df_h.to_csv(HISTO_FILE, index=False)
                                            except Exception as e:
                                                st.warning(f"Traçabilité : impossible d'écrire dans {HISTO_FILE} : {e}")

                                            st.success(f"✅ Nouveau lot **{chosen_lot_id}** enregistré pour {pattern} (portée: {scope}).")

                                    # === ✅ Promouvoir (fin de probatoire) ===
                                    if st.button("✅ Promouvoir ce lot (fin de probatoire)", key=f"promote_{_slug}", use_container_width=True):
                                        lot_number_clean = str(lot_number).strip()
                                        promote_lot_id = lot_number_clean or (lot_pick if not lot_pick.startswith("—") else "")

                                        if not promote_lot_id:
                                            st.warning("Merci d’indiquer le numéro du lot à promouvoir (champ texte ou liste).")
                                            st.stop()
                                        else:
                                            sub_scoped = _build_scope_df(scope)
                                            if sub_scoped.empty:
                                                st.warning("Aucune ligne témoin dans cette portée.")
                                            else:
                                                # 👉 Sur la plaque : la/les cibles sont déjà dans sub_scoped (df_t_uniq filtrée)
                                                for _, anchor in sub_scoped.iterrows():
                                                    pl  = str(anchor.get("plaque_id", ""))
                                                    sid = str(anchor.get("sample_id", ""))

                                                    # (1) Poser 'lot en cours <ID>' sur la cible
                                                    cur_series = base_df.loc[
                                                        (base_df.get("plaque_id","").astype(str) == pl) &
                                                        (base_df.get("sample_id","").astype(str) == sid),
                                                        "commentaire"
                                                    ]
                                                    # (1) Poser 'lot en cours <ID>' sur la cible
                                                    current_comment = cur_series.iloc[0] if len(cur_series) else ""
                                                    parts = [p.strip() for p in str(current_comment).split(";") if p and str(p).strip() != ""]
                                                    # 🧹 retire 'nouveau lot <ID>' s'il est présent sur cette même ligne
                                                    parts = [p for p in parts if not re.search(rf"(?i)^nouveau\s+lot\s+{re.escape(promote_lot_id)}$", p.strip())]
                                                    parts = [p for p in parts if not re.search(r"(?i)^lot\s*en\s*cours\s+\S+$", p.strip())]
                                                    cur_tag = f"lot en cours {promote_lot_id}"
                                                    if not any(cur_tag.lower() == p.lower() for p in parts):
                                                        parts.append(cur_tag)
                                                    new_comment = " ; ".join(parts)
                                                    base_df = _write_comment(pl, sid, new_comment)
                                                    _mirror_to_current_tsv(pl, sid, new_comment)
                                                    definitions_flu.log_temoin_lot_event(
                                                        temoin=pattern,
                                                        lot_number=promote_lot_id,
                                                        scope=scope,
                                                        run_id=str(run_sel or ""),
                                                        plaque_id=str(pl),
                                                        operateur=nom_prenom or "",
                                                        action="promotion"
                                                    )

                                                    # (2) Retirer 'lot en cours <...>' de l'AUTRE Tpos de la MÊME PLAQUE et du MÊME témoin
                                                    others = base_df[
                                                        (base_df.get("plaque_id","").astype(str) == pl) &
                                                        (base_df.get("sample_id","").astype(str).str.contains(pattern, case=False, na=False)) &
                                                        (base_df.get("sample_id","").astype(str) != sid)
                                                    ]
                                                    if not others.empty:
                                                        for _, r in others.iterrows():
                                                            pl_o  = str(r.get("plaque_id",""))
                                                            sid_o = str(r.get("sample_id",""))
                                                            comm_o = str(r.get("commentaire",""))
                                                            tokens = [t.strip() for t in comm_o.split(";") if t and t.strip() != ""]
                                                            # supprime tout tag 'lot en cours XYZ' quel que soit XYZ
                                                            tokens = [t for t in tokens if not re.search(r"(?i)^lot\s*en\s*cours\s+\S+", t)]
                                                            new_comm_o = " ; ".join(tokens)
                                                            base_df = _write_comment(pl_o, sid_o, new_comm_o)
                                                            _mirror_to_current_tsv(pl_o, sid_o, new_comm_o)  # ⬅️ miroir mémoire
                                                # Historique (optionnel)
                                                try:
                                                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                                    hist_row = pd.DataFrame([{
                                                        "date_heure": now,
                                                        "type": "temoin_promotion",
                                                        "nom_fichier": f"{pattern}:{promote_lot_id}",
                                                        "run_id": str(run_sel or ""),
                                                        "operateur": nom_prenom or ""
                                                    }])
                                                    if os.path.exists(HISTO_FILE):
                                                        df_h = pd.read_csv(HISTO_FILE)
                                                        df_h = pd.concat([df_h, hist_row], ignore_index=True)
                                                    else:
                                                        df_h = hist_row
                                                    df_h.to_csv(HISTO_FILE, index=False)
                                                except Exception as e:
                                                    st.warning(f"Traçabilité : impossible d'écrire dans {HISTO_FILE} : {e}")
                                                    
                                            st.success(f"✅ Lot **{promote_lot_id}** promu pour {pattern} (portée: {scope}).")

                    # ── Cas NT* : garder UNE seule tuile, pas d’expander
                    if pattern not in ("TposH3","TposH1","TposB"):
                        df_t = temoin_df[base_mask].copy()
                        if df_t.empty:
                            col.info(f"❓ {title} non trouvé")
                        else:
                            clade_val = df_t["summary_vcf_coinf01match"].iloc[0]
                            clade = str(clade_val).strip() if pd.notna(clade_val) and str(clade_val).strip() else "—"
                            s4 = pd.to_numeric(df_t["summary_consensus_perccoverage_S4"].iloc[0], errors="coerce")
                            s6 = pd.to_numeric(df_t["summary_consensus_perccoverage_S6"].iloc[0], errors="coerce")
                            s4 = 0.0 if pd.isna(s4) else float(s4)
                            s6 = 0.0 if pd.isna(s6) else float(s6)
                            ok = (attendu is None) or (attendu in clade)
                            icon = "🧪" if ok else "⚠️"
                            bg_color = "#e6ffed" if ok else "#ffe6e6"
                            border_color = "#2ecc71" if ok else "#e74c3c"

                            col.markdown(
                                f"""
                                <div style="background:{bg_color}; border:2px solid {border_color};
                                            border-radius:12px; padding:12px; margin-bottom:10px;">
                                  <div style='font-size:22px; font-weight:bold; margin-bottom:6px;'>{icon} {title}</div>
                                  <div style="font-size:18px; line-height:1.6">
                                    S4 couverture : <b>{s4:.1f}%</b><br>
                                    S6 couverture : <b>{s6:.1f}%</b><br>
                                    <span style='font-size:18px; font-weight:bold; color:#0066cc;'>{clade}</span>
                                  </div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                    
            else:
                st.info("❌ Aucun témoin détecté pour cette plaque.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)
                
            # =========================
            # 🔻 ÉCHANTILLONS NON TÉMOINS
            # =========================
            temoin_pattern = r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide"
            mask_temoin = df_plaque["sample_id"].str.contains(temoin_pattern, case=False, na=False)
            non_temoin_df = df_plaque.loc[~mask_temoin].copy()

            # Sécuriser les colonnes numériques
            for col_num in ["summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"]:
                if col_num in non_temoin_df.columns:
                    non_temoin_df[col_num] = pd.to_numeric(non_temoin_df[col_num], errors="coerce")

            # =========================
            # ❓ ININTERPRÉTABLES
            # =========================
            st.markdown("### ❓ Échantillons ininterprétables")
            mask_ininterpretable = (
                (non_temoin_df["summary_consensus_perccoverage_S4"] < 90) |
                (non_temoin_df["summary_consensus_perccoverage_S6"] < 90))
            ininterpretable_df = non_temoin_df.loc[mask_ininterpretable].drop_duplicates(subset="sample_id").copy()

            if ininterpretable_df.empty:
                st.info("✅ Aucun échantillon ininterprétable détecté.")
            else:
                # Détails par souche
                df_H3N2 = ininterpretable_df[ininterpretable_df["summary_reference_id"] == "EPIISL129744"]
                df_H1N1 = ininterpretable_df[ininterpretable_df["summary_reference_id"] == "EPIISL200780"]
                df_Bvic = ininterpretable_df[ininterpretable_df["summary_reference_id"] == "EPIISL219327"]
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                col3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                with st.expander("📊 Détails par souche"):
                    for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                        if df_sub.empty:
                            continue
                        st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                        styled = (df_sub[["sample_id", "plaque_id", "summary_run_id","summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"]]
                            .style
                            .set_table_styles([{"selector": "th, td", "props": [("text-align", "center")]}])
                            .set_properties(**{"text-align": "center"})
                            .set_properties(**{"font-weight": "bold"},
                                            subset=["summary_consensus_perccoverage_S4",
                                                    "summary_consensus_perccoverage_S6"]))
                        st.markdown(styled.to_html(escape=False, index=False), unsafe_allow_html=True)

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🧪 DOUBLE POPULATION
            # =========================
            st.markdown("### 🧪 Détection des doubles populations")
            df_dp_h3 = count_double_pop(non_temoin_df, "EPIISL129744").assign(strain="H3N2")
            df_dp_h1 = count_double_pop(non_temoin_df, "EPIISL200780").assign(strain="H1N1")
            df_dp_bv = count_double_pop(non_temoin_df, "EPIISL219327").assign(strain="BVic")

            if df_dp_h3.empty and df_dp_h1.empty and df_dp_bv.empty:
                st.success("✅ Aucun échantillon détecté avec une double population.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_dp_h3)}</b></div>", unsafe_allow_html=True)
                c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_dp_h1)}</b></div>", unsafe_allow_html=True)
                c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_dp_bv)}</b></div>", unsafe_allow_html=True)
                with st.expander("📊 Voir les tableaux détaillés"):
                    for label, df_sub in [("H3N2", df_dp_h3), ("H1N1", df_dp_h1), ("Bvic", df_dp_bv)]:
                        if df_sub.empty:
                            continue
                        st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                        styled_df = (
                            df_sub[["sample_id", "plaque_id", "summary_run_id"]]
                            .style.set_table_styles(
                                [{"selector": "th, td", "props": [("text-align", "center")]}])
                            .set_properties(**{"text-align": "center"}))
                        st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🎯 VARCOUNT ≥ 13
            # =========================
            st.subheader("🧬 Échantillons Varcount (≥13) pour la plaque sélectionnée")
            if "val_varcount" in non_temoin_df.columns and "summary_reference_id" in non_temoin_df.columns:
                non_temoin_df["varcount_num"] = pd.to_numeric(
                    non_temoin_df["val_varcount"].str.extract(r'VARCOUNT(\d+)', expand=False),
                    errors="coerce"
                )
                varcount_df = non_temoin_df[non_temoin_df["varcount_num"] >= 13].copy()

                if varcount_df.empty:
                    st.info("Aucun échantillon avec Varcount ≥ 13.")
                else:
                    df_H3N2 = varcount_df[varcount_df["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = varcount_df[varcount_df["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = varcount_df[varcount_df["summary_reference_id"] == "EPIISL219327"]

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Voir les tableaux détaillés"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_df = (
                                df_sub[["sample_id", "plaque_id", "summary_run_id", "val_varcount"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}])
                                .set_properties(**{"text-align": "center"})
                                .set_properties(**{"font-weight": "bold"}, subset=["val_varcount"]))
                            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.warning("Les colonnes nécessaires pour Varcount sont absentes.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # ❌ SEQ FAILED
            # =========================
            st.markdown("### ❌ Échantillons échoués (Seq failed)")
            if "summary_qc_seqcontrol" in non_temoin_df.columns:
                qc_upper = non_temoin_df["summary_qc_seqcontrol"].astype(str).str.upper()
                failed_df = non_temoin_df[qc_upper.isin(["FAILED", "0"])].copy()
                if failed_df.empty:
                    st.success("✅ Aucun échantillon échoué détecté.")
                else:
                    df_H3N2 = failed_df[failed_df["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = failed_df[failed_df["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = failed_df[failed_df["summary_reference_id"] == "EPIISL219327"]

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Voir les échantillons échoués (par souche)"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_failed_df = (
                                df_sub[["sample_id", "plaque_id", "summary_run_id", "summary_qc_seqcontrol"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}])
                                .set_properties(**{"text-align": "center"})
                                .set_properties(**{"font-weight": "bold"}, subset=["summary_qc_seqcontrol"]))
                            st.markdown(styled_failed_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.warning("Colonne 'summary_qc_seqcontrol' absente.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🧬 SOUCHES SECONDAIRES (> 0.4)
            # =========================
            st.markdown("### 🧬 Souches secondaires (score > 0.4)")

                        # 1) Uploader (persistant)
            comp_file = st.file_uploader(
                "🔍 Charger le fichier de similarité Premap_coverage (.tsv)",
                type=["tsv"],
                key="premap_cov_uploader"
            )

            if comp_file is not None:
                try:
                    similarity_df = pd.read_csv(comp_file, sep="\t", index_col=0)
                    st.session_state["premap_cov"] = similarity_df
                    st.success(f"Fichier Premap_coverage chargé ({similarity_df.shape[0]} x {similarity_df.shape[1]}) et mémorisé.")

                    # 🔹 Ajout à l’historique
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    hist_row = pd.DataFrame([{
                        "date_heure": now,
                        "type": "premap_coverage",
                        "nom_fichier": comp_file.name,
                        "run_id": run_id_for_log,
                        "operateur": nom_prenom or ""
                    }])

                    if os.path.exists(HISTO_FILE):
                        df_h = pd.read_csv(HISTO_FILE)
                        df_h = pd.concat([df_h, hist_row], ignore_index=True)
                    else:
                        df_h = hist_row

                    df_h.to_csv(HISTO_FILE, index=False)

                except Exception as e:
                    st.error(f"❌ Erreur lors du chargement de Premap_coverage : {e}")

            # 2) Récupération mémoire
            similarity_df = st.session_state.get("premap_cov", None)

            if similarity_df is None:
                st.info("📂 Aucun fichier Premap_coverage chargé.")
            else:
                try:
                    refs_exclues = {"EPIISL129744", "EPIISL200780", "EPIISL219327"}
                    sim_f = similarity_df.drop(index=refs_exclues, errors="ignore")

                    # on travaille sur TOUT le run pour construire une base persistante…
                    run_samples = new_data_filtered["sample_id"].unique().tolist()
                    sim_f = sim_f.loc[:, sim_f.columns.intersection(run_samples)].astype(float)

                    found_all = []
                    presence_mask = sim_f > 0.4
                    for sample in sim_f.columns:
                        plaque_id = new_data_filtered.loc[new_data_filtered["sample_id"] == sample, "plaque_id"].iloc[0]

                        if "GRA" in plaque_id:
                            cands = [epi for epi in sim_f.index[presence_mask[sample]] if epi not in ["EPIISL129744", "EPIISL200780"]]
                        elif "GRB" in plaque_id:
                            cands = [epi for epi in sim_f.index[presence_mask[sample]] if epi != "EPIISL219327"]
                        else:
                            cands = sim_f.index[presence_mask[sample]].tolist()

                        scores = sim_f.loc[cands, sample].tolist()
                        for epi, score in zip(cands, scores):
                            found_all.append({
                                "sample_id": sample,
                                "souche_EPIISL": epi,
                                "clade": reference_map.get(epi, epi),
                                "similarity_score": float(score),
                            })

                    if not found_all:
                        enriched_df_all = pd.DataFrame(
                            columns=["sample_id","plaque_id","summary_run_id","summary_reference_id",
                                     "souche_EPIISL","clade","similarity_score"]
                        )
                    else:
                        enriched_df_all = (
                            pd.DataFrame(found_all)
                            .merge(new_data_filtered[["sample_id","plaque_id","summary_run_id"]], on="sample_id", how="left")
                            .merge(new_data_filtered[["sample_id","summary_reference_id"]], on="sample_id", how="left")
                        )

                    # stock persistant pour l’export HTML
                    st.session_state["enriched_df_all"] = enriched_df_all

                    # …mais on n’affiche QUE la plaque sélectionnée
                    enriched_df = enriched_df_all[enriched_df_all["plaque_id"] == plaque_selectionnee].copy()

                    if enriched_df.empty:
                        st.info("✅ Aucune autre souche détectée avec un score > 0.4 pour cette plaque.")
                    else:
                        df_H3N2 = enriched_df[enriched_df["summary_reference_id"] == "EPIISL129744"]
                        df_H1N1 = enriched_df[enriched_df["summary_reference_id"] == "EPIISL200780"]
                        df_Bvic = enriched_df[enriched_df["summary_reference_id"] == "EPIISL219327"]

                        c1, c2, c3 = st.columns(3)
                        c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                        c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                        c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)
                        key_flag = f"_sec_rendered_{plaque_selectionnee}"
                        if not st.session_state.get(key_flag, False):
                            render_table_cmap(df_H3N2, "H3N2", "Reds")
                            render_table_cmap(df_H1N1, "H1N1", "Reds")
                            render_table_cmap(df_Bvic,  "BVic",  "Reds")
                            st.session_state[key_flag] = True

                except Exception as e:
                    st.error(f"❌ Erreur lors du traitement de Premap_coverage : {e}")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🧬 COINFECTIONS / RÉASSORTIMENT
            # =========================
            st.markdown("### 🧬 Coinfections et réassortiment")

            non_temoin_df_filtered = non_temoin_df[
                ~non_temoin_df["summary_vcf_coinf01match"].isin(["positif", "négatif", "TVide"])
            ].copy()

            failed_ids, warning_ids = [], []
            if "summary_bam_verif" in non_temoin_df_filtered.columns:
                valid_df = non_temoin_df_filtered.copy()
                if "val_result" in valid_df.columns:
                    valid_df = valid_df[
                        ~valid_df["val_result"].astype(str).str.upper().isin(["ININT", "REPASSE SEQ FAILED"])
                    ]
                verif = valid_df["summary_bam_verif"].astype(str).str.upper()
                failed_ids = valid_df.loc[verif == "FAILED", "sample_id"].tolist()
                warning_ids = valid_df.loc[verif == "WARNING", "sample_id"].tolist()

                if not failed_ids and not warning_ids:
                    st.success("✅ Pas de coinfections ou réassortiment détectés.")
                else:
                    st.write(f"❌ FAILED : {len(failed_ids)} | ⚠️ WARNING : {len(warning_ids)}")

                    comp_file_verif = st.file_uploader(
                        "🔍 Charger le fichier Premap_verif",
                        type="tsv",
                        key="uploader_similarity_matrix"
                    )
                    if comp_file_verif is not None:
                        try:
                            similarity_df = pd.read_csv(comp_file_verif, sep="\t", index_col=0).astype(float)
                            refs_exclues = {"EPIISL129744","EPIISL200780","EPIISL219327"}
                            filtered_sim = similarity_df.drop(index=refs_exclues, errors="ignore")
                            plaque_samples = new_data_filtered.loc[
                                new_data_filtered["plaque_id"] == plaque_selectionnee, "sample_id"
                            ].unique()
                            filtered_sim = filtered_sim.loc[:, filtered_sim.columns.intersection(plaque_samples)]

                            df_failed  = detect_souches(failed_ids,  filtered_sim, non_temoin_df_filtered)
                            df_warning = detect_souches(warning_ids, filtered_sim, non_temoin_df_filtered)

                            if not df_failed.empty:
                                st.subheader("🔬 Similarité – FAILED")
                                for epi, lbl in [("EPIISL129744", "H3N2"), ("EPIISL200780", "H1N1"), ("EPIISL219327", "BVic")]:
                                    render_table_cmap(df_failed[df_failed["souche_EPIISL"] == epi], lbl, "Reds")

                            if not df_warning.empty:
                                st.subheader("🔬 Similarité – WARNING")
                                for epi, lbl in [("EPIISL129744", "H3N2"), ("EPIISL200780", "H1N1"), ("EPIISL219327", "BVic")]:
                                    render_table_cmap(df_warning[df_warning["souche_EPIISL"] == epi], lbl, "Reds")
                        except Exception as e:
                            st.error(f"❌ Erreur traitement Premap_verif : {e}")
            else:
                st.warning("⚠️ Colonne 'summary_bam_verif' absente, pas de coinfections/réassortiment.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🔄 COINFECTION INTER-CLADE (IQR > 0)
            # =========================
            st.subheader("🧬 Coinfection inter-clade")
            if "summary_vcf_coinf02iqr" in non_temoin_df.columns:
                non_temoin_df["summary_vcf_coinf02iqr"] = pd.to_numeric(
                    non_temoin_df["summary_vcf_coinf02iqr"], errors="coerce"
                )
                interclade_samples = non_temoin_df[non_temoin_df["summary_vcf_coinf02iqr"] > 0].drop_duplicates("sample_id")

                st.markdown(
                    f"<div style='font-size:30px;'>Nombre d’échantillons avec IQR > 0 : "
                    f"<b>{len(interclade_samples)}</b></div>",
                    unsafe_allow_html=True
                )

                if not interclade_samples.empty and "summary_reference_id" in interclade_samples.columns:
                    df_H3N2 = interclade_samples[interclade_samples["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = interclade_samples[interclade_samples["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = interclade_samples[interclade_samples["summary_reference_id"] == "EPIISL219327"]

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Voir les tableaux détaillés"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_df = (
                                df_sub[["sample_id", "plaque_id", "summary_reference_id", "summary_vcf_coinf02iqr"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}]
                                )
                                .set_properties(**{"text-align": "center"})
                                .background_gradient(subset=["summary_vcf_coinf02iqr"], cmap="Reds")
                                .set_properties(**{"font-weight": "bold"}, subset=["summary_vcf_coinf02iqr"])
                            )
                            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
                else:
                    st.info("✅ Aucun échantillon avec IQR > 0 détecté.")
            else:
                st.warning("⚠️ La colonne 'summary_vcf_coinf02iqr' est absente.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # ✅ AVIS BIO / POI
            # =========================
            st.markdown("### 📌 Avis Bio (POI)")
            if "val_avisbio" in df_plaque.columns:
                avisbio_df = df_plaque[df_plaque["val_avisbio"] == "AVISBIO_POI"].copy()
                if avisbio_df.empty:
                    st.info("Aucun échantillon ‘AVISBIO_POI’ sur cette plaque.")
                else:
                    c1, c2, c3 = st.columns(3)
                    df_H3N2 = avisbio_df[avisbio_df["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = avisbio_df[avisbio_df["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = avisbio_df[avisbio_df["summary_reference_id"] == "EPIISL219327"]
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Voir les tableaux détaillés (POI)"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_df = (
                                df_sub[["sample_id", "plaque_id", "val_avisbio", "val_poi"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}]
                                )
                                .set_properties(**{"text-align": "center"})
                                .set_properties(**{"font-weight": "bold"}, subset=["val_avisbio"])
                            )
                            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.warning("Colonne 'val_avisbio' absente du fichier.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🔍 FRAMESHIFTS
            # =========================
            st.markdown("### 🔍 Échantillons avec *FrameShifts* détectés")
            if "nextclade_frameShifts" in df_plaque.columns:
                frameshift_df = df_plaque[
                    df_plaque["nextclade_frameShifts"].notna() &
                    (df_plaque["nextclade_frameShifts"].astype(str).str.strip() != "")
                ].copy()
                if frameshift_df.empty:
                    st.info("Aucun frameshift détecté sur cette plaque.")
                else:
                    c1, c2, c3 = st.columns(3)
                    df_H3N2 = frameshift_df[frameshift_df["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = frameshift_df[frameshift_df["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = frameshift_df[frameshift_df["summary_reference_id"] == "EPIISL219327"]
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Détails (Frameshifts)"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_df = (
                                df_sub[["sample_id", "nextclade_frameShifts"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}]
                                )
                                .set_properties(**{"text-align": "center"})
                                .set_properties(**{"font-weight": "bold"}, subset=["nextclade_frameShifts"])
                            )
                            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.warning("Colonne 'nextclade_frameShifts' absente du fichier.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # ➕ INSERTIONS
            # =========================
            st.markdown("### 🧬 Échantillons avec *Insertions* détectées")
            if "val_insertions" in df_plaque.columns:
                insertions_df = df_plaque.dropna(subset=["val_insertions"]).copy()
                if insertions_df.empty:
                    st.info("Aucune insertion détectée sur cette plaque.")
                else:
                    df_H3N2 = insertions_df[insertions_df["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = insertions_df[insertions_df["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = insertions_df[insertions_df["summary_reference_id"] == "EPIISL219327"]
                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Voir les tableaux détaillés (Insertions)"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_df = (
                                df_sub[["sample_id", "val_insertions"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}]
                                )
                                .set_properties(**{"text-align": "center"})
                                .set_properties(**{"font-weight": "bold"}, subset=["val_insertions"])
                            )
                            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.warning("Colonne 'val_insertions' absente du fichier.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 🧬 QC NEXTCLADE ≠ good
            # =========================
            st.markdown("### 🧬 Suivi QC Nextclade")
            if "nextclade_qc_overallStatus" in df_plaque.columns:
                qc_status_df = df_plaque[
                    df_plaque["nextclade_qc_overallStatus"].notna() &
                    (df_plaque["nextclade_qc_overallStatus"].astype(str).str.lower() != "good")
                ].copy()
                if qc_status_df.empty:
                    st.info("Tous les échantillons ont un statut QC normal ('good') ou vide.")
                else:
                    df_H3N2 = qc_status_df[qc_status_df["summary_reference_id"] == "EPIISL129744"]
                    df_H1N1 = qc_status_df[qc_status_df["summary_reference_id"] == "EPIISL200780"]
                    df_Bvic = qc_status_df[qc_status_df["summary_reference_id"] == "EPIISL219327"]

                    c1, c2, c3 = st.columns(3)
                    c1.markdown(f"<div class='big-number'>🧬 H3N2 : <b>{len(df_H3N2)}</b></div>", unsafe_allow_html=True)
                    c2.markdown(f"<div class='big-number'>🧬 H1N1 : <b>{len(df_H1N1)}</b></div>", unsafe_allow_html=True)
                    c3.markdown(f"<div class='big-number'>🧬 Bvic : <b>{len(df_Bvic)}</b></div>", unsafe_allow_html=True)

                    with st.expander("📊 Voir les tableaux détaillés (QC Nextclade)"):
                        for label, df_sub in [("H3N2", df_H3N2), ("H1N1", df_H1N1), ("Bvic", df_Bvic)]:
                            if df_sub.empty:
                                continue
                            st.markdown(f"#### {label} – {len(df_sub)} échantillons")
                            styled_df = (
                                df_sub[["sample_id", "nextclade_qc_overallStatus"]]
                                .style.set_table_styles(
                                    [{"selector": "th, td", "props": [("text-align", "center")]}]
                                )
                                .set_properties(**{"text-align": "center"})
                                .set_properties(**{"font-weight": "bold"}, subset=["nextclade_qc_overallStatus"])
                            )
                            st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
            else:
                st.warning("Colonne 'nextclade_qc_overallStatus' absente du fichier.")

            st.markdown("<hr class='separator-hr'>", unsafe_allow_html=True)

            # =========================
            # 📊 CAMEMBERT (hors témoins)
            # =========================
            st.markdown("#### 📊 Répartition des Résultats (hors témoins)")
            non_temoin_plaque_df = non_temoin_df.copy()
            if not non_temoin_plaque_df.empty and "summary_vcf_coinf01match" in non_temoin_plaque_df.columns:
                counts = non_temoin_plaque_df["summary_vcf_coinf01match"].value_counts()
                labels = counts.index.astype(str).tolist()
                values = counts.values.tolist()
                total = float(sum(values)) if sum(values) else 1.0
                percents = [v / total for v in values]
                SMALL = 0.06
                textpos = ["outside" if p < SMALL else "inside" for p in percents]

                # wrap des labels
                labels_wrapped = [_wrap_label(lbl, width=14) for lbl in labels]

                fig_pie = go.Figure(
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
                fig_pie.update_layout(
                    margin=dict(t=10, r=40, b=120, l=40),
                    height=460,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("⚠️ Aucun échantillon non témoin détecté pour cette plaque.")

            if st.button("📄 Exporter la page en HTML"):
                sections = []

                # 1) Statistiques globales du run
                total_count = len(new_data_filtered)
                total_samples_html = f"""
                  <p><strong>Total d’échantillons dans le run :</strong> {total_count}</p>
                """
                register_section_html(
                    title="📦 Statistiques du run",
                    html=total_samples_html,
                    counts=None,
                    plaque="all",
                    sections=sections
                )

                # 2) Boucle plaques
                for pl in plaques_disponibles:
                    df_pl = new_data_filtered[new_data_filtered["plaque_id"] == pl]

                    # Détail plaque
                    plaque_count = len(df_pl)
                    samples_info_html = f"""
                      <p><strong>Nombre d’échantillons dans cette plaque :</strong> {plaque_count}</p>
                    """
                    register_section_html(
                        title=f"📦 Détail de la plaque – {pl}",
                        html=samples_info_html,
                        counts=None,
                        plaque=pl,
                        sections=sections
                    )

                    # Témoins (tuiles)
                    temoin_pattern = r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide"
                    temoin_df = df_pl[df_pl["sample_id"].astype(str).str.contains(temoin_pattern, case=False, na=False)].copy()

                    tiles_html = definitions_flu.build_temoin_tiles_html(
                        temoin_df,
                        df_history=base_df   # ← historique complet pour déterminer lot/stage
                    )

                    register_section_html(
                        title="🧪 Statistiques des témoins pour la plaque sélectionnée",
                        html=tiles_html or "<p><em>Aucun témoin détecté pour cette plaque.</em></p>",
                        counts=None,           # force le rendu "HTML pur" du template
                        plaque=pl,
                        sections=sections
                    )


                    # 1) Questionnaire
                    controls_full = {
                        "Nom et Prénom":         nom_prenom,
                        "Date de l'analyse":     date_analyse.strftime("%Y-%m-%d") if date_analyse else None,
                        "Métriques du run":      metriques_run,
                        "Dosage Qubit":          dosage_qbit,
                        "Changement de lot":     changement_lot,
                        "Nouveau lot de témoin": details_lot,
                        "Nouvelle dilution":     nouvelle_dilution,
                        "Nouveau lot des amorces": details_dilution,
                        "Problèmes techniques":  problemes_tech,
                        "Expliquez" : problemes_tech_ex,
                        "Non-conf Kalilab":      non_conf,
                        "Numéro de la NC":       num_nc,
                        "Justification NC":      justification_nc
                    }
                    controls = {k: v for k, v in controls_full.items() if v not in (None, "", "Sélectionnez")}

                    # Filtrer le DataFrame global pour cette plaque
                    df_pl = new_data_filtered[new_data_filtered["plaque_id"] == pl]
                    non_temoin_df = df_pl[~df_pl["sample_id"].str.contains(temoin_pattern, case=False, na=False)].copy()
                    
                    # 1) Ininterprétables
                    # Reconstruire correctement le DataFrame des ininterprétables (comme dans build_ininterpretable_html)
                    df_tmp = non_temoin_df.copy()
                    df_tmp["summary_consensus_perccoverage_S4"] = pd.to_numeric(df_tmp["summary_consensus_perccoverage_S4"], errors="coerce")
                    df_tmp["summary_consensus_perccoverage_S6"] = pd.to_numeric(df_tmp["summary_consensus_perccoverage_S6"], errors="coerce")

                    mask = (
                        (df_tmp["summary_consensus_perccoverage_S4"] < 90) |
                        (df_tmp["summary_consensus_perccoverage_S6"] < 90)
                    )
                    ininterpretable_df = df_tmp.loc[mask].drop_duplicates(subset="sample_id")

                    # Générer le HTML
                    html_int = build_ininterpretable_html(ininterpretable_df)
                    if not html_int:
                        html_int = "<p><em>Aucun échantillon ininterprétable détecté.</em></p>"

                    register_section_html(
                        title="❓ Échantillons Ininterprétables",
                        html=html_int,
                        counts=make_counts(ininterpretable_df),
                        plaque=pl,
                        sections=sections
                    )

                    # 2) Double pop
                    df_dp_h3 = count_double_pop(non_temoin_df, "EPIISL129744").assign(strain="H3N2")
                    df_dp_h1 = count_double_pop(non_temoin_df, "EPIISL200780").assign(strain="H1N1")
                    df_dp_bv = count_double_pop(non_temoin_df, "EPIISL219327").assign(strain="BVic")
                    df_double_pop = pd.concat([df_dp_h3, df_dp_h1, df_dp_bv], ignore_index=True)
                    register_section(
                        title="🧪 Échantillons Double population",
                        df=df_double_pop,
                        cols=["strain", "sample_id", "plaque_id", "summary_run_id", "summary_vcf_dpcount"],
                        empty_msg="Aucun double pop détecté.",
                        counts=make_counts(df_double_pop),
                        plaque=pl,
                        sections=sections
                    )

                    # 3) Varcount ≥13
                    non_temoin_df["varcount_num"] = pd.to_numeric(
                        non_temoin_df["val_varcount"].str.extract(r'VARCOUNT(\d+)', expand=False),
                        errors="coerce"
                    )
                    varcount_df = non_temoin_df[non_temoin_df["varcount_num"] >= 13]
                    register_section(
                        title="🎯 Échantillons Varcount ≥13",
                        df=varcount_df,
                        cols=["sample_id", "plaque_id", "summary_run_id", "varcount_num"],
                        empty_msg="Aucun varcount ≥13.",
                        counts=make_counts(varcount_df),
                        plaque=pl,
                        sections=sections
                    )

                    # — 1) Seq failed 
                    failed_df = non_temoin_df[
                        non_temoin_df["summary_qc_seqcontrol"].str.upper().isin(["FAILED", "0"])
                    ]

                    register_section(
                        title="❌ Échantillons échoués (Seq failed)",
                        df=failed_df,
                        cols=["sample_id", "plaque_id", "summary_run_id", "summary_qc_seqcontrol"],
                        empty_msg="Aucun échantillon échoué",
                        counts=make_counts(failed_df),
                        plaque=pl,
                        sections=sections
                    )

                    # — 2) Souches secondaires (HTML-only) — PERSISTANT SUR TOUT LE RUN
                    df_sec_pl = st.session_state.get(
                        "enriched_df_all",
                        pd.DataFrame(columns=[
                            "sample_id", "plaque_id", "summary_run_id", "summary_reference_id",
                            "souche_EPIISL", "clade", "similarity_score"
                        ])
                    )

                    df_sec_pl = df_sec_pl[df_sec_pl["plaque_id"] == pl]
                    html_sec  = build_secondary_html(df_sec_pl) or "<p><em>Aucune souche secondaire détectée.</em></p>"

                    register_section_html(
                        title="🧬 Souches secondaires",
                        html=html_sec,
                        counts=make_counts(df_sec_pl),   # ok si make_counts sait gérer vide
                        plaque=pl,
                        sections=sections)

                    # — 3) Coinfections & réassortiments
                    if "summary_bam_verif" in non_temoin_df.columns:
                        verif = non_temoin_df["summary_bam_verif"].str.upper()
                        coinf_df = non_temoin_df[verif.isin(["FAILED", "WARNING"])].copy()
                    else:
                        coinf_df = pd.DataFrame()

                    register_section(
                        title="🧬 Coinfections & Réassortissements",
                        df=coinf_df,
                        cols=["sample_id", "plaque_id", "summary_reference_id", "summary_bam_verif"],
                        empty_msg="Aucune coinfection ni réassortiment détecté.",
                        counts=make_counts(coinf_df),
                        plaque=pl,
                        sections=sections
                    )

                    # — 4) Coinfection inter-clade (HTML-only)
                    if "summary_vcf_coinf02iqr" in non_temoin_df.columns:
                        non_temoin_df["summary_vcf_coinf02iqr"] = pd.to_numeric(
                            non_temoin_df["summary_vcf_coinf02iqr"], errors="coerce"
                        )
                        df_inter = non_temoin_df[
                            non_temoin_df["summary_vcf_coinf02iqr"] > 0
                        ].drop_duplicates(subset="sample_id")

                        if df_inter.empty:
                            html_inter = "<p><em>Aucune coinfection inter-clade détectée.</em></p>"
                        else:
                            html_inter = build_interclade_html(df_inter)

                        register_section_html(
                            title="🔄 Coinfection inter-clade",
                            html=html_inter,
                            counts=make_counts(df_inter),
                            plaque=pl,
                            sections=sections
                        )

                    # — 5) Avis Bio / POI
                    avispoi_pl = non_temoin_df[non_temoin_df["val_avisbio"] == "AVISBIO_POI"]
                    register_section(
                        title="✅ Avis Bio POI",
                        df=avispoi_pl,
                        cols=["sample_id","plaque_id","val_avisbio","val_poi"],
                        empty_msg="Aucun avis Bio POI",
                        counts=make_counts(avispoi_pl),
                        plaque=pl,
                        sections=sections
                    )

                    # — 6) Frameshifts
                    fs_pl = non_temoin_df[non_temoin_df["nextclade_frameShifts"].notna() & (non_temoin_df["nextclade_frameShifts"].str.strip() != "")]
                    register_section(
                        title="🔍 Frameshifts détectés",
                        df=fs_pl,
                        cols=["sample_id","plaque_id","nextclade_frameShifts"],
                        empty_msg="Aucun frameshift détecté",
                        counts=make_counts(fs_pl),
                        plaque=pl,
                        sections=sections
                    )

                    # — 7) Insertions
                    ins_pl = non_temoin_df.dropna(subset=["val_insertions"])
                    register_section(
                        title="➕ Insertions détectées",
                        df=ins_pl,
                        cols=["sample_id","plaque_id","val_insertions"],
                        empty_msg="Aucune insertion détectée",
                        counts=make_counts(ins_pl),
                        plaque=pl, 
                        sections=sections
                    )

                    # — 8) QC Nextclade
                    qc_pl = non_temoin_df[
                        non_temoin_df["nextclade_qc_overallStatus"].notna() &
                        (non_temoin_df["nextclade_qc_overallStatus"].str.lower() != "good")
                    ]
                    register_section(
                        title="⚠️ QC Nextclade",
                        df=qc_pl,
                        cols=["sample_id","plaque_id","nextclade_qc_overallStatus"],
                        empty_msg="Tous statuts QC = good",
                        counts=make_counts(qc_pl),
                        plaque=pl,
                        sections=sections
                    )

                    # — 9) Camembert de répartition 
                    plot_div_pl = build_pie_div(non_temoin_df)
                    if plot_div_pl:
                        register_section_html(
                            title=f"📊 Répartition des Résultats",
                            html=plot_div_pl,
                            counts=None,
                            plaque=pl,
                            sections=sections
                        )
                    
                   # Après avoir construit new_data_filtered et sélectionné plaque_selectionnee
                    plaques_disponibles = sorted(new_data_filtered["plaque_id"].dropna().unique())
                    plaque_selectionnee = plaque_selectionnee  # déjà définie par votre st.selectbox

                # 5) Générer et proposer le téléchargement
                # On filtre les données non témoins de la plaque sélectionnée
                non_temoin_df_plaque = new_data_filtered[
                    (new_data_filtered["plaque_id"] == plaque_selectionnee) &
                    (~new_data_filtered["sample_id"].str.contains(r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide", case=False, na=False))
                ]

                # On génère le graphique Plotly HTML
                plot_div_html = build_pie_div(non_temoin_df_plaque)
                
                # 🔹 Extraire les run_id pour la plaque sélectionnée
                run_ids = new_data_filtered.loc[
                    new_data_filtered["plaque_id"] == plaque_selectionnee, "summary_run_id"
                ].dropna().unique()

                # ✅ récupérer une seule valeur propre (ou "inconnu")
                run_id = str(run_ids[0]).strip() if len(run_ids) > 0 else "inconnu"

                # ✅ Créer le rapport avec le run_id
                html_report = make_report_html(
                    controls=controls,
                    plot_div=plot_div_html,
                    sections=sections,
                    plaques=plaques_disponibles,
                    plaque_selected=plaque_selectionnee,
                    counts=None,
                    run_id=run_id
                )

                # 🔹 Obtenir la date du jour
                today_str = datetime.now().strftime("%Y-%m-%d")

                # ✅ Nom du fichier de sortie propre
                run_id_str = run_id.replace(" ", "_").replace("/", "-")
                file_name = f"{run_id_str}_{today_str}.html"

                # 🔻 Télécharger
                st.download_button(
                    label="⬇️ Télécharger le rapport complet",
                    data=html_report,
                    file_name=file_name,
                    mime="text/html")
        else:
            st.info("📂 Veuillez charger un fichier TSV pour activer l’analyse par plaque.")

with tab2:
    page_plaque()
    st.header("🧩 Plan de plaque")

    # récup des données du 1er TSV (déjà stockées dans tab1)
    new_df   = st.session_state.get("new_data_filtered")
    plaques  = st.session_state.get("plaques_disponibles")

    # ==============================
    # PLAQUES 96 (2 cartes / ligne)
    # ==============================
    st.subheader("🔶 Plaques 96 individuelles")

    if new_df is None or plaques is None:
        st.info("Charge d’abord le 1er TSV dans l’onglet « Aide à la confirmation » pour afficher les plans 96.")
    else:
        cols = st.columns(2)
        for idx, pl in enumerate(plaques):
            with cols[idx % 2]:
                df_pl = new_df[new_df["plaque_id"] == pl].copy()
                st.markdown(
                    "<div style='background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:16px;margin-bottom:16px;'>",
                    unsafe_allow_html=True
                )
                st.markdown(f"**Plaque : {pl}**")
                fig96 = _make_plate_fig_96(df_pl)  # ronds, labels au survol selon ta définition
                st.plotly_chart(fig96, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
    st.markdown("---")           
    # ==============================
    # PLAQUE 384 (depuis Excel)
    # ==============================
    st.subheader("🧩 Plaque 384")
    up384 = st.file_uploader(
        "Charger le plan 384 (.xlsx/.xlsm/.xls)",
        type=["xlsx", "xlsm", "xls"],
        key="plate384_posxlsx"
    )

    if up384 is not None:
        try:
            # Choix moteur lecture (sans installer de package supplémentaire)
            suffix = up384.name.lower().rsplit(".", 1)[-1]
            if suffix == "xls":
                # Option A (recommandée sans xlrd) : refuser .xls proprement
                st.error("Le format .xls n’est pas pris en charge sans xlrd. Merci d’enregistrer le fichier en .xlsx.")
                st.stop()
                # Option B (si xlrd est présent et ok) :
                # engine = "xlrd"
            engine = "openpyxl"  # pour .xlsx / .xlsm

            # 1) tentative simple avec header=0
            up384.seek(0)
            df_xl = pd.read_excel(up384, sheet_name="Template lib", engine=engine, header=0, dtype=str)
            df_xl.columns = [_norm(c) for c in df_xl.columns]

            # 2) si colonnes introuvables → détecter l’entête dans les 30 premières lignes
            if not _have_all(df_xl.columns):
                up384.seek(0)
                probe = pd.read_excel(up384, sheet_name="Template lib", engine=engine, header=None, nrows=30, dtype=str)
                header_row = None
                for i in range(len(probe)):
                    row_norm = {_norm(x) for x in probe.iloc[i].tolist()}
                    if {"positionpl384", "samplesheet", "sampleproject"}.issubset(row_norm):
                        header_row = i
                        break
                if header_row is not None:
                    up384.seek(0)
                    df_xl = pd.read_excel(up384, sheet_name="Template lib", engine=engine, header=header_row, dtype=str)
                    df_xl.columns = [_norm(c) for c in df_xl.columns]
                else:
                    # 3) fallback : colonnes R:T sans en-tête fiable
                    up384.seek(0)
                    df_xl = pd.read_excel(up384, sheet_name="Template lib", engine=engine, header=None, usecols="R:T", dtype=str)
                    df_xl.columns = ["positionpl384", "samplesheet", "sampleproject"]

            # nettoyage minimal des 3 colonnes utiles
            for col in ["positionpl384", "samplesheet", "sampleproject"]:
                if col not in df_xl.columns:
                    raise ValueError(f"Colonne manquante dans l'Excel : {col}")
                df_xl[col] = df_xl[col].fillna("").apply(lambda x: str(x).replace("\xa0", " ").strip())

            # mapping -> DataFrame positions (row, col, samplesheet, sampleproject)
            df_map = _map_excel_to_384_positions(df_xl)

            if df_map.empty:
                st.info("Aucune position 384 valide détectée dans le fichier.")
            else:
                # 👉 utilise les paramètres de la fonction pour la taille
                fig384 = _make_plate384_fig_from_map(
                    df_map,
                    width=1000,
                    height=700,
                    marker_size=30,
                    showlegend=False,  # pour voir "Témoin neg/pos" dans la légende
                )

                # convertir le graphique en HTML
                fig_html = fig384.to_html(
                    full_html=False,
                    include_plotlyjs="cdn",  # si pas d’accès réseau, mets True
                    config={"displayModeBar": False}
                )

                # ===== Récup couleurs & compte par projet =====
                meta = getattr(fig384.layout, "meta", {}) or {}
                proj_colors = dict(meta.get("project_colors", {}))
                viro_project = meta.get("viro_project", "VIRO-GRIPPE")
                viro_color   = meta.get("viro_color", "#1f77b4")

                # Compter nb de puits par projet
                counts = df_map.groupby("sampleproject").size().to_dict()

                # Items de légende : VIRO d’abord, puis autres projets triés
                legend_items = [(viro_project, viro_color, counts.get(viro_project, 0))] + \
                               [(name, color, counts.get(name, 0)) for name, color in sorted(proj_colors.items(), key=lambda x: x[0].upper())]

                # Construire le HTML des chips (en Python, pour éviter les { } dans le f-string principal)
                legend_html = '<div id="plate384-legend" style="display:flex; flex-wrap:wrap; gap:10px;">'
                for name, color, n in legend_items:
                    legend_html += (
                        f'<div class="proj-chip" data-proj="{name}" '
                        'style="cursor:pointer;display:flex;align-items:center;gap:8px;'
                        'padding:6px 10px;border:1px solid #e5e7eb;border-radius:12px;background:#fafafa;">'
                        f'<span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:{color};border:1px solid #999;"></span>'
                        f'<span style="font-size:13px;white-space:nowrap;">{name} ({n})</span>'
                        '</div>'
                    )
                legend_html += '</div>'

                # ---- Dimensions card (plot fixe + légende dessous)
                plot_h = 700                         # = hauteur de ta figure dans to_html
                legend_row_h = 42                    # hauteur d’une rangée de chips
                cols = 4                             # nb de chips par rangée (approx visuel)
                rows = max(1, math.ceil(len(legend_items) / cols))
                legend_h = 16 + rows * legend_row_h  # marge + lignes
                card_padding = 16*2 + 10 + 8         # padding + titre + marges
                card_height = plot_h + legend_h + card_padding

                # ---- Affichage dans UNE seule card
                components.html(
                    f"""
                    <div id="card-plate384" style="
                        background:#fff;
                        border:1px solid #e5e7eb;
                        border-radius:16px;
                        padding:16px;
                        margin:16px auto;
                        max-width:1000px;
                        box-shadow:0 4px 14px rgba(0,0,0,0.06);
                    ">
                      <div style="font-weight:600;margin-bottom:10px;font-size:18px;">
                        🧩 Plaque 384
                      </div>

                      <!-- Zone plot : hauteur FIXE -->
                      <div id="plate384-plot" style="height:{plot_h}px; overflow:hidden;">
                        {fig_html}
                      </div>

                      <!-- Légende cliquable -->
                      <div style="margin-top:12px;">
                        <div style="font-weight:600; font-size:14px; margin-bottom:6px;">
                          Légende projets (clic = on/off)
                        </div>
                        {legend_html}
                      </div>

                      <script>
                      (function(){{
                        // Récupère le div Plotly de la figure
                        var plotWrap = document.getElementById("plate384-plot");
                        if (!plotWrap) return;
                        var plotDiv = plotWrap.querySelector(".plotly-graph-div");
                        if (!plotDiv || !plotDiv.data) return;

                        // État actif par projet (init à true)
                        var chips = Array.from(document.querySelectorAll("#plate384-legend .proj-chip"));
                        var active = Object.create(null);
                        chips.forEach(function(ch){{ active[ch.dataset.proj] = true; }});

                        // Indices des traces dont le nom = projet
                        function indicesForProject(proj){{
                          var idx = [];
                          (plotDiv.data || []).forEach(function(tr, i){{
                            if (tr && tr.name === proj) idx.push(i);
                          }});
                          return idx;
                        }}

                        // Toggle au clic : visible=true / 'legendonly'
                        chips.forEach(function(chip){{
                          chip.addEventListener("click", function(){{
                            var proj = chip.dataset.proj;
                            var now = !active[proj];
                            active[proj] = now;
                            chip.style.opacity = now ? "1" : "0.5";

                            var idxs = indicesForProject(proj);
                            if (idxs.length === 0) return;
                            var vis = now ? true : 'legendonly';
                            idxs.forEach(function(i){{
                              Plotly.restyle(plotDiv, {{'visible':[vis]}}, [i]);
                            }});
                          }});
                        }});
                      }})();
                      </script>
                    </div>
                    """,
                    height=card_height,
                    scrolling=False
                )



        except Exception as e:
            st.error(f"❌ Erreur lecture/placement Excel 384 : {e}")
    
with tab3:
    page_suivi()
    st.header("📈 Suivi de performance")

    # ────────────────────────────────────────────────────────────────────────────
    # Garde-fou minimal
    # ────────────────────────────────────────────────────────────────────────────
    if base_df.empty:
        st.info("Aucun fichier chargé pour le moment.")
        st.stop()
        
    # ────────────────────────────────────────────────────────────────────────────
    # 1) Barre de filtres unifiée
    # ────────────────────────────────────────────────────────────────────────────
    df_src = base_df  # on part de l'historique complet
    st.markdown("### 🎛️ Filtres (run / plaque / témoin)")
    colr, colp, colt, cols = st.columns([1.2, 2.0, 1.2, 1.2])

    with colr:
        runs = sorted(df_src.get("summary_run_id", pd.Series(dtype=str)).dropna().unique().tolist()) if "summary_run_id" in df_src.columns else []
        rid = st.selectbox("Run", options=["(tous)"] + runs, index=0)

    with colp:
        plaques_all = sorted(df_src.get("plaque_id", pd.Series(dtype=str)).dropna().unique().tolist()) if "plaque_id" in df_src.columns else []
        # ▶︎ Comportement identique à Run : "(toutes)" par défaut
        pl_choice = st.selectbox("Plaque", options=["(toutes)"] + plaques_all, index=0)

    with colt:
        temoins = ["TposH1","TposH3","TposB"]
        t_sel = st.multiselect("Témoins", options=temoins, default=temoins)

    with cols:
        seuil = st.slider("Seuil %", min_value=70, max_value=100, value=90, step=1)
        # ▶︎ Par défaut coché
        only_bad = st.checkbox("Sous seuil uniquement", value=True)

    # Appliquer filtres → filtered_df (nom conservé)
    filtered_df = df_src.copy()
    if rid != "(tous)" and "summary_run_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["summary_run_id"] == rid]
    if pl_choice != "(toutes)" and "plaque_id" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["plaque_id"] == pl_choice]

    st.markdown("---")
    
    # ────────────────────────────────────────────────────────────────────────────
    # 3) Outil unifié : commentaires & lots Tpos (Run / Plaque / Échantillon)
    # ────────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <style>
    #expander-unifie-anchor ~ div[data-testid="stExpander"] details summary,
    #expander-unifie-anchor ~ div[data-testid="stExpander"] details summary p,
    #expander-unifie-anchor ~ div[data-testid="stExpander"] details summary span {
      font-size: 3rem !important;
      font-weight: 800 !important;
      line-height: 1.2 !important;
    }
    #expander-unifie-anchor ~ div[data-testid="stExpander"] details summary svg { transform: scale(1.2); }
    </style>
    """, unsafe_allow_html=True)

    with st.expander("🧰 Outil commentaires", expanded=False):
           
        # ────────────────────────────────────────────────────────────────────────────
        # Bandeau scope + options (même visuel, sans "Options avancées")
        # ────────────────────────────────────────────────────────────────────────────
        c0, c1, c2 = st.columns([1.2, 1.6, 1.2])
        with c0:
            # Icônes demandées dans la portée, mais on conserve les mêmes clés internes
            _scope_choices = ["💬 Échantillon", "🏷️ Plaque", "📌 Run"]
            scope_disp = st.radio("Portée", _scope_choices, horizontal=True)
            scope = "Échantillon" if scope_disp.startswith("💬") else ("Plaque" if scope_disp.startswith("🏷️") else "Run")
        with c1:
            mode = st.selectbox("Mode", ["Ajouter à la fin", "Remplacer le commentaire"])
        with c2:
            include_temoins = st.checkbox("Inclure Tpos/NT/Tvide", value=True)

        # ────────────────────────────────────────────────────────────────────────────
        # Sélecteurs selon la portée (inchangé)
        # ────────────────────────────────────────────────────────────────────────────
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

            # ────────────────────────────────────────────────────────────────────────────
            # Presets “chips” + zone libre (TOUT sur une seule ligne, avec icônes)
            # ────────────────────────────────────────────────────────────────────────────
        st.markdown("**Composition du commentaire**")
        # ligne unique : PB / NC / 🔁 Nouveau lot Témoin + Lot n° / 🧪 Autre nouveau lot + Détail
        p1, p2, p3, p4, p5, p6 = st.columns([1.1, 1.5, 1.8, 1.1, 2.1, 1.8])

        with p1:
            preset_pb = st.toggle("PB technique", value=False)

        with p2:
            preset_nc = st.toggle("NC ouverte", value=False)
            num_nc = st.text_input("N° NC", value="", placeholder="ex: NC-2025-123") if preset_nc else ""

        with p3:
            # icône ajoutée comme demandé
            preset_lot = st.toggle("🔁 Nouveau lot Témoin", value=False)

        with p4:
            lot_txt = st.text_input("Lot n°", value="", placeholder="ex: L2345") if preset_lot else ""

        with p5:
            # icône ajoutée + même ligne que le reste
            preset_kit = st.toggle("🧪 Autre nouveau lot (kit, amorces…)", value=False)

        with p6:
            kit_txt = st.text_input("Détail (kit / amorces / autre)", value="", placeholder="ex: Kit XYZ / Amorces ABC") if preset_kit else ""

        user_comment = st.text_area("Commentaire libre", placeholder="Ajoutez un contexte…", height=80)


        # ────────────────────────────────────────────────────────────────────────────
        # Assemble le message final (libellés mis à jour)
        # ────────────────────────────────────────────────────────────────────────────
        parts = []
        if preset_pb:
            parts.append("PB technique")
        if preset_nc and num_nc.strip():
            parts.append(f"NC {num_nc.strip()}")
        if preset_lot:
            # si lot renseigné on l’affiche explicitement
            if lot_txt.strip():
                parts.append(f"Nouveau lot Témoin : {lot_txt.strip()}")
            else:
                parts.append("Nouveau lot Témoin")
        if preset_kit:
            if kit_txt.strip():
                parts.append(f"Autre nouveau lot (kit, amorces…) : {kit_txt.strip()}")
            else:
                parts.append("Autre nouveau lot (kit, amorces…)")
        if user_comment.strip():
            parts.append(user_comment.strip())

        final_comment = " ; ".join(parts).strip()

        # ────────────────────────────────────────────────────────────────────────────
        # Boutons d’action (inchangé)
        # ────────────────────────────────────────────────────────────────────────────
        a1, a2, a3 = st.columns([1, 1, 2])
        with a1:
            btn_apply = st.button("✅ Appliquer le commentaire")
        with a2:
            reset_comment = st.button("🧹 Réinitialiser la saisie")

        if reset_comment:
            st.rerun()

        # ────────────────────────────────────────────────────────────────────────────
        # Application commentaires (conserve ta logique d’origine)
        # ────────────────────────────────────────────────────────────────────────────
        if btn_apply:
            if not final_comment:
                st.warning("Veuillez renseigner au moins un preset ou un texte libre.")
                st.stop()

            scope_key = "sample" if scope == "Échantillon" else ("plaque" if scope == "Plaque" else "run")
            mode_key = "append" if mode == "Ajouter à la fin" else "replace"

            resolved_sample_ids = None

            if scope_key == "sample":
                df_target = filtered_df.copy()

                # Filtres rapides
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

                # 2) Ou tous les filtrés si case cochée
                elif apply_all_filtered:
                    resolved_sample_ids = df_target["sample_id"].dropna().unique().tolist()

                # 3) Sinon on bloque proprement
                else:
                    st.warning("Sélectionnez au moins un échantillon ou cochez « Appliquer à tous les échantillons filtrés ». Aucun changement effectué.")
                    st.stop()

                # Sécurité : si “Nouveau lot Témoin”, restreindre aux TposH1/H3/B et bloquer si vide
                # (on ne dépend PAS du lot_txt pour cette vérification)
                if preset_lot:
                    # repère les Tpos dans la sélection courante (regex insensible à la casse)
                    mask_tpos = df_target["sample_id"].astype(str).str.contains(r"TposH1|TposH3|TposB", case=False, na=False)
                    tpos_ids = df_target.loc[mask_tpos, "sample_id"].dropna().unique().tolist()

                    if target_sample_ids and len(target_sample_ids) > 0:
                        # garde seulement les IDs explicitement sélectionnés qui matchent la regex (insensible à la casse)
                        resolved_sample_ids = [s for s in resolved_sample_ids if re.search(r"TposH1|TposH3|TposB", str(s), flags=re.IGNORECASE)]
                    else:
                        # sinon, applique à tous les Tpos filtrés
                        resolved_sample_ids = tpos_ids

                    if not resolved_sample_ids:
                        st.warning("⚠️ « Nouveau lot Témoin » demandé mais aucun témoin TposH1/H3/B ciblé. Sélectionne au moins un témoin.")
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



    # ────────────────────────────────────────────────────────────────────────────
    # 2) TPOS — Graphe S4/S6 + tableau d’anomalies (UNE SEULE FOIS)
    # ────────────────────────────────────────────────────────────────────────────
    st.markdown("### 🧪 Courbes témoins (S4 plein / S6 pointillé)")
    # mapping tolérant (le multi-select peut renvoyer H3N2/H1N1/BVic ou TposH3/TposH1/TposB)
    LABEL_TO_CODE = {
        "H3N2": "TposH3", "H1N1": "TposH1", "BVic": "TposB",
        "TposH3": "TposH3", "TposH1": "TposH1", "TposB": "TposB",
    }

    for temoin_sel in t_sel:
        code = LABEL_TO_CODE.get(temoin_sel, temoin_sel)   # ← toujours un code TposH*

        # Filtrage identique aux tuiles : sample_id contient le code + ref EPIISL attendue
        sub = filtered_df[filtered_df["sample_id"].astype(str).str.contains(code, case=False, na=False)].copy()
        ref_att = definitions_flu.expected_ref_for_temoin(code)
        if ref_att and "summary_reference_id" in sub.columns:
            sub = sub[sub["summary_reference_id"].astype(str) == ref_att]

        # Titre par témoin
        st.markdown(f"#### Suivi — {temoin_sel}")

        if sub.empty:
            st.info(f"— {temoin_sel} : aucun point correspondant dans la sélection (réf attendue : {ref_att or '—'}).")
            st.divider()
            continue

        # === 1) Bandeau stats (calculées sur le même sous-ensemble filtré)
        kpi_g, kpi_lots = definitions_flu.compute_temoin_stats_cards(sub, code, seuil=seuil, x_col="plaque_id")
        definitions_flu.render_temoin_stats_compact(
            kpi_g, kpi_lots, temoin_sel, seuil=seuil, max_inline_lots=6, visible_lots=3
        )

        # === 2) Graphe par lot (segments courant/probatoire + promotion 🔁)
        # Jeu complet pour ce témoin (toutes plaques, tous runs)
        df_for_plot = base_df[base_df["sample_id"].astype(str).str.contains(code, case=False, na=False)].copy()

        # (Optionnel) si l'utilisateur a explicitement choisi un run dans l'UI "Suivi",
        # tu peux filtrer ici. Sinon NE COUPE PAS :
        # if run_sel and str(run_sel).strip() not in ("", "{tous}"):
        #     df_for_plot = df_for_plot[df_for_plot["summary_run_id"].astype(str) == str(run_sel)]

        # Traçage
        fig = definitions_flu.plot_temoin_lots_s4s6_unique(
            df_for_plot, code, seuil=seuil, x_col="plaque_id", show_dropdown=False
        )
        st.plotly_chart(fig, use_container_width=True)


        # === 3) Anomalies S4/S6 sous le seuil (toujours sur 'sub')
        bad_mask = (
            (pd.to_numeric(sub["summary_consensus_perccoverage_S4"], errors="coerce") < seuil) |
            (pd.to_numeric(sub["summary_consensus_perccoverage_S6"], errors="coerce") < seuil)
        )
        sub_bad = sub.loc[bad_mask, [
            "plaque_id", "sample_id",
            "summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"
        ]].copy()
        n_bad = len(sub_bad)

        st.divider()

        # === 3) Tableau d’anomalies (AgGrid par témoin, dans un expander) ===
        # Prépare S4/S6 & flag
        sub["S4"] = pd.to_numeric(sub.get("summary_consensus_perccoverage_S4"), errors="coerce")
        sub["S6"] = pd.to_numeric(sub.get("summary_consensus_perccoverage_S6"), errors="coerce")
        sub["sous_seuil"] = sub[["S4","S6"]].lt(seuil).any(axis=1)

        disp = sub if not only_bad else sub[sub["sous_seuil"]]
        disp = disp.copy()

        # Colonnes de vue (cohérentes même si vide)
        cols_view = [c for c in ["💬","sample_id","plaque_id","summary_run_id","S4","S6","flags","commentaire"] if c in disp.columns]
        if "commentaire" in disp.columns:
            disp["💬"] = np.where(disp["commentaire"].astype(str).str.strip() != "", "💬", "")
            if hasattr(definitions_flu, "_parse_flags_from_comment"):
                disp["flags"] = disp["commentaire"].apply(definitions_flu._parse_flags_from_comment)

        if cols_view:
            disp_view = disp[cols_view].sort_values(["plaque_id","sample_id"]) if not disp.empty else pd.DataFrame(columns=cols_view)
        else:
            base_cols = [c for c in ["sample_id","plaque_id","summary_run_id","S4","S6"] if c in disp.columns]
            disp_view = disp[base_cols].sort_values(["plaque_id","sample_id"]) if (base_cols and not disp.empty) else pd.DataFrame(columns=base_cols)

        # Compteurs pour le titre de l'expander
        n_rows = len(disp_view)
        n_bad  = int(disp["sous_seuil"].sum()) if "sous_seuil" in disp.columns else 0
        # --- Anomalies pour ce témoin (calculées sur le même sous-ensemble 'sub')
        bad_mask = (
            (pd.to_numeric(sub["summary_consensus_perccoverage_S4"], errors="coerce") < seuil) |
            (pd.to_numeric(sub["summary_consensus_perccoverage_S6"], errors="coerce") < seuil)
        )
        sub_bad = sub.loc[bad_mask, [
            "plaque_id", "sample_id",
            "summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"
        ]].copy()
        n_bad = len(sub_bad)

        title_anom = f"🔎 Anomalies — {temoin_sel} • {n_bad} sous seuil"
        with st.expander(title_anom, expanded=False):
            if disp_view.empty:
                st.info("Aucune anomalie à afficher pour ce témoin dans la sélection courante.")
            else:
                # --- AgGrid lisible + tri récence sur summary_run_id ---
                gb = GridOptionsBuilder.from_dataframe(disp_view)
                gb.configure_default_column(
                    resizable=True, sortable=True, filter=True, floatingFilter=True,
                    wrapText=True, autoHeight=True, flex=1, minWidth=120
                )
                custom_css = {
                    ".ag-cell": {"white-space": "normal !important", "line-height": "1.3 !important"},
                    ".ag-row": {"align-items": "flex-start !important"},
                    ".ag-root-wrapper": {"width": "100% !important"},
                    ".ag-theme-balham": {"width": "100% !important"}
}

                runid_cmp = JsCode("""
                function(a, b) {
                  function key(s){
                    if(s==null) return null;
                    s = String(s);
                    var m = s.match(/(20\\d{2})[-_\\/]?(0[1-9]|1[0-2])[-_\\/]?([0-2]\\d|3[01])/);
                    if(m) return new Date(parseInt(m[1]), parseInt(m[2])-1, parseInt(m[3])).getTime();
                    var m2 = s.match(/^(\\d{2})(\\d{2})(\\d{2})$/);
                    if(m2) return new Date(2000+parseInt(m2[1]), parseInt(m2[2])-1, parseInt(m2[3])).getTime();
                    var n = parseFloat(s);
                    if (!isNaN(n)) return n;
                    return s.toLowerCase();
                  }
                  var A = key(a), B = key(b);
                  if (A==null && B==null) return 0;
                  if (A==null) return -1;
                  if (B==null) return 1;
                  return A - B;
                }
                """)
                if "summary_run_id" in disp_view.columns:
                    gb.configure_column("summary_run_id", comparator=runid_cmp, sort="desc")

                grid_options = gb.build()
                grid_options["onFirstDataRendered"] = JsCode("""
                  function(p){
                    const ids = [];
                    p.columnApi.getAllColumns().forEach(c => ids.push(c.getColId()));
                    p.columnApi.autoSizeColumns(ids, false);
                  }
                """)

                custom_css = {
                    ".ag-cell": {"white-space": "normal !important", "line-height": "1.3 !important"},
                    ".ag-row": {"align-items": "flex-start !important"},
                }

                AgGrid(
                    disp_view,
                    gridOptions=grid_options,
                    allow_unsafe_jscode=True,
                    theme="balham",
                    height=360,
                    fit_columns_on_grid_load=True,
                    custom_css=custom_css,
                    use_container_width=True,
                    key=f"ag_anom_{temoin_sel}"  # 1 grille par témoin
                )

        st.markdown("---")

    # ────────────────────────────────────────────────────────────────────────────
    # 4) Ininterprétables + Varcount (GRA/GRB) + Export Excel
    # ────────────────────────────────────────────────────────────────────────────
    df_clean = filtered_df.copy()
    # Exclure Tpos/NT/Tvide
    df_clean = df_clean[~df_clean["sample_id"].str.contains("Tpos|NT|Tvide", case=False, na=False)]

    # QC seqcontrol en str
    qc = df_clean.get("summary_qc_seqcontrol")
    qc = qc.astype(str).str.upper() if qc is not None else pd.Series(index=df_clean.index, data="OK")

    # Seuil coverage
    SEUIL = 90
    s4 = pd.to_numeric(df_clean.get("summary_consensus_perccoverage_S4"), errors="coerce").fillna(0)
    s6 = pd.to_numeric(df_clean.get("summary_consensus_perccoverage_S6"), errors="coerce").fillna(0)
    df_clean["is_ininterpretable"] = (s4 < SEUIL) | (s6 < SEUIL) | (qc.isin(["FAILED", "0"]))

    # % Ininterprétables par plaque
    plaque_stats = df_clean.groupby("plaque_id").agg(
        total_samples=("sample_id", "count"),
        nb_ininterpretable=("is_ininterpretable", "sum"),
    ).reset_index()
    plaque_stats["pct_ininterpretable"] = (
        100 * plaque_stats["nb_ininterpretable"] / plaque_stats["total_samples"]
    ).round(1)

    df_gra = plaque_stats[plaque_stats["plaque_id"].str.contains("GRA", na=False)]
    df_grb = plaque_stats[plaque_stats["plaque_id"].str.contains("GRB", na=False)]

    # Paramètres communs aux graphes
    _run_id = None if rid in ("(tous)", "", None) else str(rid).strip()

    # Source des statuts de lots pour les plaques visibles dans les histogrammes
    plaques_visibles = sorted(plaque_stats["plaque_id"].astype(str).unique().tolist())
    lot_stage_hist = definitions_flu.build_lot_stage_df_for_hist(base_df, plaques=plaques_visibles)

    # Histogrammes % ininterprétables
    if not df_gra.empty:
        definitions_flu.plot_histogram_with_export(
            df_gra, "📉 % Ininterprétables par plaque (GRA)", "ininterpretable_GRA.csv",
            all_samples_df=filtered_df, run_id=_run_id, lot_stage_df=lot_stage_hist
        )
    if not df_grb.empty:
        definitions_flu.plot_histogram_with_export(
            df_grb, "📉 % Ininterprétables par plaque (GRB)", "ininterpretable_GRB.csv",
            all_samples_df=filtered_df, run_id=_run_id, lot_stage_df=lot_stage_hist
        )

    # Varcount ≥ 13 par plaque
    vc_raw = pd.to_numeric(
        df_clean.get("val_varcount", pd.Series(index=df_clean.index, dtype=object))
               .astype(str).str.extract(r"(?i)VARCOUNT\s*(\d+)", expand=False),
        errors="coerce"
    ).fillna(0)
    df_clean["varcount_num"] = vc_raw

    plaque_total = df_clean.groupby("plaque_id")["sample_id"].count().rename("total_samples")
    plaque_v13   = df_clean[df_clean["varcount_num"] >= 13].groupby("plaque_id")["sample_id"].count().rename("nb_varcount_13")
    df_varcount  = pd.concat([plaque_total, plaque_v13], axis=1).fillna(0).reset_index()
    df_varcount["% varcount >= 13"] = (100 * df_varcount["nb_varcount_13"] / df_varcount["total_samples"]).round(1)

    df_gra_v = df_varcount[df_varcount["plaque_id"].str.contains("GRA", na=False)]
    df_grb_v = df_varcount[df_varcount["plaque_id"].str.contains("GRB", na=False)]

    if not df_gra_v.empty:
        definitions_flu.plot_varcount(
            df_gra_v, "📈 % Varcount ≥ 13 par plaque (GRA)",
            all_samples_df=filtered_df, run_id=_run_id, lot_stage_df=lot_stage_hist
        )
    if not df_grb_v.empty:
        definitions_flu.plot_varcount(
            df_grb_v, "📈 % Varcount ≥ 13 par plaque (GRB)",
            all_samples_df=filtered_df, run_id=_run_id, lot_stage_df=lot_stage_hist
        )
        st.download_button("📥 Télécharger GRB (varcount)", df_grb_v.to_csv(index=False),
                           file_name="details_varcount_GRB.csv", mime="text/csv")

    # Export Excel multi-feuilles (basé sur le FILTRE courant)
    coverage_cols = [f"summary_consensus_perccoverage_S{i}" for i in range(1, 9)]
    df_tpos = filtered_df[filtered_df["sample_id"].str.contains("Tpos", case=False, na=False)].copy()
    if "plaque_id" not in df_tpos.columns:
        df_tpos["plaque_id"] = df_tpos["sample_id"].astype(str).str[:9]
    cols_tpos_export = ["sample_id", "plaque_id"] + [c for c in coverage_cols if c in df_tpos.columns]
    df_tpos_export = df_tpos[cols_tpos_export] if not df_tpos.empty else pd.DataFrame(columns=cols_tpos_export)

    df_gra_inint = df_gra.copy()
    df_grb_inint = df_grb.copy()

    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        if not df_gra_inint.empty: df_gra_inint.to_excel(writer, sheet_name="Ininterprétables GRA", index=False)
        if not df_grb_inint.empty: df_grb_inint.to_excel(writer, sheet_name="Ininterprétables GRB", index=False)
        if not df_gra_v.empty:      df_gra_v.to_excel(writer,      sheet_name="Varcount GRA", index=False)
        if not df_grb_v.empty:      df_grb_v.to_excel(writer,      sheet_name="Varcount GRB", index=False)
        if not df_tpos_export.empty:df_tpos_export.to_excel(writer, sheet_name="Suivi TPOS", index=False)

    st.download_button(
        label="📅 Télécharger Excel multi-feuilles (filtre courant)",
        data=excel_buffer.getvalue(),
        file_name="dashboard_qualite_grippe_filtré.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


with tab4:
    page_historique()

    if os.path.exists(HISTO_FILE):
        st.markdown("### 📜 Historique des chargements")
        df_h = pd.read_csv(HISTO_FILE)

        # Nettoyage colonnes inutiles
        df_h = df_h.drop(columns=["taille_Ko", "shape", "filename", "Nom_fichier", "timestamp"], errors="ignore")

        # Assure les colonnes attendues
        for col in ["date_heure", "nom_fichier", "run_id", "type", "operateur"]:
            if col not in df_h.columns:
                df_h[col] = ""

        # Conversion date + tri
        df_h["date_heure"] = pd.to_datetime(df_h["date_heure"], errors="coerce")
        df_h = df_h.sort_values("date_heure", ascending=False)

        # Format date pour affichage
        df_f = df_h.copy()
        df_f = df_f.assign(date_heure=df_f["date_heure"].dt.strftime("%Y-%m-%d %H:%M:%S"))

        # --- Affichage AgGrid (remplace st.dataframe + CSS) ---
        view_cols = ["date_heure", "nom_fichier", "run_id", "type", "operateur"]
        disp_h = df_f[view_cols].copy()

        st.markdown("### 🔍 Recherche rapide")
        q_hist = st.text_input("Tapez pour filtrer toutes les colonnes", value="", key="hist_quick")

        gb = GridOptionsBuilder.from_dataframe(disp_h)
        # 👉 Distribue l'espace horizontal pour occuper toute la largeur du conteneur
        gb.configure_default_column(resizable=True, sortable=True, filter=True, floatingFilter=True, flex=1, minWidth=110)

        # Tri date correct
        date_cmp = JsCode("""
        function(a, b){
          var da = new Date(a.replace(' ', 'T'));
          var db = new Date(b.replace(' ', 'T'));
          return da - db;
        }
        """)
        gb.configure_column("date_heure", header_name="Date/Heure", comparator=date_cmp)

        # Options de grille
        gb.configure_grid_options(
            rowSelection="single",
            rowHeight=32,
            pagination=True, paginationPageSize=25,
            animateRows=True,
            suppressMenuHide=False,
            domLayout='normal'   # 'normal' convient ici; l'important est 'flex=1' ci-dessus
        )

        grid_options = gb.build()
        if q_hist.strip():
            grid_options["quickFilterText"] = q_hist.strip()

        AgGrid(
            disp_h,
            gridOptions=grid_options,
            enable_enterprise_modules=False,
            update_mode=GridUpdateMode.NO_UPDATE,
            allow_unsafe_jscode=True,
            theme="balham",
            fit_columns_on_grid_load=True,   # ok avec flex:1, ça cale dès le rendu
            height=700,
            use_container_width=True         # 👉 prend toute la largeur disponible (avec layout="wide")
        )


    else:
        st.info("Aucun fichier chargé pour le moment.")

with tab5:
    st.header("📦 Archives — Echantillons")

    if base_df is None or base_df.empty:
        st.info("Aucune donnée dans la base pour le moment (historique_data.csv est vide).")
        st.stop()

    # ───────────────────────────────────────────────────────────────────────────
    # (A) Préparation
    # ───────────────────────────────────────────────────────────────────────────
    arch = base_df.copy()
    arch["sample_id"] = arch["sample_id"].astype(str)
    arch["Glims_id"] = arch["sample_id"].str.extract(r"(\d{12})", expand=False)
    arch["is_dup_Glims_id"] = arch["Glims_id"].duplicated(keep=False) & arch["Glims_id"].notna()

    if "run_short_id" not in arch.columns and "summary_run_id" in arch.columns:
        arch["run_short_id"] = arch["summary_run_id"].astype(str).str[:6]
    if "plaque_id" not in arch.columns:
        arch["plaque_id"] = arch["sample_id"].str[:9]

    # Vue dédupliquée
    arch_src = _arch_dedup_view(arch)

    # (re)calcul local sur la vue
    arch_src["Glims_id"] = arch_src["sample_id"].str.extract(r"(\d{12})", expand=False)
    arch_src["is_dup_Glims_id"] = arch_src["Glims_id"].duplicated(keep=False) & arch_src["Glims_id"].notna()

    # Info dédup
    n_hidden = len(arch) - len(arch_src)
    if n_hidden > 0:
        st.caption(f"🔁 Déduplication: {n_hidden} doublon(s) exact(s) (même sample/plaque/run) masqué(s) dans la vue Archives.")

    # ───────────────────────────────────────────────────────────────────────────
    # (B) Filtres
    # ───────────────────────────────────────────────────────────────────────────
    st.markdown("### 🔎 Filtres")
    c1, c2, c3 = st.columns([1.5, 1.5, 2])

    with c1:
        runs = sort_ids_by_recency(arch_src, "summary_run_id") if "summary_run_id" in arch_src.columns else []
        sel_runs = st.multiselect(
            "summary_run_id",
            options=runs,
            default=[],
            key="arch_runs",
            placeholder="Tous les runs (du + récent au + ancien)"
        )

    with c2:
        plaques = sort_ids_by_recency(arch_src, "plaque_id")
        sel_plaques = st.multiselect(
            "plaque_id",
            options=plaques,
            default=[],
            key="arch_plaques",
            placeholder="Toutes les plaques (du + récent au + ancien)"
        )

    with c3:
        q = st.text_input("Recherche (sample_id, Glims_id, clade, etc.)", value="", key="arch_query")

    # Appliquer les filtres (vide = "tous")
    f = arch_src.copy()
    if sel_runs:
        f = f[f["summary_run_id"].isin(sel_runs)]
    if sel_plaques:
        f = f[f["plaque_id"].isin(sel_plaques)]
    if q.strip():
        qlow = q.strip().lower()
        cols_search = [c for c in [
            "sample_id","Glims_id","summary_reference_id","summary_run_id",
            "plaque_id","val_varcount","val_avisbio","val_result","commentaire"
        ] if c in f.columns]
        if cols_search:
            mask = pd.Series(False, index=f.index)
            for c in cols_search:
                mask = mask | f[c].astype(str).str.lower().str.contains(qlow, na=False)
            f = f[mask]
    # ───────────────────────────────────────────────────────────────────────────
    # (B bis) Exclure les échantillons 'VIDE' / 'VIFE' (et variantes)
    # ───────────────────────────────────────────────────────────────────────────

    VIDE_REGEX = re.compile(r"\b(?:t[-_\s]*tvide\w*|tvide\w*|tvife\w*|tvife0\w*|vide\w*)\b", flags=re.IGNORECASE)
    f = f.copy()
    _mask_excl = f["sample_id"].astype(str).str.contains(VIDE_REGEX, na=False)
    nb_exclus_vide_vife = int(_mask_excl.sum())
    f = f.loc[~_mask_excl].copy()

    # ───────────────────────────────────────────────────────────────────────────
    # (C) KPIs & toggle
    # ───────────────────────────────────────────────────────────────────────────
    if "Glims_id" not in f.columns:
        f["Glims_id"] = f["sample_id"].astype(str).str.extract(r"(\d{12})", expand=False)

    _src_for_dups = arch_src.copy()
    if "Glims_id" not in _src_for_dups.columns:
        _src_for_dups["Glims_id"] = _src_for_dups["sample_id"].astype(str).str.extract(r"(\d{12})", expand=False)

    _dup_mask_src = _src_for_dups["Glims_id"].astype(str).duplicated(keep=False)
    _dup_set = set(_src_for_dups.loc[_dup_mask_src, "Glims_id"].dropna().astype(str).unique())

    total_lignes = len(f)
    nb_Glims_id = f["Glims_id"].notna().sum()
    nb_lignes_dups = f["Glims_id"].astype(str).isin(_dup_set).sum()
    nb_Glims_id_dups = f.loc[f["Glims_id"].astype(str).isin(_dup_set), "Glims_id"].nunique()
    # % échantillons repassés = (nb_Glims_id_dups / nb_Glims_id) * 100
    pct_repasses = 0.0
    if nb_Glims_id > 0:
        pct_repasses = round(100.0 * nb_Glims_id_dups / float(nb_Glims_id), 1)
        
    k1, k2, k3, k4, k5= st.columns(5)   
    k1.metric("Lignes affichées", total_lignes)
    k2.metric("Lignes avec Glims_id", nb_Glims_id)
    k3.metric("Nombre d'échantillon repassé", nb_Glims_id_dups)
    k4.metric("% d'échantillon repassé", pct_repasses)
    k5.metric("Échantillons 'TVIDE' exclus", nb_exclus_vide_vife)  # ← nouveau KPI


    st.markdown("---")
    only_dups = st.checkbox("Afficher uniquement les doublons (numéro 12 chiffres)", value=False, key="arch_only_dups")

    # Base de travail pour la grille
    base_for_grid = f.copy()
    base_for_grid["dup"] = base_for_grid["Glims_id"].astype(str).isin(_dup_set)
    if only_dups:
        base_for_grid = base_for_grid[base_for_grid["dup"]]

    # ───────────────────────────────────────────────────────────────────────────
    # (D) AgGrid — icônes 1ʳᵉ colonne + colonnes d’ORIGINE uniquement
    # ───────────────────────────────────────────────────────────────────────────
    disp = base_for_grid.copy()
    disp["sample_id"] = disp["sample_id"].astype(str)

    # Arrondir S4/S6 à 2 décimales pour l’affichage
    for _c in ["summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S6"]:
        if _c in disp.columns:
            disp[_c] = pd.to_numeric(disp[_c], errors="coerce").round(2)

    # Glims_id si manquant
    if "Glims_id" not in disp.columns:
        disp["Glims_id"] = disp["sample_id"].str.extract(r"(\d{12})", expand=False)

    # Colonne d’icônes (🕒, 🧪, 🔁, 💬, 🏷️, 📌) + tooltip
    disp = definitions_flu.add_icons_column_for_archives(disp, col_name="__icons__")

    # ⚠️ Ne garder QUE les colonnes d’origine
    cols_pref = [
        "__icons__",
        "sample_id",
        "Glims_id",
        "plaque_id",
        "summary_run_id",
        "summary_reference_id",
        "summary_consensus_perccoverage_S4",
        "summary_consensus_perccoverage_S6",
        "val_varcount",
        "val_avisbio",
        "val_result",
        "commentaire",
    ]
    disp = disp[[c for c in cols_pref if c in disp.columns]]

    # ==== JS utils

    dup_group_value_getter = JsCode("""
    function(p){
      var go = p.api && p.api.gridOptionsWrapper && p.api.gridOptionsWrapper.gridOptions;
      var t  = (go && go.context && go.context.highlightGlims_id!=null) ? String(go.context.highlightGlims_id) : null;
      var c  = (p.data && p.data.Glims_id!=null) ? String(p.data.Glims_id) : null;
      return (t && c===t) ? 0 : 1;  // 0 en tête si ciblé
    }
    """)

    runid_comparator = JsCode("""
    function(a,b){
      function P(s){
        if(s==null) return null; s=String(s);
        var m=s.match(/(20\\d{2})[-_\\/]?(0[1-9]|1[0-2])[-_\\/]?([0-2]\\d|3[01])/);
        if(m) return new Date(parseInt(m[1]),parseInt(m[2])-1,parseInt(m[3])).getTime();
        var m2=s.match(/^(\\d{2})(\\d{2})(\\d{2})$/);
        if(m2) return new Date(2000+parseInt(m2[1]),parseInt(m2[2])-1,parseInt(m2[3])).getTime();
        var n=parseFloat(s); if(!isNaN(n)) return n; return s.toLowerCase();
      }
      var A=P(a), B=P(b); if(A==null&&B==null) return 0; if(A==null) return -1; if(B==null) return 1;
      if(A<B) return -1; if(A>B) return 1; return 0;
    }
    """)

    text_formatter = JsCode("function(v){ if(v==null) return null; return String(v).toLowerCase(); }")
    two_decimals = JsCode("""
    function(p){
      if (p.value == null || p.value === '') return '';
      var n = Number(p.value); if (isNaN(n)) return '';
      return n.toFixed(2);
    }
    """)
    on_first_data_rendered = JsCode("""
    function(p){
      try{ p.api.sizeColumnsToFit(); }catch(e){}
    }
    """)

    on_grid_size_changed = JsCode("""
    function(p){
      try{ p.api.sizeColumnsToFit(); }catch(e){}
    }
    """)

    # ==== GridOptions
    gb = GridOptionsBuilder.from_dataframe(disp)
    gb.configure_default_column(
        resizable=True,
        sortable=True,
        filter=True,           # barres de recherche
        floatingFilter=True,
        wrapText=False, autoHeight=False,
        flex=1, minWidth=110
    )

    # 1ʳᵉ colonne : icônes (pas de filtre)
    gb.configure_column(
        "__icons__", header_name="", width=96, pinned="left",
        sortable=False, filter=False, floatingFilter=False, suppressMenu=True,
        tooltipField="__icons___tt",
        cellStyle={"textAlign":"center","fontSize":"18px"}
    )

    # Renommer + formatter S4/S6 (2 décimales)
    if "summary_consensus_perccoverage_S4" in disp.columns:
        gb.configure_column(
            "summary_consensus_perccoverage_S4",
            header_name="Couverture_S4",
            filter="agNumberColumnFilter",
            floatingFilter=True,
            valueFormatter=two_decimals,
            cellStyle={"textAlign": "center"},   # ← centrage des valeurs
            minWidth=120
        )
    if "summary_consensus_perccoverage_S6" in disp.columns:
        gb.configure_column(
            "summary_consensus_perccoverage_S6",
            header_name="Couverture_S6",
            filter="agNumberColumnFilter",
            floatingFilter=True,
            valueFormatter=two_decimals,
            cellStyle={"textAlign": "center"},   # ← centrage des valeurs
            minWidth=120
        )


    # Filtres texte “légers” (debounce) pour les autres colonnes textuelles
    text_cols = [c for c in [
        "sample_id","Glims_id","plaque_id","summary_run_id",
        "summary_reference_id","val_varcount","val_avisbio",
        "val_result","commentaire"
    ] if c in disp.columns]
    for c in text_cols:
        gb.configure_column(
            c,
            filter="agTextColumnFilter",
            floatingFilter=True,
            filterParams={"debounceMs": 350, "textFormatter": text_formatter}
        )

    # Colonne virtuelle (nécessaire au groupage au clic)
    gb.configure_column("dup_group", header_name="", hide=True, sortable=True,
                        valueGetter=dup_group_value_getter)

    # Masquer colonnes techniques si elles existent (pour éviter les surprises)
    if "__icons___tt" in disp.columns:
        gb.configure_column("__icons___tt", hide=True)
    if "dup" in disp.columns:
        gb.configure_column("dup", hide=True)

    gb.configure_grid_options(
        onFirstDataRendered=on_first_data_rendered,   # ← ajuste à l’affichage
        onGridSizeChanged=on_grid_size_changed,       # ← réajuste si la taille change
        rowSelection="multiple",
        suppressRowClickSelection=True,
        rowHeight=30,
        pagination=True, paginationPageSize=150,
        animateRows=False,
        rowBuffer=6,
        context={"highlightGlims_id": None},
        domLayout='normal',
        suppressColumnVirtualisation=False,
        suppressDragLeaveHidesColumns=True
    )


    grid_options = gb.build()

    # Tri par défaut “intelligent” sur summary_run_id
    for cd in grid_options.get("columnDefs", []):
        if cd.get("field") == "summary_run_id":
            cd["sort"] = "desc"
            cd["comparator"] = runid_comparator
            break

    # Quick filter global (et purge si vide)
    st.markdown("### 🔍 Recherche rapide (AgGrid)")
    q_global = st.text_input("Tapez pour filtrer toutes les colonnes", value="", key="arch_quick")
    if q_global.strip():
        grid_options["quickFilterText"] = q_global.strip()
    else:
        grid_options.pop("quickFilterText", None)

    AgGrid(
        disp,
        gridOptions=grid_options,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=True,
        theme="balham",
        fit_columns_on_grid_load=True,
        height=800,
        width='100%',
    )

    # ───────────────────────────────────────────────────────────────────────────
    # (E) Export CSV (respecte filtres + toggle 'doublons')
    # ───────────────────────────────────────────────────────────────────────────
    exp_cols = [c for c in base_for_grid.columns if c not in ["_sample_html", "_sample_badge"]]
    csv_buf = io.StringIO()
    base_for_grid[exp_cols].to_csv(csv_buf, index=False)
    st.download_button(
        "⬇️ Export CSV (filtres appliqués)",
        data=csv_buf.getvalue(),
        file_name=("archives_doublons.csv" if only_dups else "archives_filtrees.csv"),
        mime="text/csv"
    )