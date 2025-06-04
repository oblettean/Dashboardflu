import re
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

# Configuration globale
st.set_page_config(page_title="Suivi Qualité - Séquençage Grippe", layout="wide")
st.title("🧪 Suivi de Performance des Séquençages Grippe")

# Mapping clade
clade_mapping = {
    "EPIISL129744": "H3N2", "EPIISL14776052": "H5N1a", "EPIISL16209046": "H1N1s", "EPIISL166957": "BYam",
    "EPIISL177743": "H5N8a", "EPIISL179635": "H5N6a", "EPIISL200780": "H1N1", "EPIISL200936": "H3N2s",
    "EPIISL219327": "BVic", "EPIISL502881": "H7N9a", "EPIISL697602": "H9N2a", "EPIISL91639": "H1N2s"
}

DATA_FILE = "historique_data.csv"

@st.cache_data
def load_tsv(uploaded_file, usecols):
    return pd.read_csv(uploaded_file, sep="\t", dtype=str, usecols=usecols)

@st.cache_data
def load_historical_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

def extraire_plaque(sample_id):
    if not isinstance(sample_id, str):
        return None
    match = re.search(r'\d{2}P\d{3}GR[A-Z]', sample_id)
    if match:
        return match.group(0)
    if "-" in sample_id:
        return sample_id.split("-")[0]
    return None

# Charger les données historiques
base_df = load_historical_data()

# Initialiser new_data_filtered comme un DataFrame vide
new_data_filtered = pd.DataFrame()

# Définir les colonnes nécessaires
columns_needed = [
    "sample_id", "summary_reference_id", "summary_fastq_readcount", "summary_bam_readcount",
    "summary_bam_meandepth_S1", "summary_bam_meandepth_S2", "summary_bam_meandepth_S3",
    "summary_bam_meandepth_S4", "summary_bam_meandepth_S5", "summary_bam_meandepth_S6",
    "summary_bam_meandepth_S7", "summary_bam_meandepth_S8", "summary_consensus_perccoverage_S1",
    "summary_consensus_perccoverage_S2", "summary_consensus_perccoverage_S3",
    "summary_consensus_perccoverage_S4", "summary_consensus_perccoverage_S5",
    "summary_consensus_perccoverage_S6", "summary_consensus_perccoverage_S7",
    "summary_consensus_perccoverage_S8", "summary_vcf_dpcount", "summary_qc_seqcontrol",
    "summary_bam_verif", "summary_vcf_coinf01match", "summary_vcf_coinf02iqr",
    "summary_run_id", "val_poi", "val_avisbio", "val_insertions", "nextclade_qc_overallStatus",
    "nextclade_frameShifts", "nextclade_RBD"
]

tsv_file = st.file_uploader("🔄 Charger un nouveau fichier TSV", type=["tsv"])

if tsv_file is not None:
    # Charger uniquement les colonnes nécessaires
    new_data = load_tsv(tsv_file, columns_needed)

    # Créer la colonne plaque_id en appliquant la fonction extraire_plaque sur sample_id
    new_data["plaque_id"] = new_data["sample_id"].apply(extraire_plaque)
    # Conversion de summary_bam_readcount en float pour comparaison
    new_data["summary_bam_readcount"] = pd.to_numeric(new_data["summary_bam_readcount"], errors='coerce').fillna(0)

    # Filtrage sur sample_id contenant GRA
    gra_df = new_data[new_data["sample_id"].str.contains("GRA", na=False)].copy()

    # Supprimer les lignes où summary_reference_id == EPIISL219327 pour GRA
    gra_df = gra_df[gra_df["summary_reference_id"] != "EPIISL219327"]

    # Fonction de filtrage par sample_id pour GRA
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

    filtered_gra = gra_df.groupby("sample_id", group_keys=False).apply(filter_gra_group)

    # Gestion spécifique pour sample_id contenant "GRA" et "Tpos"
    mask_gra_tpos = new_data["sample_id"].str.contains("GRA", na=False) & new_data["sample_id"].str.contains("Tpos", na=False)
    gra_tpos_keep = new_data[mask_gra_tpos & (new_data["summary_reference_id"] == "EPIISL129744")]

    # Filtrage sur sample_id contenant GRB, on garde seulement ceux avec référence EPIISL219327
    grb_df = new_data[new_data["sample_id"].str.contains("GRB", na=False)].copy()
    grb_filtered = grb_df[grb_df["summary_reference_id"] == "EPIISL219327"]

    # Retirer toutes les lignes GRA et GRB du dataframe original
    new_data_no_gra_grb = new_data[~(new_data["sample_id"].str.contains("GRA", na=False) | new_data["sample_id"].str.contains("GRB", na=False))]

    # Concaténer : dataframe sans GRA/GRB + lignes filtrées GRA + lignes GRA Tpos filtrées + lignes filtrées GRB
    new_data_filtered = pd.concat([new_data_no_gra_grb, filtered_gra, gra_tpos_keep, grb_filtered], ignore_index=True)

    st.write(f"Données filtrées : {new_data_filtered.shape[0]} lignes")

    # Vérifiez les colonnes nécessaires
    missing_cols = [col for col in columns_needed if col not in new_data_filtered.columns]
    if missing_cols:
        st.error(f"❌ Erreur : Colonnes manquantes dans le fichier chargé : {missing_cols}")
        st.stop()

    if 'commentaire' not in new_data_filtered.columns:
        new_data_filtered["commentaire"] = ""

    new_data_filtered = new_data_filtered[columns_needed + ["commentaire"]].copy()

    # Mettre à jour le fichier historique avec les nouvelles données filtrées
    base_df = pd.concat([base_df, new_data_filtered], ignore_index=True)
    base_df = base_df.drop_duplicates(subset=["sample_id", "nextclade_RBD"])
    base_df.to_csv(DATA_FILE, index=False)

    # Ajouter la colonne plaque_id si elle n'existe pas
    if "plaque_id" not in new_data_filtered.columns:
        new_data_filtered["plaque_id"] = new_data_filtered["sample_id"].apply(extraire_plaque)

    # Créer le sélecteur de plaque
    if "plaque_id" in new_data_filtered.columns:
        plaques_disponibles = sorted(new_data_filtered["plaque_id"].dropna().unique())
        plaque_selectionnee = st.selectbox("🔍 Sélectionnez une plaque :", plaques_disponibles)

        # Filtrer les données pour la plaque sélectionnée
        df_plaque = new_data_filtered[new_data_filtered["plaque_id"] == plaque_selectionnee]
    else:
        st.error("❌ Erreur : La colonne 'plaque_id' est manquante dans le DataFrame.")

    # Séparer témoins et non-témoins dans cette plaque
    temoin_pattern = r"TposH3|TposH1|TposB|NT1|NT2|NT3|Tvide"
    temoin_df = df_plaque[df_plaque["sample_id"].str.contains(temoin_pattern, case=False, na=False, regex=True)]
    non_temoin_df = df_plaque[~df_plaque["sample_id"].str.contains(temoin_pattern, case=False, na=False, regex=True)]

    st.markdown("### 🧪 Aide à la confirmation pour la plaque sélectionnée")

    if not temoin_df.empty:
        temoin_df["summary_consensus_perccoverage_S4"] = pd.to_numeric(temoin_df["summary_consensus_perccoverage_S4"], errors='coerce')
        temoin_df["summary_consensus_perccoverage_S6"] = pd.to_numeric(temoin_df["summary_consensus_perccoverage_S6"], errors='coerce')

        st.markdown(f"#### 📦 Résultats pour la plaque : {plaque_selectionnee}")
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        def show_info(title, sample_filter, col, attendu=None):
            df = temoin_df[temoin_df["sample_id"].str.contains(sample_filter, case=False, na=False)]
            if not df.empty:
                sample_id = df["sample_id"].values[0]
                clade = df["summary_vcf_coinf01match"].iloc[0] if "summary_vcf_coinf01match" in df.columns else "N/A"
                s4 = df["summary_consensus_perccoverage_S4"].iloc[0]
                s6 = df["summary_consensus_perccoverage_S6"].iloc[0]
                icon = "🧪" if not attendu or attendu in str(clade) else "⚠️"
                col.metric(f"{icon} {title}", f"{clade}", help=f"Sample ID : {sample_id}")
                col.markdown(f"S4 couverture : **{s4:.1f}%**  \nS6 couverture : **{s6:.1f}%**")
            else:
                col.info(f"❓ {title} non trouvé")

        show_info("TposH3", "TposH3", col1, attendu="H3")
        show_info("TposH1", "TposH1", col2, attendu="H1")
        show_info("TposB",  "TposB",  col3, attendu="B")
        show_info("NT1",    "NT1",    col4)
        show_info("NT2",    "NT2",    col5)
        show_info("NT3",    "NT3",    col6)
    else:
        st.info("❌ Aucun témoin détecté pour cette plaque.")

    # Détection des doubles populations
    st.markdown("### 🧪 Détection des doubles populations")

    def count_double_pop(df, epiisl_reference, seuil=6):
        df = df.copy()
        df["summary_vcf_dpcount"] = pd.to_numeric(df["summary_vcf_dpcount"], errors="coerce")
        return len(df[(df["summary_reference_id"] == epiisl_reference) & (df["summary_vcf_dpcount"] >= seuil)])

    double_H3N2 = count_double_pop(non_temoin_df, "EPIISL129744")
    double_H1N1 = count_double_pop(non_temoin_df, "EPIISL200780")
    double_Bvic = count_double_pop(non_temoin_df, "EPIISL219327")

    col_H3N2, col_H1N1, col_Bvic = st.columns(3)
    col_H3N2.metric("🧬 Double pop. H3N2 (≥6)", f"{double_H3N2} échantillons")
    col_H1N1.metric("🧬 Double pop. H1N1 (≥6)", f"{double_H1N1} échantillons")
    col_Bvic.metric("🧬 Double pop. Bvic (≥6)", f"{double_Bvic} échantillons")

    # Échantillons échoués (Seq failed)
    st.markdown("### ❌ Échantillons échoués (Seq failed)")
    failed_df = non_temoin_df[non_temoin_df["summary_qc_seqcontrol"].str.upper().isin(["FAILED", "0"])]
    st.metric("Seq failed", f"{len(failed_df)} échantillons")

    # Échantillons Ininterprétables
    st.markdown("### ❓ Échantillons Ininterprétables (<90% S4 ou S6)")
    non_temoin_df["summary_consensus_perccoverage_S4"] = pd.to_numeric(non_temoin_df["summary_consensus_perccoverage_S4"], errors="coerce")
    non_temoin_df["summary_consensus_perccoverage_S6"] = pd.to_numeric(non_temoin_df["summary_consensus_perccoverage_S6"], errors="coerce")
    mask_ininterpretable = (non_temoin_df["summary_consensus_perccoverage_S4"] < 90) | (non_temoin_df["summary_consensus_perccoverage_S6"] < 90)
    ininterpretable_df = non_temoin_df[mask_ininterpretable].drop_duplicates(subset="sample_id")

    col_H3N2, col_H1N1, col_BVic = st.columns(3)

    with col_H3N2:
        count_H3N2 = len(ininterpretable_df[ininterpretable_df["summary_reference_id"] == "EPIISL129744"])
        st.metric("H3N2 ininterprétables", f"{count_H3N2} échantillons")

    with col_H1N1:
        count_H1N1 = len(ininterpretable_df[ininterpretable_df["summary_reference_id"] == "EPIISL200780"])
        st.metric("H1N1 ininterprétables", f"{count_H1N1} échantillons")

    with col_BVic:
        count_BVic = len(ininterpretable_df[ininterpretable_df["summary_reference_id"] == "EPIISL219327"])
        st.metric("BVic ininterprétables", f"{count_BVic} échantillons")

    # Souches secondaires (score > 0.4)
    st.markdown("### 🧬 Souches secondaires (score > 0.4)")
    references_exclues = {"EPIISL129744", "EPIISL200780", "EPIISL219327"}
    comp_file = st.file_uploader("🔍 Charger le fichier de similarité (matrice EPIISL vs échantillons)", type=["tsv"])

    if comp_file is not None:
        try:
            similarity_df = pd.read_csv(comp_file, sep="\t", index_col=0)
            filtered_similarity_df = similarity_df.drop(index=references_exclues, errors='ignore')

            plaque_samples = new_data_filtered[new_data_filtered["plaque_id"] == plaque_selectionnee]["sample_id"].unique()
            filtered_similarity_df = filtered_similarity_df[plaque_samples]

            filtered_similarity_df = filtered_similarity_df.astype(float)

            presence_mask = filtered_similarity_df > 0.4
            found_souches = []

            for sample in filtered_similarity_df.columns:
                sample_plaque_id = new_data_filtered[new_data_filtered["sample_id"] == sample]["plaque_id"].iloc[0]

                if "GRA" in sample_plaque_id:
                    souches_detectees = [souche for souche in filtered_similarity_df.index[presence_mask[sample]] if souche not in ["EPIISL129744", "EPIISL200780"]]
                elif "GRB" in sample_plaque_id:
                    souches_detectees = [souche for souche in filtered_similarity_df.index[presence_mask[sample]] if souche != "EPIISL219327"]
                else:
                    souches_detectees = filtered_similarity_df.index[presence_mask[sample]].tolist()

                scores = filtered_similarity_df[sample][presence_mask[sample]].tolist()

                for souche, score in zip(souches_detectees, scores):
                    clade = clade_mapping.get(souche, souche)
                    found_souches.append({
                        "sample_id": sample,
                        "souche_EPIISL": souche,
                        "clade": clade,
                        "similarity_score": round(score, 3)
                    })

            if found_souches:
                found_df = pd.DataFrame(found_souches)
                enriched_df = found_df.merge(new_data_filtered[["sample_id", "plaque_id", "summary_run_id", "commentaire"]], on="sample_id", how="left")
                st.dataframe(enriched_df)
            else:
                st.info("✅ Aucune autre souche détectée avec un score > 0.4.")
        except Exception as e:
            st.error(f"❌ Erreur lors du traitement du fichier de similarité : {e}")

    # Coinfections et réassortiment
    st.markdown("### 🧬 Coinfections et réassortiment")

    non_temoin_df_filtered = non_temoin_df[~non_temoin_df["summary_vcf_coinf01match"].isin(["positif", "négatif", "TVide"])]

    failed_or_warning_samples = pd.DataFrame()

    if "summary_bam_verif" in non_temoin_df_filtered.columns:
        failed_or_warning_samples = non_temoin_df_filtered[non_temoin_df_filtered["summary_bam_verif"].str.upper().isin(["FAILED", "WARNING"])]

    if not failed_or_warning_samples.empty:
        failed_sample_ids = failed_or_warning_samples["sample_id"].tolist()
        st.markdown("#### 📉 Échantillons avec vérification échouée (FAILED ou WARNING) dans la colonne bam_verif :")
        st.write(failed_sample_ids)

        non_temoin_df_filtered = non_temoin_df_filtered[~non_temoin_df_filtered["sample_id"].isin(failed_sample_ids)]
    else:
        st.warning("⚠️ La colonne 'bam_verif' est manquante dans les données.")

    # Coinfection inter-clade
    st.markdown("### 🧬 Coinfection inter-clade")

    if "summary_vcf_coinf02iqr" in non_temoin_df.columns:
        non_temoin_df["summary_vcf_coinf02iqr"] = pd.to_numeric(non_temoin_df["summary_vcf_coinf02iqr"], errors="coerce")

        interclade_samples = non_temoin_df[non_temoin_df["summary_vcf_coinf02iqr"] > 0]

        if not interclade_samples.empty:
            sample_ids_interclade = interclade_samples["sample_id"].tolist()
            st.markdown("#### 🔍 Échantillons détectés avec un IQR > 0 dans vcf_coinf02iqr (possibles coinfections inter-clade) :")
            st.write(sample_ids_interclade)

            st.dataframe(interclade_samples, use_container_width=True)

            st.download_button(
                label="📥 Télécharger les échantillons inter-clade en CSV",
                data=interclade_samples.to_csv(index=False),
                file_name="echantillons_interclade.csv",
                mime="text/csv"
            )
        else:
            st.info("✅ Aucun échantillon avec IQR > 0 détecté dans vcf_coinf02iqr.")
    else:
        st.warning("⚠️ La colonne vcf_coinf02iqr est absente du fichier chargé.")

    st.subheader(f"🧠 Avis Bio - Plaque {plaque_selectionnee}")

    if "val_avisbio" in df_plaque.columns:
        avisbio_df = df_plaque[df_plaque["val_avisbio"] == "AVISBIO_POI"]
        if not avisbio_df.empty:
            st.markdown("### 📌 Échantillons avec un *Avis Bio* (POI)")
            st.dataframe(avisbio_df[["sample_id", "val_avisbio", "val_poi"]])
        else:
            st.info("Aucun échantillon avec un avis biologique 'AVISBIO_POI' sur cette plaque.")
    else:
        st.warning("Colonne 'val_avisbio' absente du fichier.")

    if "nextclade_frameShifts" in df_plaque.columns:
        frameshift_df = df_plaque.dropna(subset=["nextclade_frameShifts"])
        if not frameshift_df.empty:
            st.markdown("### 🔍 Échantillons avec *FrameShifts* détectés")
            st.dataframe(frameshift_df[["sample_id", "nextclade_frameShifts"]])
        else:
            st.info("Aucun frameshift détecté sur cette plaque.")
    else:
        st.warning("Colonne 'nextclade_frameShifts' absente du fichier.")

    insertions_df = df_plaque.dropna(subset=["val_insertions"])
    if not insertions_df.empty:
        st.markdown("### 🧬 Échantillons avec *Insertions* détectées")
        st.dataframe(insertions_df[["sample_id", "val_insertions"]])
    else:
        st.info("Aucune insertion détectée sur cette plaque.")

    if "nextclade_qc_overallStatus" in df_plaque.columns:
        qc_status_df = df_plaque[df_plaque["nextclade_qc_overallStatus"].notna() & (df_plaque["nextclade_qc_overallStatus"].str.lower() != "good")]
        if not qc_status_df.empty:
            st.markdown("### ⚠️ Échantillons avec un statut *QC Nextclade* signalé")
            st.dataframe(qc_status_df[["sample_id", "nextclade_qc_overallStatus"]])
        else:
            st.info("Tous les échantillons ont un statut QC normal ('good') ou vide sur cette plaque.")
    else:
        st.warning("Colonne 'nextclade_qc_overallStatus' absente du fichier.")

    # Camembert de répartition
    st.markdown("#### 📊 Répartition des Résultats (hors témoins)")
    non_temoin_plaque_df = non_temoin_df[non_temoin_df["plaque_id"] == plaque_selectionnee]

    if not non_temoin_plaque_df.empty:
        result_counts = non_temoin_plaque_df['summary_vcf_coinf01match'].value_counts()

        fig_pie = go.Figure(data=[go.Pie(labels=result_counts.index, values=result_counts.values, hole=0.3)])
        fig_pie.update_traces(hoverinfo="label+percent+value", textinfo="label+percent")
        fig_pie.update_layout(title="Distribution des Résultats", showlegend=True)

        st.plotly_chart(fig_pie, use_container_width=True)

        selected_label = st.selectbox("Sélectionner un résultat pour voir les échantillons :", result_counts.index)

        filtered_by_label = non_temoin_plaque_df[non_temoin_plaque_df["summary_vcf_coinf01match"] == selected_label]

        st.markdown(f"### 🧾 Échantillons avec le résultat **{selected_label}**")
        st.dataframe(filtered_by_label, use_container_width=True)

        st.download_button(
            label=f"📥 Télécharger les échantillons '{selected_label}' en CSV",
            data=filtered_by_label.to_csv(index=False),
            file_name=f"echantillons_{selected_label}.csv",
            mime="text/csv"
        )
    else:
        st.warning("⚠️ Aucun échantillon non témoin détecté pour cette plaque.")

else:
    st.info("📂 Veuillez charger un fichier TSV pour activer l’analyse par plaque.")

# Nettoyage
if 'summary_consensus_perccoverage_S4' in base_df.columns:
    base_df["summary_consensus_perccoverage_S4"] = pd.to_numeric(base_df["summary_consensus_perccoverage_S4"], errors="coerce")
else:
    base_df["summary_consensus_perccoverage_S4"] = pd.NA

if 'summary_consensus_perccoverage_S6' in base_df.columns:
    base_df["summary_consensus_perccoverage_S6"] = pd.to_numeric(base_df["summary_consensus_perccoverage_S6"], errors="coerce")
else:
    base_df["summary_consensus_perccoverage_S6"] = pd.NA

if 'summary_run_id' in base_df.columns:
    base_df["run_short_id"] = base_df["summary_run_id"].str[:6]
else:
    base_df["run_short_id"] = pd.NA

if 'summary_run_id' in base_df.columns:
    base_df['Date'] = pd.to_datetime(base_df['summary_run_id'].apply(lambda x: x[:6] if isinstance(x, str) else None), format='%Y%m', errors='coerce')
else:
    base_df['Date'] = pd.NA

# Compteur
if 'sample_id' in base_df.columns:
    samples_with_tvide = base_df[base_df['sample_id'].str.contains('Tvide', case=False, na=False)]
    samples_without_tvide = base_df[~base_df['sample_id'].str.contains('Tvide', case=False, na=False)]

    st.subheader("🔢 Compteur d'échantillons importés")
    st.write(f"Nombre d'échantillons avec Tvide : {len(samples_with_tvide)}")
    st.write(f"Nombre d'échantillons sans Tvide : {len(samples_without_tvide)}")
else:
    st.error("❌ Erreur : La colonne 'sample_id' est manquante dans le DataFrame.")

# Filtrer les runs
if 'summary_run_id' in base_df.columns:
    with st.expander("🧪 Filtrer les runs affichés"):
        run_ids = sorted(base_df["summary_run_id"].dropna().unique())
        select_all = st.checkbox("✅ Tout sélectionner", value=True, key="select_all_checkbox")
        if select_all:
            selected_runs = run_ids
        else:
            selected_runs = st.multiselect("Sélectionner des run_id :", run_ids, default=[], key="run_id_selector")

    filtered_df = base_df[base_df["summary_run_id"].isin(selected_runs)]
else:
    st.error("❌ Erreur : La colonne 'summary_run_id' est manquante dans le DataFrame.")
    filtered_df = pd.DataFrame()

filtered_df["plaque_id"] = filtered_df["sample_id"].apply(extraire_plaque)

# Utiliser le cache pour les commentaires
@st.cache_data
def load_comments(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return pd.DataFrame()

# Sauvegarder les commentaires
def save_comments(df, file_path):
    df.to_csv(file_path, index=False)

# Charger les commentaires
comments_df = load_comments(DATA_FILE)

# Assurez-vous que la colonne plaque_id existe dans comments_df
if 'plaque_id' not in comments_df.columns and 'sample_id' in comments_df.columns:
    comments_df['plaque_id'] = comments_df['sample_id'].apply(extraire_plaque)

# Vérifiez que la colonne plaque_id est bien remplie
if not comments_df.empty:
    print(comments_df[['sample_id', 'plaque_id']].head())

# Ajouter / Modifier un commentaire par Plaque ID
if not filtered_df.empty and 'plaque_id' in filtered_df.columns:
    st.subheader("📝 Ajouter ou modifier un commentaire (par Plaque ID)")

    plaques_disponibles = sorted(filtered_df["plaque_id"].dropna().unique())
    plaque_selectionnee = st.selectbox("Sélectionner une Plaque ID :", plaques_disponibles)

    samples_in_plaque = filtered_df[filtered_df["plaque_id"] == plaque_selectionnee]

    if not samples_in_plaque.empty:
        sample_id = st.selectbox(
            "Sélectionner un échantillon à commenter :",
            sorted(samples_in_plaque['sample_id'].unique())
        )

        # Charger le commentaire actuel
        current_comment = comments_df.loc[
            (comments_df['plaque_id'] == plaque_selectionnee) & (comments_df['sample_id'] == sample_id),
            'commentaire'
        ].values[0] if not comments_df[(comments_df['plaque_id'] == plaque_selectionnee) & (comments_df['sample_id'] == sample_id)].empty else ""

        new_comment = st.text_area("Modifier le commentaire :", value=current_comment)

        if new_comment != current_comment:
            # Mettre à jour le commentaire
            if comments_df[(comments_df['plaque_id'] == plaque_selectionnee) & (comments_df['sample_id'] == sample_id)].empty:
                new_row = pd.DataFrame({
                    'plaque_id': [plaque_selectionnee],
                    'sample_id': [sample_id],
                    'commentaire': [new_comment]
                })
                comments_df = pd.concat([comments_df, new_row], ignore_index=True)
            else:
                comments_df.loc[
                    (comments_df['plaque_id'] == plaque_selectionnee) & (comments_df['sample_id'] == sample_id),
                    'commentaire'
                ] = new_comment
            save_comments(comments_df, DATA_FILE)
            st.success("✅ Commentaire mis à jour.")
    else:
        st.info("⚠️ Aucun échantillon trouvé pour cette plaque.")
else:
    st.info("⚠️ Aucun échantillon trouvé pour cette plaque.")

# Fonction pour afficher les statistiques et graphiques avec les ajouts
def plot_couverture_par_temoin_par_plaque(df):
    st.subheader("🧬 Couverture des fragments S1 à S8 par plaque et par témoin")

    fragments = [f"summary_consensus_perccoverage_S{i}" for i in range(1, 9)]
    temoins = ["TposH3", "TposH1", "TposB"]

    for temoin in temoins:
        st.markdown(f"### 🔬 {temoin}")
        subset = df[df["sample_id"].str.contains(temoin, case=False, na=False)]
        if subset.empty:
            st.warning(f"Aucun échantillon trouvé pour le témoin {temoin}.")
            continue

        subset = subset.dropna(subset=["plaque_id"] + fragments)
        subset = subset.sort_values(by="plaque_id")

        fig = go.Figure()
        for fragment in fragments:
            if fragment == "summary_consensus_perccoverage_S4":
                line_style = dict(width=4, color='red')
            elif fragment == "summary_consensus_perccoverage_S6":
                line_style = dict(width=4, color='orange')
            else:
                line_style = dict(width=2, color='lightgray')

            fig.add_trace(go.Scatter(
                x=subset["plaque_id"],
                y=subset[fragment],
                mode="lines+markers",
                name=fragment.replace("summary_consensus_perccoverage_", "S"),
                line=line_style
            ))

        for i, row in subset.iterrows():
            commentaire = row.get("commentaire", "")
            x_value = row.get("plaque_id")
            y_value = row.get(fragment)
            sample_id = row.get("sample_id")

            if pd.notna(commentaire) and commentaire.strip() != "" and pd.notna(y_value) and pd.notna(x_value):
                fig.add_annotation(
                    x=x_value,
                    y=y_value,
                    text=f"{sample_id}: {commentaire}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    ax=0,
                    ay=-40,
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255, 255, 255, 0.7)"
                )

        fig.add_hline(
            y=90,
            line=dict(color="red", dash="dash"),
            annotation_text="Seuil 90%",
            annotation_position="top left",
            line_width=2
        )

        fig.update_layout(
            title=f"Couverture des fragments S1 à S8 - {temoin} (par plaque)",
            xaxis_title="Plaque ID",
            yaxis_title="% Couverture",
            yaxis=dict(range=[0, 110]),
            xaxis=dict(
                tickangle=-45,
                tickmode='array',
                tickvals=subset["plaque_id"],
                ticktext=subset["plaque_id"].astype(str),
            ),
            legend_title="Fragments",
            margin=dict(l=20, r=20, t=50, b=60),
        )

        st.plotly_chart(fig, use_container_width=True)

# Appeler la fonction
if not filtered_df.empty:
    plot_couverture_par_temoin_par_plaque(filtered_df)

# % d'échantillons ininterprétables par run
st.subheader("📊 Pourcentage d'échantillons ininterprétables par run")

ininterpretable_or_failed_df = base_df[
    ((base_df['summary_consensus_perccoverage_S4'] < 90) |
     (base_df['summary_consensus_perccoverage_S6'] < 90) |
     (base_df['summary_qc_seqcontrol'].str.upper().isin(["FAILED", "0"])))
]

total_by_run = base_df.groupby('summary_run_id').size()

ininterpretable_or_failed_by_run = ininterpretable_or_failed_df.groupby('summary_run_id').size()

percentage_ininterpretable_or_failed = (ininterpretable_or_failed_by_run / total_by_run) * 100

percentage_ininterpretable_or_failed = percentage_ininterpretable_or_failed.reindex(total_by_run.index, fill_value=0)

fig = go.Figure(go.Bar(x=percentage_ininterpretable_or_failed.index, y=percentage_ininterpretable_or_failed.values, name="Ininterprétables ou Seq failed (%)"))

fig.update_layout(
    title="Pourcentage d'échantillons ininterprétables ou Seq failed par run",
    xaxis_title="Run ID",
    yaxis_title="Ininterprétables ou Seq failed (%)",
    xaxis=dict(tickangle=-45),
    margin=dict(l=0, r=0, t=30, b=50)
)

st.plotly_chart(fig, use_container_width=True)
