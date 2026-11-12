"""Quand semer ? — prototype de l'outil d'aide à la décision agricole (Angers, 30 km).

DONNÉES FICTIVES : les températures sont générées pour ressembler au climat angevin.
Elles seront remplacées par les données réelles (Météo-France) et les projections (DRIAS).

Lancement en local :  streamlit run app.py
"""
import itertools

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Quand semer ? — prototype", page_icon="🌱", layout="wide")

MOIS = ["janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc."]
VARS = {"T min (°C)": "tmin", "T max (°C)": "tmax", "T moyenne (°C)": "tmoy", "Degrés-jours cumulés": "dj"}
MODES = ["Date fixe (tester chaque jour)", "Automatique : température", "Automatique : cumul de degrés-jours"]
VERT, ORANGE, ROUGE = "#5DAE5B", "#F2A93B", "#E06C6C"
ANNEE_DEBUT, ANNEE_FIN = 1996, 2025


# ---------------------------------------------------------------- données
@st.cache_data
def donnees():
    """Températures quotidiennes FICTIVES (2 stations moyennées), forme proche du climat angevin."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(f"{ANNEE_DEBUT}-01-01", f"{ANNEE_FIN}-12-31", freq="D")
    doy = dates.dayofyear.to_numpy()
    clim = 12.7 - 7.3 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    amp = 9.5 - 2.0 * np.cos(2 * np.pi * (doy - 15) / 365.25)
    tendance = 0.04 * (dates.year.to_numpy() - ANNEE_DEBUT)  # léger réchauffement
    tmins, tmaxs = [], []
    for decalage in (0.0, 0.3):
        eps = rng.normal(0, 1.6, len(dates))
        bruit = np.zeros(len(dates))
        for t in range(1, len(dates)):
            bruit[t] = 0.8 * bruit[t - 1] + eps[t]
        tmoy = clim + tendance + bruit + decalage
        a = amp + rng.normal(0, 1.5, len(dates))
        nuits_froides = rng.exponential(4.0, len(dates)) * (rng.random(len(dates)) < 0.25)
        tmins.append(tmoy - a / 2 - nuits_froides)
        tmaxs.append(tmoy + a / 2)
    df = pd.DataFrame({"tmin": np.mean(tmins, 0), "tmax": np.mean(tmaxs, 0)}, index=dates).round(1)
    df["tmoy"] = (df.tmin + df.tmax) / 2
    return df


# ---------------------------------------------------------------- préréglages (fiches culture)
PRESETS = {
    "Saisie manuelle": dict(
        mode=MODES[0], seuil_t=10.0, jours_t=10, base_dj=10.0, seuil_dj=150.0, mois=MOIS[2:8],
        conditions=[dict(var="T moyenne (°C)", op="≥", seuil=15.0, duree=20, base=10.0)]),
    "Vigne – bourgeons (exemple illustratif)": dict(
        mode=MODES[2], seuil_t=10.0, jours_t=10, base_dj=10.0, seuil_dj=20.0, mois=MOIS[1:5],
        conditions=[dict(var="T min (°C)", op="≥", seuil=-1.0, duree=30, base=10.0)]),
    "Culture test – semis (exemple illustratif)": dict(
        mode=MODES[1], seuil_t=10.0, jours_t=10, base_dj=6.0, seuil_dj=150.0, mois=MOIS[2:7],
        conditions=[dict(var="Degrés-jours cumulés", op="≥", seuil=900.0, duree=90, base=6.0),
                    dict(var="T min (°C)", op="≥", seuil=0.0, duree=90, base=6.0)]),
}
_ids = itertools.count(1000)


def appliquer_preset():
    p = PRESETS[st.session_state.culture]
    for k in ("mode", "seuil_t", "jours_t", "base_dj", "seuil_dj", "mois"):
        st.session_state[k] = p[k]
    st.session_state.conds = []
    for c in p["conditions"]:
        ajouter_condition(c)
    st.session_state.pop("resultat", None)


def ajouter_condition(c=None):
    c = c or dict(var="T min (°C)", op="≥", seuil=0.0, duree=30, base=10.0)
    i = next(_ids)
    st.session_state.conds.append(i)
    st.session_state[f"var{i}"], st.session_state[f"op{i}"] = c["var"], c["op"]
    st.session_state[f"seuil{i}"], st.session_state[f"duree{i}"] = float(c["seuil"]), int(c["duree"])
    st.session_state[f"base{i}"] = float(c["base"])


def supprimer_condition(i):
    st.session_state.conds.remove(i)


if "conds" not in st.session_state:
    st.session_state.culture = "Saisie manuelle"
    appliquer_preset()


# ---------------------------------------------------------------- calcul
def tester_conditions(df, conds, debuts):
    """Pour chaque position de début : True si toutes les conditions sont respectées."""
    n = len(df)
    ok_total = np.ones(len(debuts), bool)
    detail = {}
    for c in conds:
        d = c["duree"]
        fin = np.minimum(debuts + d, n)
        if VARS[c["var"]] == "dj":
            val = np.clip(df.tmoy.to_numpy() - c["base"], 0, None)
            cs = np.concatenate([[0.0], np.cumsum(val)])
            somme = cs[fin] - cs[debuts]
            ok = somme >= c["seuil"] if c["op"] == "≥" else somme <= c["seuil"]
        else:
            v = df[VARS[c["var"]]].to_numpy()
            jour_ok = v >= c["seuil"] if c["op"] == "≥" else v <= c["seuil"]
            cs = np.concatenate([[0], np.cumsum(jour_ok)])
            ok = (cs[fin] - cs[debuts]) == d
        ok &= (debuts + d) <= n
        detail[c["libelle"]] = ok
        ok_total &= ok
    return ok_total, detail


def decade(ts):
    return f"{ts.month:02d}-{min((ts.day - 1) // 10 + 1, 3)}"


def libelle_decade(code):
    m, d = int(code[:2]), code[3]
    return {"1": "1–10", "2": "11–20", "3": "21–fin"}[d] + " " + MOIS[m - 1]


def classe(p, fav, risque):
    return "Favorable" if p >= fav else ("À risque" if p >= risque else "Défavorable")


def calcul_date_fixe(df, conds, mois_num):
    masque = df.index.month.isin(mois_num)
    debuts = np.flatnonzero(masque)
    ok, detail = tester_conditions(df, conds, debuts)
    r = pd.DataFrame({"date": df.index[debuts], "reussite": ok, **detail})
    r["annee"] = r.date.dt.year
    r["decade"] = r.date.map(decade)
    return r


def calcul_automatique(df, conds, mois_num, mode, seuil_t, jours_t, base_dj, seuil_dj):
    lignes = []
    tmoy = df.tmoy.to_numpy()
    for annee in range(ANNEE_DEBUT, ANNEE_FIN + 1):
        pos_annee = np.flatnonzero(df.index.year == annee)
        dans_mois = pos_annee[np.isin(df.index.month[pos_annee], mois_num)]
        a, b = dans_mois[0], dans_mois[-1]
        debut = None
        if mode == MODES[1]:
            chaud = np.concatenate([[0], np.cumsum(tmoy > seuil_t)])
            for i in range(max(a, jours_t - 1), b + 1):
                if chaud[i + 1] - chaud[i + 1 - jours_t] == jours_t:
                    debut = i
                    break
        else:
            cumul = np.cumsum(np.clip(tmoy[pos_annee] - base_dj, 0, None))
            atteint = np.flatnonzero(cumul >= seuil_dj)
            if len(atteint) and a <= pos_annee[atteint[0]] <= b:
                debut = pos_annee[atteint[0]]
        ligne = {"annee": annee, "date": df.index[debut] if debut is not None else pd.NaT}
        if debut is not None:
            ok, detail = tester_conditions(df, conds, np.array([debut]))
            ligne["reussite"] = bool(ok[0])
            ligne.update({k: bool(v[0]) for k, v in detail.items()})
        else:
            ligne["reussite"] = False
        lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------- interface
st.title("🌱 Quand semer ? — prototype")
st.warning("**Données FICTIVES.** Les températures sont générées pour tester la logique de l'outil. "
           "Les résultats ne sont pas des conseils agronomiques.", icon="⚠️")

gauche, droite = st.columns([1, 1.6], gap="large")

with gauche:
    st.subheader("Décrivez votre culture")

    st.markdown("**① Lieu**")
    st.selectbox("Lieu", ["Angers – rayon 30 km"], label_visibility="collapsed")

    st.markdown("**② Culture** (facultatif : pré-remplit les champs ③ à ⑤)")
    st.selectbox("Culture", list(PRESETS), key="culture", on_change=appliquer_preset,
                 label_visibility="collapsed")

    st.markdown("**③ Démarrage du cycle**")
    mode = st.radio("Démarrage", MODES, key="mode", label_visibility="collapsed")
    if mode == MODES[1]:
        c1, c2 = st.columns(2)
        c1.number_input("dès que T moyenne > (°C)", key="seuil_t", step=1.0)
        c2.number_input("pendant (jours consécutifs)", key="jours_t", min_value=1, step=1)
    elif mode == MODES[2]:
        c1, c2 = st.columns(2)
        c1.number_input("cumul depuis le 1er janvier ≥ (degrés-jours)", key="seuil_dj", step=10.0)
        c2.number_input("en base (°C)", key="base_dj", step=1.0)
    else:
        st.caption("Chaque jour des mois choisis est testé comme date de semis possible.")

    st.markdown("**④ Besoins de la plante** (à respecter après le démarrage)")
    for i in list(st.session_state.conds):
        c = st.columns([2.2, 0.8, 1.1, 1.1, 0.5])
        c[0].selectbox("Variable", list(VARS), key=f"var{i}", label_visibility="collapsed")
        c[1].selectbox("Op", ["≥", "≤"], key=f"op{i}", label_visibility="collapsed")
        c[2].number_input("Seuil", key=f"seuil{i}", step=1.0, label_visibility="collapsed")
        c[3].number_input("jours", key=f"duree{i}", min_value=1, step=1, label_visibility="collapsed")
        c[4].button("−", key=f"del{i}", on_click=supprimer_condition, args=(i,), help="Supprimer")
        if st.session_state[f"var{i}"] == "Degrés-jours cumulés":
            st.number_input("↳ température de base (°C)", key=f"base{i}", step=1.0)
    st.caption("Lecture d'une ligne : variable · opérateur · seuil · pendant N jours.")
    st.button("+ Ajouter une condition", on_click=ajouter_condition)

    st.markdown("**⑤ Mois à étudier**")
    st.multiselect("Mois", MOIS, key="mois", label_visibility="collapsed")

    st.markdown("**⑥ Évolution**")
    evolution = st.checkbox("Montrer l'évolution dans le temps", value=True)

    with st.expander("Réglages des classes"):
        fav = st.slider("Favorable si probabilité ≥", 0.5, 1.0, 0.8, 0.05)
        risque = st.slider("À risque si probabilité ≥", 0.0, 0.8, 0.5, 0.05)

    lancer = st.button("⑦ Lancer l'analyse", type="primary", width="stretch")


def lire_conditions():
    out = []
    for i in st.session_state.conds:
        c = dict(var=st.session_state[f"var{i}"], op=st.session_state[f"op{i}"],
                 seuil=st.session_state[f"seuil{i}"], duree=int(st.session_state[f"duree{i}"]),
                 base=st.session_state[f"base{i}"])
        unite = "DJ" if VARS[c["var"]] == "dj" else "°C"
        nom = c["var"].split(" (")[0]
        c["libelle"] = f"{nom} {c['op']} {c['seuil']:g} {unite} sur {c['duree']} j"
        out.append(c)
    return out


with droite:
    if lancer:
        conds = lire_conditions()
        mois_num = sorted(MOIS.index(m) + 1 for m in st.session_state.mois)
        if not conds:
            st.error("Ajoutez au moins une condition.")
        elif not mois_num:
            st.error("Choisissez au moins un mois.")
        else:
            df = donnees()
            if mode == MODES[0]:
                r = calcul_date_fixe(df, conds, mois_num)
            else:
                r = calcul_automatique(df, conds, mois_num, mode, st.session_state.seuil_t,
                                       int(st.session_state.jours_t), st.session_state.base_dj,
                                       st.session_state.seuil_dj)
            st.session_state.resultat = dict(r=r, mode=mode, conds=conds)

    res = st.session_state.get("resultat")
    if not res:
        st.info("Remplissez le formulaire puis cliquez sur **Lancer l'analyse**.")
    elif res["mode"] == MODES[0]:
        # ------------------------------------------------ résultat : date fixe
        r, conds = res["r"], res["conds"]
        g = r.groupby("decade").agg(p=("reussite", "mean")).reset_index()
        g["periode"] = g.decade.map(libelle_decade)
        g["Classe"] = g.p.map(lambda p: classe(p, fav, risque))
        g["%"] = (g.p * 100).round(0)
        meilleure = g.loc[g.p.idxmax()]
        favorables = g[g.Classe == "Favorable"]
        st.subheader("Quand semer ?")
        m1, m2, m3 = st.columns(3)
        m1.metric("Meilleure période", meilleure.periode)
        m2.metric("Probabilité", f"{meilleure['%']:.0f} %")
        m3.metric("Décades favorables", f"{len(favorables)} / {len(g)}")
        st.caption(f"Probabilité = part des {r.annee.nunique()} années ({ANNEE_DEBUT}–{ANNEE_FIN}) "
                   "où toutes les conditions ont été respectées, pour une date de semis dans la décade.")
        ordre = list(g.periode)
        barres = alt.Chart(g).mark_bar().encode(
            x=alt.X("periode:N", sort=ordre, title=None, axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("%:Q", scale=alt.Scale(domain=[0, 100]), title="Probabilité (%)"),
            color=alt.Color("Classe:N", scale=alt.Scale(domain=["Favorable", "À risque", "Défavorable"],
                                                        range=[VERT, ORANGE, ROUGE])),
            tooltip=["periode", "%", "Classe"])
        seuils = alt.Chart(pd.DataFrame({"y": [fav * 100, risque * 100]})).mark_rule(
            strokeDash=[4, 4], color="gray").encode(y="y:Q")
        st.altair_chart(barres + seuils, width="stretch")

        with st.expander("Quelle condition bloque ? (détail par condition)"):
            det = r.groupby("decade")[[c["libelle"] for c in conds]].mean().mul(100).round(0)
            det.index = det.index.map(libelle_decade)
            st.dataframe(det.astype(int).astype(str) + " %", width="stretch")

        if evolution:
            st.subheader("Est-ce que ça change ?")
            r["Période"] = np.where(r.annee <= 2010, f"{ANNEE_DEBUT}–2010", f"2011–{ANNEE_FIN}")
            e = r.groupby(["decade", "Période"]).reussite.mean().mul(100).round(0).reset_index()
            e["periode"] = e.decade.map(libelle_decade)
            ch = alt.Chart(e).mark_bar().encode(
                x=alt.X("periode:N", sort=ordre, title=None, axis=alt.Axis(labelAngle=-45)),
                xOffset="Période:N",
                y=alt.Y("reussite:Q", title="Probabilité (%)", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Période:N", scale=alt.Scale(range=["#B0BEC5", "#3B7DD8"])),
                tooltip=["periode", "Période", "reussite"])
            st.altair_chart(ch, width="stretch")
            st.caption("Comparaison des 15 premières et 15 dernières années. "
                       "Les projections futures (DRIAS) viendront compléter ce graphique.")
    else:
        # ------------------------------------------------ résultat : démarrage automatique
        r, conds = res["r"], res["conds"]
        avec = r.dropna(subset=["date"])
        p = r.reussite.mean()
        st.subheader("Quand démarre le cycle, et ça se passe bien ?")
        m1, m2, m3 = st.columns(3)
        if len(avec):
            med = int(avec.date.dt.dayofyear.median())
            d_med = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=med - 1)
            m1.metric("Démarrage habituel (médiane)", f"{d_med.day} {MOIS[d_med.month - 1]}")
        else:
            m1.metric("Démarrage habituel", "jamais")
        m2.metric("Réussite", f"{p * 100:.0f} % des années", help=classe(p, fav, risque))
        m3.metric("Années sans démarrage", f"{r.date.isna().sum()} / {len(r)}")
        st.markdown(f"**Verdict : {classe(p, fav, risque)}** — conditions respectées "
                    f"{int(r.reussite.sum())} années sur {len(r)}.")

        if len(avec):
            avec = avec.assign(decade=avec.date.map(decade),
                               Résultat=np.where(avec.reussite, "Réussi", "Raté"))
            avec["periode"] = avec.decade.map(libelle_decade)
            ordre = [libelle_decade(d) for d in sorted(avec.decade.unique())]
            ch = alt.Chart(avec).mark_bar().encode(
                x=alt.X("periode:N", sort=ordre, title="Date de démarrage", axis=alt.Axis(labelAngle=-45)),
                y=alt.Y("count():Q", title="Nombre d'années"),
                color=alt.Color("Résultat:N", scale=alt.Scale(domain=["Réussi", "Raté"], range=[VERT, ROUGE])),
                tooltip=["periode", "Résultat", "count()"])
            st.altair_chart(ch, width="stretch")

        with st.expander("Détail année par année"):
            t = r.copy()
            t["date"] = t.date.map(lambda d: "—" if pd.isna(d) else f"{d.day} {MOIS[d.month - 1]}")
            t = t.rename(columns={"annee": "Année", "date": "Démarrage", "reussite": "Réussi"})
            st.dataframe(t.replace({True: "✅", False: "❌"}).fillna("—"),
                         hide_index=True, width="stretch")

        if evolution and len(avec) >= 3:
            st.subheader("Est-ce que ça change ?")
            avec["jour"] = avec.date.dt.dayofyear
            pente = np.polyfit(avec.annee, avec.jour, 1)[0] * 10
            base = alt.Chart(avec).encode(x=alt.X("annee:Q", title="Année", scale=alt.Scale(zero=False),
                                                  axis=alt.Axis(format="d")))
            pts = base.mark_circle(size=70).encode(
                y=alt.Y("jour:Q", title="Jour de démarrage (jour de l'année)", scale=alt.Scale(zero=False)),
                color=alt.Color("Résultat:N", scale=alt.Scale(domain=["Réussi", "Raté"], range=[VERT, ROUGE])),
                tooltip=["annee", "periode", "Résultat"])
            tend = base.transform_regression("annee", "jour").mark_line(color="gray", strokeDash=[4, 4]).encode(y="jour:Q")
            st.altair_chart(pts + tend, width="stretch")
            sens = "plus tôt" if pente < 0 else "plus tard"
            st.caption(f"Tendance : le démarrage arrive environ {abs(pente):.1f} jours {sens} "
                       "par décennie (données fictives). Les projections futures (DRIAS) prolongeront cette courbe.")

st.divider()
st.caption("Prototype ESEO — partie Data. Méthode : pour chaque année de l'historique, on vérifie les conditions ; "
           "la probabilité est la part des années où elles sont toutes respectées. "
           "Degrés-jours = somme de max(0, T moyenne − T base).")
