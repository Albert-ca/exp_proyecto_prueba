
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, chi2_contingency
from statsmodels.stats.multitest import multipletests

def analizar_mecanismos_nulos(df, alpha=0.05, min_grupo=10):

    resultados = []
    resumen = []

    # Columnas que tienen valores nulos
    columnas_nulas = df.columns[df.isnull().any()].tolist()

    for target in columnas_nulas:

        # Indicador de ausencia: 1 = nulo, 0 = presente
        indicador = df[target].isnull()

        nulos = indicador.sum()
        porcentaje = indicador.mean() * 100

        pruebas = []

        # Analizar las demás variables
        for variable in df.columns:

            if variable == target:
                continue

            # Registros donde la variable predictora está disponible
            datos = pd.DataFrame({
                "x": df[variable],
                "missing": indicador
            }).dropna(subset=["x"])

            grupo_nulo = datos.loc[
                datos["missing"], "x"
            ]

            grupo_presente = datos.loc[
                ~datos["missing"], "x"
            ]

            if (len(grupo_nulo) < min_grupo or
                len(grupo_presente) < min_grupo):
                continue

            try:

                # Variables numéricas
                if pd.api.types.is_numeric_dtype(datos["x"]):

                    if datos["x"].nunique() < 2:
                        continue

                    estadistico, pvalor = mannwhitneyu(
                        grupo_nulo,
                        grupo_presente,
                        alternative="two-sided"
                    )

                    tipo_prueba = "Mann-Whitney U"

                # Variables categóricas
                else:

                    tabla = pd.crosstab(
                        datos["x"],
                        datos["missing"]
                    )

                    if tabla.shape[0] < 2 or tabla.shape[1] < 2:
                        continue

                    chi2, pvalor, _, _ = chi2_contingency(
                        tabla
                    )

                    tipo_prueba = "Chi-cuadrado"

                pruebas.append({
                    "variable_nula": target,
                    "variable_explicativa": variable,
                    "tipo_prueba": tipo_prueba,
                    "pvalor": pvalor,
                    "n_muestra": len(datos)
                })

            except (ValueError, TypeError):
                continue

        # Corrección por múltiples comparaciones
        if pruebas:

            pvalores = [p["pvalor"] for p in pruebas]

            _, p_ajustados, _, _ = multipletests(
                pvalores,
                alpha=alpha,
                method="fdr_bh"
            )

            for prueba, p_ajustado in zip(
                pruebas, p_ajustados
            ):

                prueba["pvalor_ajustado"] = p_ajustado

                prueba["asociacion"] = (
                    "Significativa"
                    if p_ajustado < alpha
                    else "No significativa"
                )

                resultados.append(prueba)

            significativas = sum(
                p < alpha for p in p_ajustados
            )

        else:
            significativas = 0

        # Diagnóstico orientativo
        if significativas > 0:

            diagnostico = "MCAR rechazada - compatible con MAR"

        else:

            diagnostico = (
                "Sin evidencia contra MCAR - no concluyente"
            )

        resumen.append({
            "variable": target,
            "nulos": nulos,
            "porcentaje_nulos": round(porcentaje, 2),
            "variables_evaluadas": len(pruebas),
            "asociaciones_significativas": significativas,
            "diagnostico": diagnostico
        })

    df_resumen = pd.DataFrame(resumen)

    df_detalle = pd.DataFrame(resultados)

    if not df_detalle.empty:
        df_detalle = df_detalle.sort_values(
            ["variable_nula", "pvalor_ajustado"]
        )

    return df_resumen, df_detalle
