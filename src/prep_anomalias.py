
#librerias basicas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
def detectar_anomalias(
    df,
    contamination=0.02,
    iqr_factor=1.5,
    mad_threshold=3.5,
    random_state=42,
    excluir=None
):

    excluir = set(excluir or [])

    # Detectar identificadores por nombre
    patrones_id = ("ID", "CODIGO", "COD_", "DNI")

    excluir.update([ # evitando que los identificadores participen en la detección.
        c for c in df.columns
        if any(p in c.upper() for p in patrones_id)
    ])

    # Columnas numéricas
    numericas = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    columnas = [
        c for c in numericas
        if c not in excluir
        and df[c].nunique(dropna=True) > 10 #Esto significa que solo analizas columnas numéricas que tengan más de 10 valores distintos.
    ]

    if not columnas:
        raise ValueError("No hay variables numéricas aptas para analizar.")

    # Copia de trabajo
    datos = df[columnas].replace(
        [np.inf, -np.inf], np.nan
    )

    resumen = []
    flags = pd.DataFrame(index=df.index)

    # ==========================================
    # 1. ANÁLISIS UNIVARIADO: IQR Y MAD
    # ==========================================
    # La palabra univariado significa que analizas una variable a la vez, sin considerar las demás.
    # Por ejemplo, puedes analizar si la velocidad de un cliente es inusualmente alta o baja sin importar cuánto paga o cuánto consume.
    for col in columnas:

        x = datos[col].dropna()

        if len(x) < 10:
            continue

        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)

        iqr = q3 - q1

        limite_inf = q1 - iqr_factor * iqr
        limite_sup = q3 + iqr_factor * iqr

        flag_iqr = ( #resultado de si es o no un outlier segun el metodo IQR, es decir si esta fuera de los limites inferior y superior
            (datos[col] < limite_inf) |
            (datos[col] > limite_sup)
        ).fillna(False)
        # Nota:  IQR identifica valores estadísticamente extremos, no necesariamente errores. Un cliente con 200 Mbps 
        # podría tener perfectamente ese servicio contratado.

        # MAD: desviación absoluta mediana
        # Esta técnica también identifica valores extremos, pero utiliza como referencia la mediana.
        mediana = x.median()

        mad = np.median(
            np.abs(x - mediana)
        )

        if mad > 0:

            robust_z = (
                0.67448975 *
                (datos[col] - mediana) / mad
            )

            flag_mad = (
                robust_z.abs() > mad_threshold
            ).fillna(False)

        else:
            robust_z = pd.Series(
                np.nan, index=df.index
            )

            flag_mad = pd.Series(
                False, index=df.index
            )

        flags[f"{col}_IQR"] = flag_iqr
        flags[f"{col}_MAD"] = flag_mad

        resumen.append({
            "variable": col,
            "n_validos": len(x),
            "n_nulos": df[col].isna().sum(),
            "mediana": mediana,
            "Q1": q1,
            "Q3": q3,
            "limite_inferior": limite_inf,
            "limite_superior": limite_sup,
            "outliers_IQR": int(flag_iqr.sum()),
            "porcentaje_IQR": round(
                flag_iqr.mean() * 100, 2
            ),
            "outliers_MAD": int(flag_mad.sum()),
            "porcentaje_MAD": round(
                flag_mad.mean() * 100, 2
            )
        })

    resumen_anomalias = pd.DataFrame(resumen)
    
    # ¿Por qué utilizar IQR y MAD simultáneamente?
    # Porque no identifican exactamente los mismos valores. Sus criterios de detección son diferentes.
    # IQR trabaja con los cuartiles y MAD trabaja con la dispersión absoluta respecto a la mediana.
    # Incluso pueden marcar un valor de forma distinta.

    # ==========================================
    # 2. ANÁLISIS MULTIVARIADO: ISOLATION FOREST
    # ==========================================
    # independiente mente cada una de sus features podria estar dentro de los limites de IQR y MAD, pero en conjunto pueden ser anómalas.
    columnas_if = [
        c for c in columnas
        if datos[c].nunique(dropna=True) > 2
    ]
    
    #preproceso de datos para el algoritmo Isolation Forest
    if len(columnas_if) >= 2: # Seleccionas variables con más de dos valores distintos.
        # Construyes la matriz que utilizará Isolation Forest.
        matriz = datos[columnas_if].copy()

        # Imputación temporal para el algoritmo
        matriz = SimpleImputer(
            strategy="median"  # Aquí reemplazas temporalmente los valores faltantes con la mediana de cada variable.
        ).fit_transform(matriz)
        # esta imputación es exclusivamente para que Isolation Forest pueda procesar la matriz. No modifica tu DataFrame original.
        # Sin embargo, para tu análisis de nulos, hay una consideración: un valor imputado puede influir en el resultado del detector. 
        # Por eso conviene conservar indicadores de ausencia y evaluar qué ocurre con los registros que tienen muchos nulos.
        
        # Escalamiento robusto
        # Este escalador transforma las variables considerando su mediana y su rango intercuartílico.
        # Esto ayuda a que las diferencias de escala no dominen la representación de los datos.
        
        matriz = RobustScaler().fit_transform(matriz)
        
        # Entrenamiento de Isolation Forest
        modelo = IsolationForest( 
            n_estimators=200, #Construye 200 árboles de aislamiento.
            contamination=contamination, # Asume aproximadamente 2% de anomalías para definir el umbral de clasificación.
            random_state=random_state,
            n_jobs=-1
        )
        # Esto no significa que el algoritmo haya descubierto que el 2% de tus clientes son anómalos.
        # Significa que estás indicando al modelo que utilice un umbral asociado a una proporción esperada de anomalías del 2%.
        
        predicciones = modelo.fit_predict(matriz)
        # El método fit_predict ajusta el modelo a los datos y devuelve una predicción para cada registro: -1 indica anomalía y 1 indica normalidad.
        flags["ISOLATION_FOREST"] = (
            predicciones == -1
        )

        scores = modelo.decision_function(matriz)
        # Aquí obtienes una puntuación:
        # Valores negativos: registros clasificados como anomalías.
        #Valores positivos: registros clasificados como normales.
        #Cuanto más negativo sea el valor, más anómalo es el registro respecto al umbral aprendido.

        flags["IF_SCORE"] = scores

    else:

        flags["ISOLATION_FOREST"] = False

    # ==========================================
    # 3. INDICADORES GENERALES POR REGISTRO
    # ==========================================

    columnas_iqr = [
        c for c in flags.columns
        if c.endswith("_IQR")
    ]

    columnas_mad = [
        c for c in flags.columns
        if c.endswith("_MAD")
    ]

    flags["TOTAL_OUTLIERS_IQR"] = ( # Esto cuenta cuántas variables fueron marcadas por IQR para cada cliente.
        flags[columnas_iqr].sum(axis=1)
        if columnas_iqr else 0
    )

    flags["TOTAL_OUTLIERS_MAD"] = ( #total para cada cliente de variables marcadas por MAD
        flags[columnas_mad].sum(axis=1)
        if columnas_mad else 0
    )

    flags["ANOMALIA_IF"] = ( # indica si el registro fue marcado como anómalo por Isolation Forest
        flags["ISOLATION_FOREST"].astype(int)
    )

    # Clasificación orientativa
    flags["NIVEL_ANOMALIA"] = np.select(
        [
            (flags["TOTAL_OUTLIERS_IQR"] +
             flags["TOTAL_OUTLIERS_MAD"] >= 3) |
            (flags["ANOMALIA_IF"] == 1),

            (flags["TOTAL_OUTLIERS_IQR"] +
             flags["TOTAL_OUTLIERS_MAD"] >= 1)
        ],
        [
            "Revisar",
            "Moderado"
        ],
        default="Sin alerta"
    )

    # ==========================================
    # 4. DETECTAR INFINITOS
    # ==========================================
    # Los valores infinitos pueden surgir de divisiones por cero o de operaciones matemáticas extremas.
    # aqui buscamos el numero de valores infinitos por variable numerica, para luego poder analizarlos y decidir que hacer con ellos
    infinitos = pd.DataFrame(
        np.isinf(df[numericas]),
        columns=numericas,
        index=df.index
    )

    resumen_infinitos = infinitos.sum().rename(
        "valores_infinitos"
    )

    # ==========================================
    # 5. RESULTADOS
    # ==========================================
    # Tu función devuelve cuatro objetos:
    #resumen_anomalias	Resumen por variable: límites, cantidad y porcentaje de outliers.
    #df_anomalias;	Clientes que presentan alguna alerta.
    #df_scoring;	DataFrame original más los indicadores de anomalías.
    # resumen_infinitos;	Cantidad de infinitos por variable.

    df_scoring = df.join(flags)

    df_anomalias = df_scoring[
        (df_scoring["NIVEL_ANOMALIA"] != "Sin alerta")
    ].copy()

    return (
        resumen_anomalias.sort_values(
            "porcentaje_IQR",
            ascending=False
        ),
        df_anomalias,
        df_scoring,
        resumen_infinitos
    )
