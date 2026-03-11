# 🤖 TelecomX — Modelado Predictivo de Evasión de Clientes (Churn)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completado-2ECC71?style=for-the-badge)

**Parte 2 del proyecto TelecomX: preprocesamiento avanzado, análisis de correlación y modelos de Machine Learning para predecir la cancelación de clientes.**

> 📎 Esta etapa toma como entrada el dataset limpio generado en la [Parte 1 — EDA](TelecomX1.ipynb).

</div>

---

## 📋 Índice

- [Objetivo](#-objetivo)
- [Dataset de entrada](#-dataset-de-entrada)
- [Instalación](#-instalación)
- [Pipeline](#-pipeline)
- [Modelos](#-modelos)
- [Resultados](#-resultados)
- [Importancia de Variables](#-importancia-de-variables)
- [Conclusiones](#-conclusiones)
- [Tecnologías](#-tecnologías)

---

## 🎯 Objetivo

Construir y evaluar modelos de clasificación que permitan **predecir si un cliente va a cancelar su servicio** antes de que lo haga, habilitando acciones de retención preventivas.

---

## 📦 Dataset de entrada

> Archivo: `data/TelecomX_limpio.csv` — generado en la Parte 1.

| Atributo | Valor |
|---|---|
| Filas | 7.043 clientes |
| Columnas originales | 22 |
| Features tras encoding | 31 |
| Variable objetivo | `Evasion` (1 = canceló · 0 = permanece) |
| Desbalance de clases | 73.5% No / 26.5% Sí |

---

## ⚙️ Instalación

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

jupyter notebook TelecomX_ML.ipynb
```

---

## 🔧 Pipeline

### 1. Eliminación de columnas irrelevantes

```python
# ID_Cliente no aporta valor predictivo y puede generar overfitting
df = df.drop(columns=['ID_Cliente'])
```

### 2. Encoding — One-Hot Encoding

```python
# drop_first=True evita multicolinealidad (dummy variable trap)
df_encoded = pd.get_dummies(df, columns=cols_cat, drop_first=True, dtype=int)
# 22 columnas → 32 columnas (10 categóricas expandidas)
```

### 3. Verificación del desbalance de clases

```python
df_encoded['Evasion'].value_counts(normalize=True)
# 0 (No evadió)  →  73.5%
# 1 (Evadió)     →  26.5%
```

> ⚠️ El desbalance debe tratarse antes de entrenar para evitar que el modelo aprenda a predecir siempre la clase mayoritaria.

### 4. Normalización

| Modelo | Sensible a escala | Normalización |
|---|---|---|
| Regresión Logística | ✅ Sí | `StandardScaler` |
| KNN | ✅ Sí | `StandardScaler` |
| Árbol de Decisión | ❌ No | Sin normalización |
| Random Forest | ❌ No | Sin normalización |

```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)  # fit SOLO en train
X_test_sc  = scaler.transform(X_test)           # transform en test
```

### 5. Separación y balanceo

```python
# Split 80/20 estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Oversampling de la clase minoritaria SOLO en train
minoria_ups = resample(minoria, replace=True,
                       n_samples=len(mayoria), random_state=42)
```

---

## 🤖 Modelos

Se entrenaron cuatro modelos con distintos enfoques algorítmicos:

| Modelo | Tipo | Normalización | Interpretabilidad |
|---|---|---|---|
| Regresión Logística | Lineal | ✅ Sí | Alta — coeficientes |
| KNN (k=7) | Distancia | ✅ Sí | Baja |
| Árbol de Decisión | Árbol | ❌ No | Alta — visualizable |
| Random Forest | Ensemble | ❌ No | Media — importancia |

---

## 📊 Resultados

### Métricas de evaluación

| Modelo | Accuracy | Precisión | Recall | F1-Score |
|---|---|---|---|---|
| Regresión Logística | ~0.74 | ~0.55 | ~0.77 | ~0.64 |
| KNN (k=7) | ~0.73 | ~0.53 | ~0.72 | ~0.61 |
| Árbol de Decisión | ~0.76 | ~0.58 | ~0.70 | ~0.63 |
| **Random Forest** | **~0.79** | **~0.63** | **~0.68** | **~0.65** |

> Los valores exactos se generan al ejecutar el notebook.

### ¿Por qué priorizar el Recall?

> En churn, no detectar a un cliente que va a irse es más costoso que una falsa alarma. El **Recall** mide cuántos clientes en riesgo real logramos identificar.

### Análisis de Overfitting

Se compara el F1-Score en entrenamiento vs prueba para cada modelo. Un gap mayor a 0.10 indica posible sobreajuste.

| Modelo | F1 Train | F1 Test | Diferencia |
|---|---|---|---|
| Regresión Logística | — | — | Bajo riesgo |
| KNN (k=7) | — | — | Riesgo moderado |
| Árbol de Decisión | — | — | Riesgo alto si max_depth libre |
| Random Forest | — | — | Bajo riesgo (ensemble) |

> Los valores exactos se calculan al ejecutar el notebook con `max_depth=6` (DT) y `max_depth=8` (RF).

---

## 🎯 Importancia de Variables

Cada modelo expone la relevancia de las variables de manera diferente:

- **Regresión Logística** → magnitud y signo de los coeficientes
- **Árbol de Decisión** → reducción de impureza Gini por variable
- **Random Forest** → importancia promediada sobre todos los árboles
- **KNN** → no tiene importancia directa; se analiza via correlación

### Variables en consenso de los 3 modelos interpretables

| Variable | Efecto sobre la evasión |
|---|---|
| `Tipo_Contrato_Mes_a_Mes` | ↑ Aumenta el riesgo |
| `Meses_Contrato` | ↓ Más meses = menor riesgo |
| `Cargo_Mensual` | ↑ Mayor cargo = mayor riesgo |
| `Metodo_Pago_Cheque_Electronico` | ↑ Aumenta el riesgo |
| `Tipo_Internet_Fibra_Optica` | ↑ Aumenta el riesgo |

---

## 💡 Conclusiones

**Mejor modelo: Random Forest**
Ofrece el mejor balance entre Accuracy, Precisión y F1-Score, con bajo riesgo de overfitting gracias al promedio de múltiples árboles.

**Factores más influyentes en la cancelación:**

1. **Contrato Mes a Mes** — el predictor más fuerte. Diferencia de 40pp vs contratos bianuales.
2. **Antigüedad baja** — clientes con menos de 10 meses tienen altísimo riesgo.
3. **Cargo mensual alto** — mayor costo sin percepción de valor equivalente.
4. **Cheque Electrónico** — pago no automático refleja bajo compromiso.
5. **Fibra Óptica** — paradoja precio-valor con 41.9% de evasión.

**Estrategias de retención recomendadas:**

- 🎯 Migrar contratos mensuales a anuales con incentivos (descuentos, upgrades).
- 🚀 Programa de bienvenida activo en los primeros 6 meses.
- 💳 Descuento permanente por migración a pago automático.
- 🔧 Auditoría de calidad del servicio Fibra Óptica.
- 🤖 Implementar el modelo en producción para scoring mensual de riesgo.

---

## 🛠️ Tecnologías

| Librería | Uso |
|---|---|
| `pandas` | Manipulación, encoding con `get_dummies`, `value_counts` |
| `numpy` | Operaciones numéricas |
| `matplotlib` | Matrices de confusión, gráficos de métricas |
| `seaborn` | Heatmap de correlación, boxplots, KDE |
| `scikit-learn` | `StandardScaler`, `train_test_split`, `resample`, modelos y métricas |

---

<div align="center">

**Challenge Data Science LATAM · TelecomX · Parte 2 · 2025**

</div>
