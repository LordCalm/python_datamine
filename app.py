import yaml
import json
import os

# Отключаем GPU, чтобы избежать ошибки BLAS/StreamExecutor
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import category_encoders as ce
import pickle

# === Настройки страницы ===
st.set_page_config(page_title="Оценка стоимости жилья", layout="wide")
st.title("Веб-интерфейс: Прогноз стоимости (Москва)")

# === Папка экспериментов ===
EXP_FOLDER = "saved_exps"


# ------------------------------------------------------------------------------
# 1) Поиск всех экспериментов
# ------------------------------------------------------------------------------
def list_experiments():
    exps = []
    if os.path.exists(EXP_FOLDER):
        for name in os.listdir(EXP_FOLDER):
            full = os.path.join(EXP_FOLDER, name)
            if os.path.isdir(full):
                exps.append(name)
    return sorted(exps)


exp_list = list_experiments()

if not exp_list:
    st.error("В папке saved_exps нет экспериментов.")
    st.stop()

# Выбор эксперимента
selected_exp = st.sidebar.selectbox(
    "Выберите эксперимент",
    exp_list,
    index=0
)

EXP_PATH = os.path.join(EXP_FOLDER, selected_exp)
st.sidebar.success(f"Вы выбран эксперимент: {selected_exp}")


# ------------------------------------------------------------------------------
# 2) Загрузка ресурсов эксперимента
# ------------------------------------------------------------------------------
@st.cache_resource
def load_resources(exp_path):
    # 1. params.yaml
    params_file = os.path.join(exp_path, "params.yaml")
    with open(params_file, "r") as f:
        info = yaml.safe_load(f)

    # 2. Scalers
    with open(os.path.join(EXP_FOLDER, "scalerNormX.pkl"), "rb") as f:
        sX = pickle.load(f)

    with open(os.path.join(EXP_FOLDER, "scalerNormY.pkl"), "rb") as f:
        sY = pickle.load(f)

    # 3. Encoder
    encoder = None
    enc_path = os.path.join(EXP_FOLDER, "encoder.pkl")
    if os.path.exists(enc_path):
        with open(enc_path, "rb") as f:
            encoder = pickle.load(f)

    # 4. Model
    model_path = os.path.join(exp_path, "model.h5")

    if model_path.endswith(".h5"):
        # tensorflow model
        m = tf.keras.models.load_model(model_path)
    else:
        # pickle model
        with open(model_path, "rb") as f:
            m = pickle.load(f)

    return info, sX, sY, encoder, m


# Загружаем выбранный эксперимент
try:
    model_info, scalerX, scalerY, encoder, model = load_resources(EXP_PATH)
    st.success(f"Эксперимент '{selected_exp}' успешно загружен.")
except Exception as e:
    st.error(f"Ошибка загрузки эксперимента '{selected_exp}': {e}")
    st.stop()


# ------------------------------------------------------------------------------
# 3) Интерфейс
# ------------------------------------------------------------------------------
# --- Загружаем эталонный список признаков и статистику ---
if hasattr(scalerX, "feature_names_in_"):
    model_features = list(scalerX.feature_names_in_)
else:
    model_features = [
        "Renovation_N_European-style renovation", "Floor", "Is_first_floor",
        "Number of floors", "Living_share", "Minutes to metro",
        "Renovation_N_Without renovation", "Region_Digit", "Kitchen_share",
        "Apartment_type_Digit", "Renovation_N_Cosmetic", "Renovation_N_Designer",
        "Number of rooms", "Kitchen area", "Metro station", "Living area", "Area"
    ]

# --- Загружаем данные для заполнения дефолтных значений (медианы) из JSON ---
stats_path = os.path.join(EXP_FOLDER, "df_columns.json")
stats_dict = {}
if os.path.exists(stats_path):
    with open(stats_path, "r", encoding="utf-8") as f:
        stats_dict = json.load(f)

# --- Определяем, что вводит пользователь ---
IMPORTANT_FEATURES = [
    "Area", 
    "Living area", 
    "Kitchen area", 
    "Renovation_N_Designer", 
    "Number of rooms",
    "Metro station"
]

metro_list = model_info.get("meta", {}).get("metro_stations_list", [])

st.sidebar.header("Параметры квартиры")
input_data = {}

st.sidebar.markdown("### Основные параметры")

# Генерируем поля ввода
for feature in IMPORTANT_FEATURES:
    if feature in model_features:
        if feature == "Metro station":
            if metro_list:
                input_data[feature] = st.sidebar.selectbox("Метро", metro_list)
            else:
                input_data[feature] = st.sidebar.text_input("Метро", "Other")
        else:
            # Получаем медиану из json или 0.0
            def_val = 0.0
            if feature in stats_dict and stats_dict[feature]:
                def_val = stats_dict[feature].get("median", 0.0)
            
            input_data[feature] = st.sidebar.number_input(feature, value=float(def_val))

# --- Заполняем пропущенные признаки медианами ---
for feature in model_features:
    if feature not in input_data:
        val = 0.0
        if feature in stats_dict and stats_dict[feature]:
            val = stats_dict[feature].get("median", 0.0)
        
        # Если статистики нет (например, Metro station в JSON имеет null), ставим заглушку
        else:
            # Для категориальных колонок, если они не введены пользователем, 
            # безопаснее передать пустую строку или наиболее частое значение,
            # но так как Metro station в Important, сюда мы попадем редко.
            # Если это числовая колонка без статов -> 0.0
            if feature == "Metro station":
                val = metro_list[0] if metro_list else "Other"
                
        input_data[feature] = val

# ------------------------------------------------------------------------------
# 4) Показ входных данных
# ------------------------------------------------------------------------------
st.subheader("1. Входные данные")
df_input = pd.DataFrame([input_data])

df_input = df_input[model_features]
# Показываем пользователю только то, что он ввел
st.write("Вы ввели:")
st.dataframe(df_input[ [f for f in IMPORTANT_FEATURES if f in df_input.columns] ])

with st.expander("Посмотреть полный вектор для модели (со скрытыми нулями)"):
    st.dataframe(df_input)

# ------------------------------------------------------------------------------
# 5) Предсказание
# ------------------------------------------------------------------------------
if st.button("Рассчитать стоимость"):
    # --- Encoding ---
    if encoder:
        try:
            X_encoded = encoder.transform(df_input)
        except Exception as e:
            st.error(f"Ошибка Target Encoding: {e}")
            st.stop()
    else:
        X_encoded = df_input.copy()

    # --- Scaling ---
    try:
        X_norm_val = scalerX.transform(X_encoded)
    except Exception as e:
        st.error(f"Ошибка нормализации (Scaling): {e}")
        st.write("Проверьте соответствие колонок:")
        st.write("Ожидается:", getattr(scalerX, "feature_names_in_", "Неизвестно"))
        st.write("Пришло:", list(X_encoded.columns))
        st.stop()
    df_norm = pd.DataFrame(X_norm_val, columns=model_features)

    with st.expander("Показать нормализованные данные (вход в модель)"):
        st.dataframe(df_norm)

    # --- Model ---
    st.subheader("2. Прогноз")

    try:
        pred_norm = model.predict(X_norm_val)
    except Exception as e:
        st.error(f"Ошибка предсказания модели: {e}")
        st.stop()

    # Денормализация
    pred = scalerY.inverse_transform(np.array(pred_norm).reshape(-1, 1))

    st.metric(
        "Прогнозируемая цена",
        f"{pred[0][0]:,.2f} ₽"
    )
