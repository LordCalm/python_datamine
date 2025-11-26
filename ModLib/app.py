import yaml
import json
import os
import ast

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
st.sidebar.success(f"Выбран эксперимент: {selected_exp}")


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
    
    # 5. Metrics (info.json)
    metrics_data = {}
    json_path = os.path.join(exp_path, "info.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding='utf-8') as f:
            metrics_data = json.load(f)

    return info, sX, sY, encoder, m, metrics_data


# Загружаем выбранный эксперимент
try:
    model_info, scalerX, scalerY, encoder, model, metrics_json = load_resources(EXP_PATH)
    st.success(f"Эксперимент '{selected_exp}' успешно загружен.")
except Exception as e:
    st.error(f"Ошибка загрузки эксперимента '{selected_exp}': {e}")
    st.stop()


# ------------------------------------------------------------------------------
# 3) Интерфейс
# ------------------------------------------------------------------------------
# --- Загружаем эталонный список признаков и статистику ---
# Признаки, которые ждет СКАЛЕР (17)
if hasattr(scalerX, "feature_names_in_"):
    scaler_features = list(scalerX.feature_names_in_)
else:
    scaler_features = [
        "Renovation_N_European-style renovation", "Floor", "Is_first_floor",
        "Number of floors", "Living_share", "Minutes to metro",
        "Renovation_N_Without renovation", "Region_Digit", "Kitchen_share",
        "Apartment_type_Digit", "Renovation_N_Cosmetic", "Renovation_N_Designer",
        "Number of rooms", "Kitchen area", "Metro station", "Living area", "Area"
    ]


# Признаки, которые ждет МОДЕЛЬ (может быть только 2, а может все)
# Читаем из params.yaml текущего эксперимента
raw_features = model_info.get("features", [])
# Если features записаны как строка "['Area', ...]", парсим её
if isinstance(raw_features, str):
    try:
        model_needed_features = ast.literal_eval(raw_features)
    except:
        model_needed_features = []
else:
    model_needed_features = raw_features

# Если список пуст, значит модель училась на всех
if not model_needed_features:
    model_needed_features = scaler_features

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

st.sidebar.header("Параметры квартиры")
input_data = {}

st.sidebar.markdown("### Основные параметры")

# Генерируем поля ввода
for feature in IMPORTANT_FEATURES:
    # Показываем инпут, только если этот признак вообще существует в глобальном списке
    if feature in model_needed_features:
        if feature == "Metro station":
            input_data[feature] = st.sidebar.text_input("Metro station", "Other")
        else:
            def_val = 0.0
            if feature in stats_dict and stats_dict[feature]:
                def_val = stats_dict[feature].get("median", 0.0)
            input_data[feature] = st.sidebar.number_input(feature, value=float(def_val))

# --- Заполняем пропущенные признаки медианами ---
for feature in scaler_features:
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
                val = "Other"
                
        input_data[feature] = val

# ------------------------------------------------------------------------------
# 4) Показ входных данных
# ------------------------------------------------------------------------------
with st.expander("Входные данные"):
    df_input = pd.DataFrame([input_data])
    # Сортируем как ждет скалер
    df_input = df_input[scaler_features]
    
    # Показываем только то, что вводит юзер
    st.write("Вы ввели:")
    st.dataframe(df_input[[f for f in IMPORTANT_FEATURES if f in df_input.columns]])
    
    st.write("Все параметры:")
    st.dataframe(df_input)

with st.expander("Метрики модели"):
    # Парсинг info.json для вывода R2 и RMSE
    try:
        # Обычно DVC возвращает список, берем последний актуальный эксперимент или workspace
        metrics_found = False
        
        # Функция для безопасного поиска в словаре
        def get_metrics_from_node(node):
            try:
                # Путь к метрикам внутри json
                return node["data"]["metrics"]["dvc_experiments\\metrics.json"]["data"]
            except (KeyError, TypeError):
                return None

        vals = None
        if isinstance(metrics_json, list):
            for item in metrics_json:
                vals = get_metrics_from_node(item)
                if vals: break # Нашли первые попавшиеся метрики
        elif isinstance(metrics_json, dict):
            vals = get_metrics_from_node(metrics_json)

        if vals:
            r2 = vals.get("R2_score", 0)
            rmse = vals.get("final_val_loss", 0)
            hidden_layers = model_info.get("hidden_layers", '[]')
            
            st.metric("R2 Score", f"{r2:.4f}")
            st.metric("Final val loss", f"{rmse:.4f}")
            st.metric("Hidden layers", str(hidden_layers))
        else:
            st.info("Метрики не найдены в info.json")
            
    except Exception as e:
        st.warning(f"Не удалось распарсить info.json: {e}")

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

    # --- Scaling (Тут получается 17 колонок) ---
    try:
        X_norm_val = scalerX.transform(X_encoded)
        # Превращаем обратно в DF чтобы удобно выбрать столбцы по именам
        df_norm_full = pd.DataFrame(X_norm_val, columns=scaler_features)
    except Exception as e:
        st.error(f"Ошибка нормализации (Scaling): {e}")
        st.write("Проверьте соответствие колонок:")
        st.write("Ожидается:", getattr(scalerX, "feature_names_in_", "Неизвестно"))
        st.write("Пришло:", list(X_encoded.columns))
        st.stop()
    
    # --- Filtering ---
    try:
        df_norm_model = df_norm_full[model_needed_features]
    except KeyError as e:
        st.error(f"Ошибка: модель требует признаки {model_needed_features}, но в данных их нет. Проверьте params.yaml.")
        st.stop()
    
    # Показываем что идет в модель
    with st.expander(f"Данные для модели (Признаков: {len(model_needed_features)})"):
        st.dataframe(df_norm_model)

    # --- Model ---
    st.header("Прогноз")

    try:
        pred_norm = model.predict(df_norm_model)
    except Exception as e:
        st.error(f"Ошибка предсказания модели: {e}")
        st.stop()

    # Денормализация
    pred = scalerY.inverse_transform(np.array(pred_norm).reshape(-1, 1))

    st.metric(
        "Прогнозируемая цена",
        f"{pred[0][0]:,.2f} ₽"
    )
