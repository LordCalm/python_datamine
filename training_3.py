import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import category_encoders as ce
from tensorflow.keras.callbacks import EarlyStopping
from dvclive.keras import DVCLiveCallback
from dvclive import Live
import argparse
import os
import ast
import json
import yaml


# ------------------------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------------------------
def GET_METRICS_SINGLE(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    print(f"R2 Score: {r2:.4f}")
    return {"R2": r2}


def PLOT34(y_true, y_pred):
    print("PLOT34 function called (stub).")


# ------------------------------------------------------------------------------
# Основные функции
# ------------------------------------------------------------------------------
def load_data(data_path, features_list, target_col, valid_size, seed):
    df = pd.read_csv(
        './Data/Moscow_Housing_Price_FILTERED.csv', # путь к файлу, (используй автодотолнение)
        sep=',', # разделитель данных в файле
        header=0, # номер строки с заголовками, нумерация с нуля
        # header='None', # если заголовки отсутствуют
        )
        # If no features_list provided, infer numeric features excluding the target

    if not features_list:
        args.features = [col for col in df.columns if col != target_col]
        features_list = args.features
    
    for col in features_list:
        if col not in df.columns:
            raise ValueError(f"Признак '{col}' не найден в CSV файле.")
    
    print("features_list")
    print(features_list)
    
    # target_col is expected to be a single column name (string)
    X = df[features_list]
    y = df[target_col]

    x_train, x_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
    )

    return x_train, x_test, y_train, y_test


def preprocess(x_train, x_test, y_train, y_test, features_list):
    cat_features_in_use = [col for col in ["Metro station"] if col in features_list]

    if cat_features_in_use:
        encoder = ce.TargetEncoder(cols=cat_features_in_use)
        x_train_enc = encoder.fit_transform(x_train, y_train)
        x_test_enc = encoder.transform(x_test)
    else:
        x_train_enc = x_train.copy()
        x_test_enc = x_test.copy()

    scalerNormX = MinMaxScaler()
    x_train_norm = pd.DataFrame(
        scalerNormX.fit_transform(x_train_enc),
        columns=x_train_enc.columns,
        index=x_train_enc.index,
    )
    x_test_norm = pd.DataFrame(
        scalerNormX.transform(x_test_enc),
        columns=x_test_enc.columns,
        index=x_test_enc.index,
    )

    y_train_df = y_train.to_frame()
    y_test_df = y_test.to_frame()

    scalerNormY = MinMaxScaler()
    y_train_norm = pd.DataFrame(
        scalerNormY.fit_transform(y_train_df),
        columns=y_train_df.columns,
        index=y_train_df.index,
    )
    y_test_norm = pd.DataFrame(
        scalerNormY.transform(y_test_df),
        columns=y_test_df.columns,
        index=y_test_df.index,
    )

    return x_train_norm, x_test_norm, y_train_norm, y_test_norm, scalerNormY


def build_model(input_shape, hidden_layers, learning_rate):
    print(f"Building model with hidden layers: {hidden_layers}")
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_shape,)))
    for neurons in hidden_layers:
        model.add(tf.keras.layers.Dense(units=neurons, activation="relu"))
    model.add(tf.keras.layers.Dense(units=1, activation=None))
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["mean_squared_error"]
    )
    return model


def main(args):
    with Live(dir="dvc_experiments", save_dvc_exp=True) as live:
        # Логирование параметров
        live.log_param("data_path", args.data_path)
        live.log_param("features", str(args.features))
        live.log_param("hidden_layers", str(args.hidden_layers))
        live.log_param("epochs", args.epochs)
        live.log_param("learning_rate", args.learning_rate)
        live.log_param("valid_size", args.valid_size)
        live.log_param("seed", args.seed)

        x_train, x_test, y_train, y_test = load_data(
            data_path=args.data_path,
            features_list=args.features,
            target_col="Price",
            valid_size=args.valid_size,
            seed=args.seed,
        )

        x_train_norm, x_test_norm, y_train_norm, y_test_norm, scalerNormY = preprocess(
            x_train, x_test, y_train, y_test, args.features
        )

        model = build_model(
            input_shape=x_train_norm.shape[1],
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate,
        )
        model.summary()

        early_stop = EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        dvc_callback = DVCLiveCallback(live=live)

        print("x_train_norm")
        print(x_train_norm[:2])

        history = model.fit(
            x_train_norm,
            y_train_norm,
            validation_data=(x_test_norm, y_test_norm),
            epochs=args.epochs,
            batch_size=len(x_train_norm),
            verbose=1,
            callbacks=[early_stop, dvc_callback],
        )

        y_norm_pred = model.predict(x_test_norm)

        y_test_orig = scalerNormY.inverse_transform(y_test_norm)
        y_pred_orig = scalerNormY.inverse_transform(y_norm_pred)

        metrics = GET_METRICS_SINGLE(y_test_orig, y_pred_orig)
        final_val_loss = history.history["val_loss"][-1]

        live.log_metric("final_val_loss", final_val_loss)
        live.log_metric("R2_score", metrics["R2"])

        # логируем модель как артефакт, не перезаписывая вручную
        model_path = os.path.join(live.dir, "model.h5")
        model.save(model_path)
        live.log_artifact(model_path)

        PLOT34(y_test_orig, y_pred_orig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with DVC experiments.")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--valid-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=8)

    # путь к params.yaml
    parser.add_argument(
        "--params-path",
        type=str,
        default="dvc_experiments/params.yaml",
        help="Путь к файлу params.yaml"
    )

    # опциональные оверрайды
    parser.add_argument("--features", nargs="+", help="Список признаков")
    parser.add_argument("--hidden-layers", type=str, help='JSON-строка, например "[64, 32]"')

    args = parser.parse_args()

    # читаем YAML
    with open(args.params_path, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})

    # если в CLI не передали features — берём из YAML
    if args.features is None:
        args.features = train_params.get("features", [])
    # если в CLI не передали hidden_layers — берём из YAML
    if args.hidden_layers is None:
        args.hidden_layers = train_params.get("hidden_layers", [])
    else:
        # если передали строкой — парсим JSON
        args.hidden_layers = json.loads(args.hidden_layers.replace("'", '"'))

    main(args)
