import sys
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import category_encoders as ce
from matplotlib import pyplot as plt
import os
import yaml
from dvclive import Live

# Настройки
with open("params.yaml") as f:
    params = yaml.safe_load(f)
rand_seed = params["train"]["rand_seed"]
valid_size = 0.3

# Загрузка данных
df = pd.read_csv("Data/Moscow_Housing_Price_FILTERED.csv")

features = [
    'Renovation_N_European-style renovation',
    'Floor',
    'Is_first_floor',
    'Number of floors',
    'Living_share',
    'Minutes to metro',
    'Renovation_N_Without renovation',
    'Region_Digit',
    'Kitchen_share',
    'Apartment_type_Digit',
    'Renovation_N_Cosmetic',
    'Renovation_N_Designer',
    'Number of rooms',
    'Kitchen area',
    'Metro station',
    'Living area',
    'Area'
]
target = ['Price']

# Разделение данных
x_train, x_test, y_train, y_test = train_test_split(
    df[features],
    df[target],
    test_size=valid_size,
    random_state=rand_seed,
    shuffle=True
)
y_train = y_train['Price']
y_test = y_test['Price']

# Кодирование категориальных признаков
encoder = ce.TargetEncoder(cols=['Metro station'])
x_train_enc = encoder.fit_transform(x_train, y_train)
x_test_enc = encoder.transform(x_test)

# Стандартизация
scalerX = StandardScaler()
scalerY = StandardScaler()

x_train_std = pd.DataFrame(scalerX.fit_transform(x_train_enc), columns=x_train_enc.columns)
x_test_std = pd.DataFrame(scalerX.transform(x_test_enc), columns=x_test_enc.columns)
y_train_std = scalerY.fit_transform(y_train.to_frame()).ravel()
y_test_std = scalerY.transform(y_test.to_frame()).ravel()

# Функция вычисления метрик
def GET_METRICS_SINGLE(y_test, y_pred):
    '''
    Вычисление и вывод метрик: MAE, RMSE, R2. Используются функции из библиотеки sklearn
    На основе сравнения проверочных и вычисленных.
    :param y_test: - проверочные значения целевой переменной
    :param y_pred: - вычисленные значения целевой переменной
    '''
    mae  = metrics.mean_absolute_error        (y_test, y_pred)
    mse  = metrics.mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = metrics.r2_score                   (y_test, y_pred)
    return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2)
#--------------------------------------------------------------------------

# Графики
def PLOT34(y_test, y_pred, model_name="model", live: Live = None):
    '''
    Функция построения графиков
    :param y_test: - проверочные значения целевой переменной
    :param y_pred: - вычисленные значения целевой переменной
    '''
    fig = plt.figure(figsize=(12,6))
    os.makedirs("plots", exist_ok=True)

    # == Диаграмма рассеяния вычисленных значений ==
    # создать первое полотно 121: 1-строк, 2-столбцов, 1-индекс текущего полотна в сетке
    plt.subplot(121)
    plt.scatter(y_test, y_pred,  alpha=0.1, color = "#17becf")
    plt.plot(
        [ np.min(y_test), np.max(y_test) ], # x1,x2
        [ np.min(y_test), np.max(y_test) ], # y1,y2
        '--',
        alpha=0.7, lw=3, color = "black")
    plt.title('Диаграмма рассеяния вычисленных значений')
    plt.xlabel('Проверочное Y')
    plt.ylabel('Вычисленное Y')
    plt.grid(True)  # Сетка. Доп параметры color='black', linewidth=0.7

    # == Диаграмма рассеяния ошибок ==
    # создать второе полотно 121: 1-строк, 2-столбцов, 2-индекс текущего полотна в сетке
    plt.subplot(122)
    plt.scatter(y_test, (y_test - y_pred)**2,  alpha=0.1, color = "#17becf")
    plt.title('Диаграмма рассеяния квадрата абсолютной ошибки')
    plt.xlabel('Проверочное Y')
    plt.ylabel('Квадрат абсолютной ошибки')
    plt.grid(True)  # Сетка. Доп параметры color='black', linewidth=0.7
    
    # === Сохраняем ===
    plot_path = f"plots/{model_name}_scatter.png"
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close(fig)
    
    # === Логирование в DVCLive ===
    if live is not None:
        live.log_image(f"{model_name}_scatter", plot_path)
        live.next_step()
        print(f"График для {model_name} сохранён и залогирован в DVCLive: {plot_path}")
    else:
        print(f"График для {model_name} сохранён локально: {plot_path}")
#--------------------------------------------------------------------------

# Эксперименты
results = []

# 1. SGDRegressor без стандартизации
sgd_nonstd = linear_model.SGDRegressor(loss='squared_error', max_iter=10000, tol=1e-3, random_state=rand_seed)
sgd_nonstd.fit(x_train_enc, y_train)
y_pred_nonstd = sgd_nonstd.predict(x_test_enc)
results.append({"model": "SGDRegressor_nonstd", **GET_METRICS_SINGLE(y_test, y_pred_nonstd)})

# 2. LinearRegression без стандартизации
lr_nonstd = linear_model.LinearRegression()
lr_nonstd.fit(x_train_enc, y_train)
y_pred_lr = lr_nonstd.predict(x_test_enc)
results.append({"model": "LinearRegression_nonstd", **GET_METRICS_SINGLE(y_test, y_pred_lr)})

# 3. SGDRegressor со стандартизацией
sgd_std = linear_model.SGDRegressor(loss='squared_error', max_iter=10000, tol=1e-3, random_state=rand_seed)
sgd_std.fit(x_train_std, y_train_std)
y_pred_std = sgd_std.predict(x_test_std)
results.append({"model": "SGDRegressor_std", **GET_METRICS_SINGLE(y_test_std, y_pred_std)})

# Сохранение результатов
df_results = pd.DataFrame(results)
df_results.to_csv("results.csv", index=False)

# Логирование через DVC Live
with Live("dvclive", resume=True) as live:
    for r in results:
        model_name = r["model"]
        # логирование метрик
        for k, v in r.items():
            if k != "model":
                live.log_metric(f"{r['model']}/{k}", v)
        # логирование графиков
        if model_name == "SGDRegressor_nonstd":
            PLOT34(y_test, y_pred_nonstd, model_name=model_name, live=live)
        elif model_name == "LinearRegression_nonstd":
            PLOT34(y_test, y_pred_lr, model_name=model_name, live=live)
        elif model_name == "SGDRegressor_std":
            PLOT34(y_test_std, y_pred_std, model_name=model_name, live=live)

print(df_results)
