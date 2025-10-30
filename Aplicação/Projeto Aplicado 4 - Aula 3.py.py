from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools

# ===== 1) Leitura e pré-processamento =====
df = pd.read_csv(
    "E:\\Mack C Dados\\5 Semestre\\Projeto Aplicado IV\\carteira_investimento_mcid.csv",
    encoding="latin1",   # ou "cp1252"
    sep=";",             # muitos CSVs no BR usam ;
    decimal=",",         # números com vírgula decimal
    thousands=".",       # separador de milhar
    engine="python"      # parser mais tolerante
)

for col in ["dte_assinatura_contrato","dte_fim_contrato","dte_inicio_obra","dte_fim_obra","dte_controle"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")
df["dte_paralisacao"] = pd.to_datetime(df.get("dte_paralisacao"), errors="coerce")

cat_cols = ["txt_origem","txt_uf","txt_regiao","cod_ibge_7dig","txt_tipo_instrumento",
            "dsc_situacao_contrato_mcid","dsc_situacao_objeto_mcid"]
df[cat_cols] = df[cat_cols].fillna("Desconhecido")

num_cols_zero = ["vlr_investimento","vlr_repasse","vlr_contrapartida","vlr_empenhado",
                 "vlr_desembolsado","vlr_pago","prc_execucao_fisica","qtd_uh",
                 "qtd_entregues","qtd_uh_distratadas","qtd_vigentes"]
df[num_cols_zero] = df[num_cols_zero].fillna(0)

# ===== 2) Série mensal de obras ativas =====
horiz_ts = pd.to_datetime(df["dte_controle"].max())
if pd.isna(horiz_ts):
    horiz_ts = pd.to_datetime(pd.Series([
        df["dte_inicio_obra"].max(),
        df["dte_fim_obra"].max(),
        df["dte_assinatura_contrato"].max(),
        df["dte_fim_contrato"].max()
    ]).max())

start = df["dte_inicio_obra"].fillna(df["dte_assinatura_contrato"])
end   = df["dte_fim_obra"].combine_first(df["dte_fim_contrato"])
end_cap = end.fillna(horiz_ts).clip(upper=horiz_ts)

mask_start = start.notna()
mask_pair  = mask_start & end_cap.notna() & (end_cap >= start)

start_m = start[mask_start].dt.to_period("M")
end_m   = (end_cap[mask_pair].dt.to_period("M") + 1)

delta = start_m.value_counts().sort_index()
delta = delta.add(end_m.value_counts().sort_index().mul(-1), fill_value=0)

min_pm = start[mask_start].min().to_period("M")
idx = pd.period_range(start=min_pm, end=horiz_ts.to_period("M"), freq="M")

ativos = delta.reindex(idx, fill_value=0).cumsum().clip(lower=0).astype(float)
ativos.index = ativos.index.to_timestamp()
ativos = ativos.asfreq("MS")

# ===== 3) Decomposição (opcional) =====
q_hi = np.nanpercentile(ativos.values, 99.5)
ativos_w = np.minimum(ativos, q_hi)
res = seasonal_decompose(ativos_w, model="multiplicative", period=12, extrapolate_trend="freq")

# ===== 4) Modelagem =====
# Troca de fillna(method="ffill") por .ffill() (pandas >= 2.0)
y = ativos.copy().astype(float).asfreq("MS").ffill()

n = len(y)
n_train = int(n * 0.70)
n_val   = int(n * 0.15)
y_train = y.iloc[:n_train]
y_val   = y.iloc[n_train:n_train+n_val]
y_test  = y.iloc[n_train+n_val:]

def mape(y_true, y_pred, eps=1e-8):
    denom = np.where(np.abs(y_true) < eps, eps, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100.0

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_ = mean_absolute_error(y_true, y_pred)
    mp_  = mape(np.array(y_true), np.array(y_pred))
    return rmse, mae_, mp_

# Baselines
yhat_naive_val  = pd.Series(y_train.iloc[-1], index=y_val.index)

# Substitui .append por pd.concat
_last_train_val = pd.concat([y_train, y_val])
yhat_naive_test = pd.Series(_last_train_val.iloc[-1], index=y_test.index)

def seasonal_naive(series, horizon, season=12):
    last_season = series.iloc[-season:]
    reps = int(np.ceil(horizon / season))
    pred = pd.Series(
        np.tile(last_season.values, reps)[:horizon],
        index=pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    )
    return pred

yhat_snaive_val  = seasonal_naive(y_train, len(y_val), season=12).reindex(y_val.index)
yhat_snaive_test = seasonal_naive(pd.concat([y_train, y_val]), len(y_test), season=12).reindex(y_test.index)

# Holt-Winters
hw_configs = [
    dict(trend="add", seasonal="add", seasonal_periods=12),
    dict(trend="add", seasonal="mul", seasonal_periods=12),
    dict(trend="mul", seasonal="mul", seasonal_periods=12),
]
best_rmse = np.inf; best_cfg = None
for cfg in hw_configs:
    fit = ExponentialSmoothing(y_train, **cfg, initialization_method="estimated").fit(optimized=True)
    pred_val = fit.forecast(len(y_val))
    rmse, _, _ = eval_metrics(y_val, pred_val)
    if rmse < best_rmse:
        best_rmse = rmse; best_cfg = cfg

fit_hw_full  = ExponentialSmoothing(pd.concat([y_train, y_val]), **best_cfg, initialization_method="estimated").fit(optimized=True)
yhat_hw_test = fit_hw_full.forecast(len(y_test))

# SARIMA (grid compacto)
p = d = q = [0,1,2]
P = D = Q = [0,1]
s = 12
best_sar_rmse = np.inf; best_orders = None
for (pi,di,qi) in itertools.product(p,d,q):
    for (Pi,Di,Qi) in itertools.product(P,D,Q):
        try:
            sar = SARIMAX(
                y_train,
                order=(pi,di,qi),
                seasonal_order=(Pi,Di,Qi,s),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
            pred_val = sar.get_forecast(steps=len(y_val)).predicted_mean
            rmse, _, _ = eval_metrics(y_val, pred_val)
            if rmse < best_sar_rmse:
                best_sar_rmse = rmse
                best_orders = ((pi,di,qi),(Pi,Di,Qi,s))
        except Exception:
            pass

sar_full = SARIMAX(
    pd.concat([y_train, y_val]),
    order=best_orders[0],
    seasonal_order=best_orders[1],
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)
yhat_sar_test = sar_full.get_forecast(steps=len(y_test)).predicted_mean

# ===== 5) Avaliação =====
def add_result(name, y_true, y_pred, acc):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_ = mean_absolute_error(y_true, y_pred)
    mp_  = mape(np.array(y_true), np.array(y_pred))
    acc.append(dict(Modelo=name, RMSE=rmse, MAE=mae_, MAPE=mp_))

acc = []
add_result("Naive", y_test, yhat_naive_test, acc)
add_result("Sazonal (lag12)", y_test, yhat_snaive_test, acc)
add_result(f"Holt-Winters {best_cfg}", y_test, yhat_hw_test, acc)
add_result(f"SARIMA{best_orders[0]}×{best_orders[1]}", y_test, yhat_sar_test, acc)

df_scores = pd.DataFrame(acc).sort_values("RMSE")
print("\nDesempenho em TESTE:")
print(df_scores.to_string(index=False))

# ===== 6) Gráfico Observado vs Previsto (teste) =====
plt.figure(figsize=(12,5))
y.plot(label="Observado")
yhat_naive_test.plot(label="Naive (teste)")
yhat_snaive_test.plot(label="Sazonal 12 (teste)")
yhat_hw_test.plot(label="Holt-Winters (teste)")
yhat_sar_test.plot(label="SARIMA (teste)")
plt.title("Modelagem – observado vs previsto (conjunto de teste)")
plt.xlabel("Mês"); plt.ylabel("Obras ativas")
plt.legend(); plt.tight_layout(); plt.show()
