from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

df = pd.read_csv("E:https://dados.gov.br/dados/conjuntos-dados/base-de-dados-da-carteira-de-investimento-do-ministerio-das-cidades.csv",
    encoding="latin1",   # ou "cp1252"
    sep=";",            # muitos CSVs no BR usam ;
    decimal=",",        # números com vírgula decimal
    thousands=".",      # separador de milhar
    engine="python"     # parser mais tolerante
    )


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

print(df.shape)
print(df.head(10))
print(df.dtypes)

# VERIFICAR SE TEM VALORES AUSENTES
print('\nSoma dos valores nulos em cada coluna\n')
print(df.isna().sum())
print('\nLinhas antes de dropna: ', len(df))


# REDUZIR DIMENSIONALIDADE DA BASE 
colunas_excluir = [
    # Legados / sistemas antigos
    "cod_saci",
    "cod_mdr_antigo",
    "bln_carga_legado_tci",
    "cod_cipi_projeto_invest",
    "cod_cipi_intervencao",
    "id_governa",
    "num_generico_contrato",
    # Identificadores técnicos (se não for analisar por agente)
    "cod_ag_operador",
    "cod_ag_financeiro",
    "cnpj_agente_financeiro",
    "txt_agente_financeiro",
    # Secretaria (só útil se análise institucional for necessária)
    "txt_sigla_secretaria",
    "txt_nome_secretaria",
    # PAC e Emendas (muitos nulos, só manter se foco for político-institucional)
    "bln_pac",
    "dsc_fase_pac",
    "bln_emenda",
    "num_emendas",
    "qtd_emendas",
    # Flags administrativas internas
    "bln_carteira_mcid",
    "bln_carteira_ativa_mcid",
    "bln_carteira_andamento",
    # Fonte de dados (baixa completude)
    "dsc_fonte",
    "txt_fonte",
    "dsc_sub_fonte",
    "cod_acao_ultimo_empenho",
    # Financeiro de baixa relevância
    "vlr_desbloqueado",
    "vlr_taxa_adm",
    # Datas pouco preenchidas
    "dte_atualizacao_situacao_atual",
    # IDs redundantes
    "num_convenio",
    "cod_proposta",
    "num_proposta",
    "cod_operacao",
    "cod_dv",
    "cod_contrato",
    # Textos longos (reduzem desempenho e não servem para correlação direta)
    "dsc_objeto_instrumento",
    "dsc_detalhamento_motivo_paralisacao",
    "dsc_motivo_paralisacao",
    # Códigos geográficos redundantes
    "cod_ibge_6dig",
    # Tomador (se não for analisar por agente/tomador específico)
    "cod_tomador",
    "txt_tomador",
    # Ano redundante (já inferível da data de assinatura)
    "num_ano_instrumento",
    # Extras sugeridos (pouco úteis para análise temporal/correlação)
    "dte_carga",                       # apenas log de carga do sistema
    "cod_tci",                         # chave, não precisa em análise
    "dsc_concedente",                  # sempre igual (Ministério das Cidades)
    "txt_municipio",                   # alta cardinalidade
    "txt_motivo_paralisacao_mcid",     # texto aberto
    "txt_principal_motivo_paralisacao",
    "dsc_situacao_atual",              # muito aberto, difícil correlação
]
# Aplicando exclusão
df_reduzido = df.drop(columns=colunas_excluir)
df = df_reduzido.copy()
del df_reduzido
del colunas_excluir


# VERFICAR SHAPE, HEAD E DTYPES NOVAMENTE 
print(df.shape)
print(df.head(10))
print(df.dtypes)


# VERIFICAR NOVAMENTE SE TEM VALORES AUSENTES
print('\nSoma dos valores nulos em cada coluna\n')
print(df.isna().sum())
print('\nLinhas antes de dropna: ', len(df))


# CONVERTER DATAS PARA FORMATO CORRETO
datas = [
    "dte_assinatura_contrato",
    "dte_fim_contrato",
    "dte_inicio_obra",
    "dte_fim_obra",
    "dte_controle",
]
for col in datas:
    df[col] = pd.to_datetime(df[col], errors="coerce")
# corrigir dte_paralisacao (veio float)
df["dte_paralisacao"] = pd.to_datetime(df["dte_paralisacao"], errors="coerce")
del col
del datas


# TRATAR VALORES NULOS PARA NÃO ESTRAGAR GRÁFICOS ETC
# Categóricas -> 'Desconhecido'
cat_cols = ["txt_origem","txt_uf","txt_regiao","cod_ibge_7dig",
            "txt_tipo_instrumento","dsc_situacao_contrato_mcid",
            "dsc_situacao_objeto_mcid"]
df[cat_cols] = df[cat_cols].fillna("Desconhecido")
# Numéricas -> 0
num_cols_zero = ["vlr_investimento","vlr_repasse","vlr_contrapartida",
                 "vlr_empenhado","vlr_desembolsado","vlr_pago",
                 "prc_execucao_fisica","qtd_uh","qtd_entregues",
                 "qtd_uh_distratadas","qtd_vigentes"]
df[num_cols_zero] = df[num_cols_zero].fillna(0)
del cat_cols
del num_cols_zero



# VERIFICAR NOVAMENTE SE TEM VALORES AUSENTES
print('\nSoma dos valores nulos em cada coluna\n')
print(df.isna().sum())
print('\nLinhas antes de dropna: ', len(df))


##### GRÁFICOS COM MESES VS OUTROS DADOS
mes_lbl = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}
# =========================
# 1) Sazonalidade: INÍCIOS
# =========================
df_ini = df.dropna(subset=["dte_inicio_obra"]).copy()
df_ini["mes"] = df_ini["dte_inicio_obra"].dt.month
starts = df_ini.groupby("mes").size().reindex(range(1,13), fill_value=0)
anos_ini = df_ini["dte_inicio_obra"].dt.year.nunique()
starts = starts / max(anos_ini,1)
starts.index = starts.index.map(mes_lbl)
plt.figure(figsize=(12,6))
starts.plot(kind="bar")
plt.title("Sazonalidade: obras iniciadas por mês do ano (média por mês/ano)")
plt.xlabel("Mês"); plt.ylabel("Quantidade média")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
# ===========================
# 2) Sazonalidade: CONCLUSÕES
# ===========================
df_fim = df.dropna(subset=["dte_fim_obra"]).copy()
df_fim["mes"] = df_fim["dte_fim_obra"].dt.month
finishes = df_fim.groupby("mes").size().reindex(range(1,13), fill_value=0)
anos_fim = df_fim["dte_fim_obra"].dt.year.nunique()
finishes = finishes / max(anos_fim,1)
finishes.index = finishes.index.map(mes_lbl)
plt.figure(figsize=(12,6))
finishes.plot(kind="bar")
plt.title("Sazonalidade: obras concluídas por mês do ano (média por mês/ano)")
plt.xlabel("Mês"); plt.ylabel("Quantidade média")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()
# ============================================
# 3) Obras ATIVAS por mês (cap até último mês observado)
# ============================================
# horizonte real (prefere dte_controle; fallback: maiores datas conhecidas)
horiz_ts = pd.to_datetime(df["dte_controle"].max())
if pd.isna(horiz_ts):
    horiz_ts = pd.to_datetime(pd.Series([
        df["dte_inicio_obra"].max(),
        df["dte_fim_obra"].max(),
        df["dte_assinatura_contrato"].max(),
        df["dte_fim_contrato"].max()
    ]).max())
horiz_pm = horiz_ts.to_period("M")
# âncoras
start = df["dte_inicio_obra"].fillna(df["dte_assinatura_contrato"])
end   = df["dte_fim_obra"].combine_first(df["dte_fim_contrato"])
# cap de fim: ausente -> horizonte; futuro -> horizonte
end_cap = end.fillna(horiz_ts).clip(upper=horiz_ts)
# pares válidos
mask_start = start.notna()
mask_pair  = mask_start & end_cap.notna() & (end_cap >= start)
start_m = start[mask_start].dt.to_period("M")
end_m   = (end_cap[mask_pair].dt.to_period("M") + 1)  # encerra no mês seguinte
# delta-cumsum
delta = start_m.value_counts().sort_index()
delta = delta.add(end_m.value_counts().sort_index().mul(-1), fill_value=0)
# índice mensal do 1º início ao horizonte
min_pm = start[mask_start].min().to_period("M")
idx = pd.period_range(start=min_pm, end=horiz_pm, freq="M")
ativos = delta.reindex(idx, fill_value=0).cumsum().clip(lower=0).astype(int)
ativos.index = ativos.index.to_timestamp()
plt.figure(figsize=(12,6))
ativos.plot()
plt.title("Obras ativas por mês (início=assinatura se início ausente; fim=fim/contrato; cap no horizonte)")
plt.xlabel("Mês"); plt.ylabel("Quantidade de obras ativas")
plt.tight_layout(); plt.show()

### OUTROS GRAFICOS DECOMPOSIÇÃO
# --------- Série mensal: obras ativas (cap no horizonte real) ---------
horiz_ts = pd.to_datetime(df["dte_controle"].max())
if pd.isna(horiz_ts):
    horiz_ts = pd.to_datetime(pd.Series([
        df["dte_inicio_obra"].max(),
        df["dte_fim_obra"].max(),
        df["dte_assinatura_contrato"].max(),
        df["dte_fim_contrato"].max()
    ]).max())
horiz_pm = horiz_ts.to_period("M")
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
idx = pd.period_range(start=min_pm, end=horiz_pm, freq="M")
ativos = delta.reindex(idx, fill_value=0).cumsum().clip(lower=0).astype(float)
ativos.index = ativos.index.to_timestamp()  # DatetimeIndex mensal
ativos = ativos.asfreq("MS")                # frequência mensal explícita (início do mês)
# opcional: suavizar outliers muito grandes (winsorização leve)
q_hi = np.nanpercentile(ativos.values, 99.5)
ativos_w = np.minimum(ativos, q_hi)
# --------- Decomposição clássica multiplicativa (period=12) ---------
res = seasonal_decompose(ativos_w, model="multiplicative", period=12, extrapolate_trend="freq")
# Observado
plt.figure(figsize=(12,5))
res.observed.plot()
plt.title("Obras ativas por mês — observado")
plt.xlabel("Mês"); plt.ylabel("Obras ativas")
plt.tight_layout(); plt.show()
# Tendência (médias móveis centrais)
plt.figure(figsize=(12,5))
res.trend.plot()
plt.title("Tendência (médias móveis) — obras ativas")
plt.xlabel("Mês"); plt.ylabel("Tendência")
plt.tight_layout(); plt.show()
# Sazonalidade (fator mensal)
plt.figure(figsize=(12,5))
res.seasonal.plot()
plt.title("Sazonalidade multiplicativa (periodicidade 12)")
plt.xlabel("Mês"); plt.ylabel("Fator sazonal")
plt.tight_layout(); plt.show()
# Resíduo
plt.figure(figsize=(12,5))
res.resid.plot()
plt.title("Resíduo (observado / (tendência × sazonalidade))")
plt.xlabel("Mês"); plt.ylabel("Resíduo")
plt.tight_layout(); plt.show()
# Série dessazonalizada (observado / sazonalidade)
dessaz = res.observed / res.seasonal
plt.figure(figsize=(12,5))
dessaz.plot()
plt.title("Obras ativas — dessazonalizada")
plt.xlabel("Mês"); plt.ylabel("Nível dessazonalizado")
plt.tight_layout(); plt.show()
# Diagnóstico rápido
print("Período da série:", ativos.index.min().date(), "→", ativos.index.max().date())
print("Primeiros/últimos NaN na tendência (bordas do filtro):",
      res.trend.isna().head(6).sum(), "/", res.trend.isna().tail(6).sum())

