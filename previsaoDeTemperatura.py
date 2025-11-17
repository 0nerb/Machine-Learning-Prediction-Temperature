#pip install calplot

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import calplot
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import requests

# Configurações
ULAN_BATOR, BOGOTA = "Ulan-bator", "Bogota"
cidade_escolhida = BOGOTA
WEATHER_API_KEY = 'a21922d8762f4e86ae6221530242411'


# ========== FUNÇÕES DE LIMPEZA DE DADOS ==========

def carregar_dados(arquivo):
    """Carrega o arquivo CSV de temperatura."""
    return pd.read_csv(arquivo)


def remover_duplicatas(df, colunas_chave):
    """Remove linhas duplicadas com base nas colunas chave."""
    tamanho_inicial = len(df)
    df_limpo = df[~df.duplicated(subset=colunas_chave, keep=False)]
    tamanho_final = len(df_limpo)
    duplicatas_removidas = tamanho_inicial - tamanho_final
    print(f"Duplicatas removidas: {duplicatas_removidas}")
    return df_limpo


def remover_outliers_temperatura(df, limite_minimo=-99):
    """Remove outliers de temperatura abaixo do limite especificado."""
    outlier_indices = df[df['AvgTemperature'] <= limite_minimo].index
    print(f"Outliers removidos: {len(outlier_indices)}")
    return df.drop(outlier_indices)


def visualizar_boxplot(df):
    """Exibe boxplot da coluna de temperatura."""
    df.boxplot(column=['AvgTemperature'])
    plt.ylabel('Temperatura')
    plt.show()


def contar_cidades_por_ano(df, ano_inicio=1995, ano_fim=2021):
    """Retorna dicionário com quantidade de cidades por ano."""
    cidades_por_ano = {}
    for year in range(ano_inicio, ano_fim):
        cidades_por_ano[year] = len(df[df["Year"] == year]["City"].unique())
    return cidades_por_ano


def plotar_cidades_por_ano(cidades_por_ano):
    """Exibe gráfico de barras de cidades por ano."""
    plt.figure()
    plt.bar(cidades_por_ano.keys(), cidades_por_ano.values())
    plt.xlabel("Ano")
    plt.ylabel("Quantidade de cidades")
    plt.title("Número de cidades por ano")
    plt.show()


def processar_dados_brutos(arquivo_entrada, arquivo_saida):
    """Pipeline completo de limpeza de dados."""
    df = carregar_dados(arquivo_entrada)
    
    # Remover duplicatas
    colunas_duplicatas = ['Region', 'Country', 'State', 'City', 'Month', 'Year', 'Day']
    df = remover_duplicatas(df, colunas_duplicatas)
    
    # Visualizar outliers
    visualizar_boxplot(df)
    
    # Remover outliers
    df = remover_outliers_temperatura(df)
    
    # Mostrar tendência de cidades por ano
    cidades_por_ano = contar_cidades_por_ano(df)
    plotar_cidades_por_ano(cidades_por_ano)
    
    # Salvar dados processados
    df.to_csv(arquivo_saida)
    return df



# ========== FUNÇÕES DE ANÁLISE DE VARIÂNCIA ==========

def calcular_variancia_por_cidade(df):
    """Calcula a variância de temperatura para cada cidade."""
    return df.groupby('City')['AvgTemperature'].var().sort_values()


def obter_cidade_extrema(variancia_series, extremo='min'):
    """Obtém a cidade com menor ou maior variância."""
    if extremo == 'min':
        cidade = variancia_series.idxmin()
        valor = variancia_series.min()
    else:  # max
        cidade = variancia_series.idxmax()
        valor = variancia_series.max()
    return cidade, valor


def analisar_variancia_cidades(arquivo_processado):
    """Analisa e retorna cidades com menor e maior variância."""
    df = pd.read_csv(arquivo_processado)
    variancia = calcular_variancia_por_cidade(df)
    
    cidade_menor, valor_menor = obter_cidade_extrema(variancia, 'min')
    cidade_maior, valor_maior = obter_cidade_extrema(variancia, 'max')
    
    print(f"Cidade menor variância: {cidade_menor} ({valor_menor:.2f})")
    print(f"Cidade maior variância: {cidade_maior} ({valor_maior:.2f})")
    
    return variancia


# ========== FUNÇÕES DE TRANSFORMAÇÃO DE DADOS ==========

def converter_fahrenheit_para_celsius(df, coluna='AvgTemperature'):
    """Converte temperatura de Fahrenheit para Celsius."""
    df_copia = df.copy()
    df_copia[coluna] = (df_copia[coluna] - 32) * 5/9
    return df_copia


def filtrar_por_cidade(df, cidade):
    """Filtra dados de uma cidade específica."""
    return df[df['City'] == cidade]


def adicionar_coluna_data(df):
    """Cria coluna Date combinando Year, Month, Day."""
    df_copia = df.copy()
    df_copia = df_copia.astype({'Year': 'int32', 'Month': 'int32', 'Day': 'int32'})
    df_copia['Date'] = pd.to_datetime(df_copia[['Year', 'Month', 'Day']])
    return df_copia


def preparar_dados_cidade(arquivo_processado, cidade):
    """Pipeline de preparação de dados de uma cidade específica."""
    df = pd.read_csv(arquivo_processado)
    df = filtrar_por_cidade(df, cidade)
    df = converter_fahrenheit_para_celsius(df)
    df.to_csv(f'{cidade}.csv')
    df = adicionar_coluna_data(df)
    return df


def gerar_calplot(df, cidade):
    """Gera calendário de calor com as temperaturas."""
    df_indexed = df.set_index('Date')
    fig, ax = calplot.calplot(df_indexed["AvgTemperature"], how="mean", cmap='jet')
    fig.savefig(f"calplot_{cidade}.png")
    return df.reset_index(drop=True)


# ========== FUNÇÕES DE FEATURES PARA MODELO LINEAR SIMPLES ==========

def adicionar_ordinal_date(df):
    """Adiciona coluna com data em formato ordinal."""
    df_copia = df.copy()
    df_copia['Date_Ordinal'] = df_copia['Date'].apply(lambda date: date.toordinal())
    return df_copia


def extrair_features_simples(df):
    """Extrai features (X) e target (y) para modelo simples."""
    X = df['Date_Ordinal']
    y = df['AvgTemperature']
    return X, y


# ========== FUNÇÕES DE TREINAMENTO ==========

def treinar_modelo_kfold(X, y, n_splits=2):
    """Treina modelo com K-Fold e retorna resultados."""
    kf = KFold(n_splits=n_splits)
    mses_train = []
    mses_test = []
    predicted_dates = []
    predicted_temps = []
    actual_temps = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index].values.reshape(-1, 1), \
                          X.iloc[test_index].values.reshape(-1, 1)
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        mses_train.append(mse_train)
        
        y_pred_test = model.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mses_test.append(mse_test)
        
        predicted_dates.append(test_index)
        predicted_temps.append(y_pred_test)
        actual_temps.append(y_test)
    
    return {
        'mses_train': mses_train,
        'mses_test': mses_test,
        'predicted_dates': predicted_dates,
        'predicted_temps': predicted_temps,
        'actual_temps': actual_temps,
        'model': model
    }


# ========== FUNÇÕES DE VISUALIZAÇÃO ==========

def plotar_mse_por_fold(mses_train, mses_test):
    """Exibe gráfico de MSE para treino e teste."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mses_train) + 1), mses_train, marker='o', linestyle='-', label='Treino')
    plt.plot(range(1, len(mses_test) + 1), mses_test, marker='o', linestyle='-', label='Teste')
    plt.title('MSE por Fold')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def plotar_predicoes_vs_reais(y_train, y_pred_train, y_test, y_pred_test):
    """Exibe gráfico de previsões vs valores reais."""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_train)), y_train.values, label='Valores reais (Treino)', 
                color='blue', alpha=0.6, s=10)
    plt.plot(range(len(y_train)), y_pred_train, label='Previsões (Treino)', color='red', linestyle='-')
    plt.scatter(range(len(y_train), len(y_train) + len(y_test)), y_test.values, 
                label='Valores reais (Teste)', color='orange', alpha=0.6, s=10)
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred_test, 
             label='Previsões (Teste)', color='green', linestyle='-')
    plt.title('Previsões do Modelo - Treino e Teste')
    plt.xlabel('Índice dos Dados')
    plt.ylabel('Temperatura Média (°C)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========== FUNÇÕES DE PREVISÃO SIMPLES ==========

def fazer_predicoes_simples(df, model, ano=2024):
    """Faz previsões para cada mês do ano usando modelo simples."""
    temperaturas_previstas = []
    
    for mes in range(1, 12):
        data = datetime.date(ano, mes, 5)
        data_ordinal = data.toordinal()
        
        df_input = pd.DataFrame({'Date_Ordinal': [data_ordinal]})
        temperatura = model.predict(df_input[['Date_Ordinal']].values)[0]
        temperaturas_previstas.append(round(temperatura, 1))
        
        print(f"Temperatura média prevista 5/{mes}/{ano}: {temperatura:.2f} °C")
    
    return temperaturas_previstas


# ========== FUNÇÕES DE API ==========

def obter_temperatura_real_da_api(cidade, mes, ano=2024):
    """Obtém temperatura real da API weatherapi."""
    data = f"{ano}-{mes}-5"
    url = f"http://api.weatherapi.com/v1/history.json?key={WEATHER_API_KEY}&q={cidade}&dt={data}"
    resposta = requests.get(url)
    temperatura = resposta.json()["forecast"]["forecastday"][0]["day"]["avgtemp_c"]
    return temperatura


def obter_temperaturas_reais_mes(cidade, ano=2024):
    """Obtém temperaturas reais para todos os meses do ano."""
    temperaturas = []
    
    for mes in range(1, 12):
        temperatura = obter_temperatura_real_da_api(cidade, mes, ano)
        temperaturas.append(temperatura)
        print(f"Temperatura média real 5/{mes}/{ano}: {temperatura:.2f} °C")
    
    return temperaturas


# ========== FUNÇÕES DE COMPARAÇÃO ==========

def calcular_erro_percentual(temperatura_prevista, temperatura_real):
    """Calcula erro percentual entre previsão e valor real."""
    erro = abs(temperatura_prevista - temperatura_real)
    erro_percentual = abs((erro / temperatura_real) * 100)
    return erro_percentual


def comparar_predicoes(temperaturas_previstas, temperaturas_reais):
    """Compara previsões com valores reais e exibe erros."""
    for idx in range(len(temperaturas_previstas)):
        erro_pct = calcular_erro_percentual(temperaturas_previstas[idx], temperaturas_reais[idx])
        print(f"Erro percentual para o mês {idx + 1}: {erro_pct:.2f}%")


# ========== FUNÇÕES DE FEATURES AVANÇADAS ==========

def gerar_features_sazonalidade(df):
    """Gera features de sazonalidade (sin, cos) para mês e ano."""
    df_copia = df.copy()
    df_copia['Month_sin'] = np.sin(2 * np.pi * df_copia['Month'] / 12)
    df_copia['Month_cos'] = np.cos(2 * np.pi * df_copia['Month'] / 12)
    df_copia['Year_sin'] = np.sin(2 * np.pi * df_copia['Year'] / df_copia['Year'].max())
    df_copia['Year_cos'] = np.cos(2 * np.pi * df_copia['Year'] / df_copia['Year'].max())
    df_copia['Month_Year_interaction'] = df_copia['Month_sin'] * df_copia['Year_sin']
    return df_copia


def extrair_features_avancadas(df):
    """Extrai features (X) e target (y) para modelo com sazonalidade."""
    X = df[['Day', 'Month_sin', 'Month_cos', 'Year_sin', 'Year_cos', 'Month_Year_interaction']]
    y = df['AvgTemperature']
    return X, y


def criar_features_data_especifica(dia, mes, ano, year_max):
    """Cria features sazonais para uma data específica."""
    month_sin = np.sin(2 * np.pi * mes / 12)
    month_cos = np.cos(2 * np.pi * mes / 12)
    year_sin = np.sin(2 * np.pi * ano / year_max)
    year_cos = np.cos(2 * np.pi * ano / year_max)
    month_year_interaction = month_sin * year_sin
    
    return pd.DataFrame({
        'Day': [dia],
        'Month_sin': [month_sin],
        'Month_cos': [month_cos],
        'Year_sin': [year_sin],
        'Year_cos': [year_cos],
        'Month_Year_interaction': [month_year_interaction]
    })


# ========== FUNÇÕES DE PREVISÃO AVANÇADA ==========

def fazer_predicoes_avancadas(df, model, ano=2024):
    """Faz previsões para cada mês usando features de sazonalidade."""
    temperaturas_previstas = []
    year_max = df['Year'].max()
    
    for mes in range(1, 12):
        df_features = criar_features_data_especifica(5, mes, ano, year_max)
        temperatura = model.predict(df_features)[0]
        temperaturas_previstas.append(round(temperatura, 1))
        
        print(f"Temperatura média prevista 5/{mes}/{ano}: {temperatura:.2f} °C")
    
    return temperaturas_previstas


def plotar_mse_avancado(mses_train, mses_test):
    """Exibe gráfico de MSE para modelo avançado."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(mses_train) + 1), mses_train, marker='o', linestyle='-', 
             label='Treino', color='blue')
    plt.plot(range(1, len(mses_test) + 1), mses_test, marker='o', linestyle='-', 
             label='Teste', color='orange')
    plt.title('Erro Quadrático Médio (MSE)')
    plt.xlabel('Fold')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plotar_predicoes_avancadas(y_train, y_pred_train, y_test, y_pred_test):
    """Exibe gráfico de previsões vs valores reais para modelo avançado."""
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_train)), y_train.values, label='Valores reais (Treino)', 
                color='blue', alpha=0.6, s=10)
    plt.plot(range(len(y_train)), y_pred_train, label='Previsões (Treino)', color='red', linestyle='-')
    plt.scatter(range(len(y_train), len(y_train) + len(y_test)), y_test.values, 
                label='Valores reais (Teste)', color='orange', alpha=0.6, s=10)
    plt.plot(range(len(y_train), len(y_train) + len(y_test)), y_pred_test, 
             label='Previsões (Teste)', color='blue', linestyle='-')
    plt.title('Previsões do Modelo - Treino e Teste')
    plt.xlabel('Índice dos Dados')
    plt.ylabel('Temperatura Média')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========== MAIN ==========

if __name__ == "__main__":
    # Fase 1: Limpeza de dados
    print("=== FASE 1: Limpeza de Dados ===")
    processar_dados_brutos("city_temperature.csv", "temperatura_cidades_processado.csv")
    
    # Fase 2: Análise de variância
    print("\n=== FASE 2: Análise de Variância ===")
    analisar_variancia_cidades("temperatura_cidades_processado.csv")
    
    # Fase 3: Preparação de dados
    print("\n=== FASE 3: Preparação de Dados ===")
    df = preparar_dados_cidade("temperatura_cidades_processado.csv", cidade_escolhida)
    df = gerar_calplot(df, cidade_escolhida)
    df = adicionar_ordinal_date(df)
    
    # Fase 4: Modelo Linear Simples
    print("\n=== FASE 4: Modelo Linear Simples ===")
    X, y = extrair_features_simples(df)
    resultado_simples = treinar_modelo_kfold(X, y, n_splits=2)
    
    plotar_mse_por_fold(resultado_simples['mses_train'], resultado_simples['mses_test'])
    plotar_predicoes_vs_reais(y.iloc[resultado_simples['predicted_dates'][0]], 
                              resultado_simples['predicted_temps'][0],
                              y.iloc[resultado_simples['predicted_dates'][1]], 
                              resultado_simples['predicted_temps'][1])
    
    # Previsões simples
    temps_previstas_simples = fazer_predicoes_simples(df, resultado_simples['model'])
    temps_reais = obter_temperaturas_reais_mes(cidade_escolhida)
    print("\n=== Comparação Modelo Simples ===")
    comparar_predicoes(temps_previstas_simples, temps_reais)
    
    # Fase 5: Modelo com Features Avançadas
    print("\n=== FASE 5: Modelo com Sazonalidade ===")
    df_features = gerar_features_sazonalidade(df)
    X_adv, y_adv = extrair_features_avancadas(df_features)
    resultado_avancado = treinar_modelo_kfold(X_adv, y_adv, n_splits=10)
    
    plotar_mse_avancado(resultado_avancado['mses_train'], resultado_avancado['mses_test'])
    plotar_predicoes_avancadas(y_adv.iloc[resultado_avancado['predicted_dates'][0]], 
                               resultado_avancado['predicted_temps'][0],
                               y_adv.iloc[resultado_avancado['predicted_dates'][1]], 
                               resultado_avancado['predicted_temps'][1])
    
    # Previsões avançadas
    temps_previstas_avancadas = fazer_predicoes_avancadas(df_features, resultado_avancado['model'])
    print("\n=== Comparação Modelo Avançado ===")
    comparar_predicoes(temps_previstas_avancadas, temps_reais)