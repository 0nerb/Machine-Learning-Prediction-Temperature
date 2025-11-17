# Previsão de Temperatura - Machine Learning

Um projeto de machine learning para prever temperaturas médias diárias usando dados históricos e modelos de regressão linear com análise de sazonalidade.

## 📋 Descrição

Este projeto implementa dois modelos de regressão linear para prever temperaturas:

1. **Modelo Linear Simples**: Usa apenas a data (ordinal) como feature
2. **Modelo com Sazonalidade**: Usa features engineered com componentes trigonométricas (sin/cos) para capturar padrões sazonais

O projeto inclui limpeza de dados, exploração, treinamento com K-Fold Cross-Validation e validação contra dados reais da API WeatherAPI.

## 🚀 Funcionalidades

- ✅ **Limpeza de dados**: Remoção de duplicatas e outliers
- ✅ **Análise exploratória**: Boxplot de temperaturas e tendências por ano
- ✅ **Feature engineering**: Geração de features sazonais (sin/cos)
- ✅ **Treinamento com K-Fold**: Validação cruzada em 2 e 10 folds
- ✅ **Visualizações**: Gráficos de MSE, previsões vs valores reais, calendário de calor
- ✅ **Validação em tempo real**: Comparação com dados da API WeatherAPI
- ✅ **Cálculo de métricas**: Erro percentual entre previsões e valores reais

## 📦 Requisitos

```bash
pip install pandas matplotlib numpy scikit-learn requests calplot
```

### Dependências:
- **pandas**: Manipulação e análise de dados
- **numpy**: Computação numérica
- **scikit-learn**: Modelos de machine learning e métricas
- **matplotlib**: Visualização de dados
- **requests**: Chamadas HTTP para API
- **calplot**: Gráficos de calendário de calor

## 🔧 Configuração

### 1. Preparar o dataset

Você precisa de um arquivo `city_temperature.csv` com as seguintes colunas:
- `City`: Nome da cidade
- `Region`: Região/Estado
- `Country`: País
- `Year`: Ano (inteiro)
- `Month`: Mês (1-12)
- `Day`: Dia (1-31)
- `AvgTemperature`: Temperatura média em Fahrenheit

### 2. Configurar a cidade

Edite a variável no início do script:

```python
cidade_escolhida = BOGOTA  # Pode ser ULAN_BATOR ou outra cidade
```

### 3. API Key (opcional)

Para validação com dados reais, configure sua chave da WeatherAPI:

```python
WEATHER_API_KEY = 'sua_chave_aqui'
```

Obtenha gratuitamente em: https://www.weatherapi.com/

## 📊 Estrutura do Código

O código está organizado em funções especializadas:

### Limpeza de Dados
- `carregar_dados()` - Carrega CSV
- `remover_duplicatas()` - Remove linhas duplicadas
- `remover_outliers_temperatura()` - Remove outliers
- `processar_dados_brutos()` - Pipeline completo

### Transformação
- `converter_fahrenheit_para_celsius()` - Converte temperatura
- `filtrar_por_cidade()` - Filtra dados específicos
- `adicionar_coluna_data()` - Cria coluna Date
- `gerar_calplot()` - Cria calendário de calor

### Features
- `adicionar_ordinal_date()` - Converte data para ordinal
- `gerar_features_sazonalidade()` - Gera sin/cos para sazonalidade
- `criar_features_data_especifica()` - Cria features para previsão

### Treinamento
- `treinar_modelo_kfold()` - Treina com K-Fold Cross-Validation
- `fazer_predicoes_simples()` - Previsões modelo linear
- `fazer_predicoes_avancadas()` - Previsões com sazonalidade

### Visualização
- `plotar_mse_por_fold()` - Gráfico de MSE
- `plotar_predicoes_vs_reais()` - Gráfico de previsões
- `plotar_mse_avancado()` - MSE do modelo avançado
- `plotar_predicoes_avancadas()` - Previsões avançadas

### Validação
- `obter_temperatura_real_da_api()` - Busca dados reais
- `obter_temperaturas_reais_mes()` - Obtém 11 meses
- `calcular_erro_percentual()` - Calcula erro %
- `comparar_predicoes()` - Compara previsões vs reais

## 🏃 Como Executar

```bash
python previsaoDeTemperatura.py
```

### Saída esperada:

```
=== FASE 1: Limpeza de Dados ===
Duplicatas removidas: 1234
Outliers removidos: 567

=== FASE 2: Análise de Variância ===
Cidade menor variância: Bogota (45.23)
Cidade maior variância: Ulan-bator (234.56)

=== FASE 3: Preparação de Dados ===
[Exibe boxplot, gráfico de cidades por ano, calendário de calor]

=== FASE 4: Modelo Linear Simples ===
[Treina modelo e exibe gráficos]
Temperatura média prevista 5/1/2024: 21.34 °C
...

=== Comparação Modelo Simples ===
Erro percentual para o mês 1: 5.23%
...

=== FASE 5: Modelo com Sazonalidade ===
[Treina modelo avançado com 10 folds]
...

=== Comparação Modelo Avançado ===
Erro percentual para o mês 1: 2.15%
...
```

## 📈 Resultados

O script gera:
- **Gráficos interativos** (MSE, previsões, boxplot)
- **Imagem calendário** (`calplot_Bogota.png`)
- **CSVs processados**:
  - `temperatura_cidades_processado.csv` - Dados limpos
  - `Bogota.csv` - Dados da cidade

## 🔍 Análise dos Modelos

### Modelo Linear Simples
- **Features**: Data ordinal
- **K-Fold**: 2 splits
- **Métrica**: MSE

### Modelo com Sazonalidade
- **Features**: 
  - `Day`: Dia do mês
  - `Month_sin`: sin(2π × mês / 12)
  - `Month_cos`: cos(2π × mês / 12)
  - `Year_sin`: sin(2π × ano / ano_máximo)
  - `Year_cos`: cos(2π × ano / ano_máximo)
  - `Month_Year_interaction`: Interação entre mês e ano
- **K-Fold**: 10 splits
- **Métrica**: MSE

## 📝 Notas Importantes

1. **Arquivo grande**: O `city_temperature.csv` geralmente excede 100MB. Adicione ao `.gitignore`
2. **API Rate Limit**: WeatherAPI tem limite gratuito de 1.000 chamadas/dia
3. **Temperaturas em Celsius**: O código converte automaticamente de Fahrenheit
4. **Visualizações**: Requerem interface gráfica (X11 em SSH)

## 🐛 Troubleshooting

### Erro: "No module named 'sklearn'"
```bash
pip install scikit-learn
```

### Erro: "File not found 'city_temperature.csv'"
Certifique-se que o arquivo está no mesmo diretório do script

### Erro: "API key invalid"
Verifique sua chave em https://www.weatherapi.com/

### Gráficos não aparecem (SSH/Linux)
Use backend diferente:
```python
import matplotlib
matplotlib.use('Agg')  # Adicione antes de importar pyplot
```

## 📚 Referências

- [scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [WeatherAPI Documentation](https://www.weatherapi.com/docs/)
- [K-Fold Cross-Validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html)

## 🤝 Contribuindo

Sugestões de melhorias:
- Testar outros modelos (PolynomialRegression, RandomForest)
- Adicionar mais features (umidade, pressão)
- Implementar Prophet ou ARIMA para series temporais
- Otimizar hiperparâmetros

## 📄 Licença

Projeto de estudo pessoal

## ✍️ Autor

Breno Krang

---

**Última atualização**: Novembro de 2025
