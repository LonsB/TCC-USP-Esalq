#%%
import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import statsmodels.api as sm 
from scipy import stats
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from statsmodels.discrete.discrete_model import MNLogit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix,f1_score, roc_auc_score,
                             roc_curve,auc)

import warnings
warnings.filterwarnings('ignore')
#%% Import da base de dados

df = pd.read_csv('top5_leagues_data.csv',delimiter=',')

# Características das variáveis do dataset
df.info()

# Estatísticas univariadas
df.describe()

#%% Remoção das estatísticas de Bets

for i, col in enumerate(df.columns):
    print(i, col)

df_filtrado = df.drop(columns=['B365H', 'B365D', 'B365A',
                               'BWH','BWD', 'BWA',
                               'BFH','BFD', 'BFA',
                               'PSH','PSD', 'PSA',
                               'WHH','WHD', 'WHA',
                               '1XBH','1XBD', '1XBA',
                               'BFEH','BFED', 'BFEA',
                               'B365>2.5','B365<2.5', 'P>2.5',
                               'P<2.5','BFE>2.5', 'BFE<2.5',
                               'B365AHH','B365AHA', 'PAHH',
                               'PAHA','BFEAHH', 'BFEAHA',
                               'B365CH','B365CD', 'B365CA',
                               'BWCH','BWCD', 'BWCA',
                               'BFCH','BFCD', 'BFCA',
                               'PSCH','PSCD', 'PSCA',
                               'WHCH','WHCD', 'WHCA',
                               '1XBCH','1XBCD', '1XBCA',
                               'B365C>2.5','B365C<2.5', 'PC>2.5',
                               'PC<2.5','BFEC>2.5', 'BFEC<2.5',
                               'AHCh','B365CAHH', 'B365CAHA',
                               'PCAHH','PCAHA', 'BFECAHH',
                               'BFECAHA','BFECH', 'BFECD', 'BFECA',
                               'MaxH', 'MaxD', 'MaxA',
                               'AvgH', 'AvgD', 'AvgA',
                               'Max>2.5', 'Max<2.5', 'Avg>2.5',
                               'Avg<2.5', 'AHh', 'MaxAHH',
                               'MaxAHA', 'AvgAHH', 'AvgAHA',
                               'AvgCH', 'AvgCD', 'AvgCA',
                               'MaxC>2.5', 'MaxC<2.5', 'AvgC>2.5',
                               'AvgC<2.5', 'MaxCAHH', 'MaxCAHA',
                               'over_implied_probs', 'under_implied_probs', 'books_margin_uo',
                               'over_fair_probs', 'under_fair_probs', 'home_implied_probs',
                               'draw_implied_probs', 'away_implied_probs', 'books_margin_res',
                               'home_fair_probs', 'draw_fair_probs', 'away_fair_probs',
                               'MaxCH', 'MaxCD', 'MaxCA','Season','Over_Under_2.5',
                               'AvgCAHH', 'AvgCAHA','Unnamed: 0','over_under_ht',
                               'home_elo_after','away_elo_after'
                               ])

  #%%
for col in df_filtrado.columns:
    print(f"\n===== {col} =====")
   
#%% Tradução e Ajuste de nome das colunas

mapa_colunas = {

# Identificação da partida
'Div': 'divisao',
'League': 'liga',
'Season': 'temporada',
'Date': 'data',
'Time': 'hora',
'HomeTeam': 'time_mandante',
'AwayTeam': 'time_visitante',
'Referee': 'arbitro',

# Resultado da partida
'FTHG': 'gols_mandante_final',
'FTAG': 'gols_visitante_final',
'FTR': 'resultado_final',
'HTHG': 'gols_mandante_intervalo',
'HTAG': 'gols_visitante_intervalo',
'HTR': 'resultado_intervalo',

# Estatísticas da partida (jogo atual)
'HS': 'finalizacoes_mandante',
'AS': 'finalizacoes_visitante',
'HST': 'finalizacoes_no_alvo_mandante',
'AST': 'finalizacoes_no_alvo_visitante',
'HF': 'faltas_mandante',
'AF': 'faltas_visitante',
'HC': 'escanteios_mandante',
'AC': 'escanteios_visitante',
'HY': 'cartoes_amarelos_mandante',
'AY': 'cartoes_amarelos_visitante',
'HR': 'cartoes_vermelhos_mandante',
'AR': 'cartoes_vermelhos_visitante',

# Variáveis derivadas do jogo
'total_goals': 'total_gols_partida',
'Over_Under_2.5': 'mais_menos_2_5_gols',
'total_ht_goals': 'total_gols_intervalo',
'over_under_ht': 'mais_menos_gols_intervalo',
'total_shots': 'total_finalizacoes',
'total_shots_on_target': 'total_finalizacoes_no_alvo',

# Eficiência ofensiva
'shots_per_goal': 'finalizacoes_por_gol',
'shots_per_goal_home': 'finalizacoes_por_gol_mandante',
'shots_per_goal_away': 'finalizacoes_por_gol_visitante',

'total_shot_accuracy': 'precisao_finalizacoes_total',
'total_shot_accuracy_home': 'precisao_finalizacoes_mandante',
'total_shot_accuracy_away': 'precisao_finalizacoes_visitante',

'shot_conversion': 'conversao_finalizacoes',
'shot_conversion_home': 'conversao_finalizacoes_mandante',
'shot_conversion_away': 'conversao_finalizacoes_visitante',

# Sequências (streaks)
'Home_WinStreak': 'sequencia_vitorias_mandante',
'Home_DrawStreak': 'sequencia_empates_mandante',
'Home_LossStreak': 'sequencia_derrotas_mandante',

'Away_WinStreak': 'sequencia_vitorias_visitante',
'Away_DrawStreak': 'sequencia_empates_visitante',
'Away_LossStreak': 'sequencia_derrotas_visitante',

# Forma recente (últimos 5 jogos)
'Home_AvgMarginVictory_last5': 'media_margem_vitoria_mandante_ultimos5',
'Away_AvgMarginVictory_last5': 'media_margem_vitoria_visitante_ultimos5',

'Home_AvgGoalDiff_last5': 'media_saldo_gols_mandante_ultimos5',
'Away_AvgGoalDiff_last5': 'media_saldo_gols_visitante_ultimos5',

'Home_WinRate_last5': 'taxa_vitorias_mandante_ultimos5',
'Home_DrawRate_last5': 'taxa_empates_mandante_ultimos5',
'Home_LossRate_last5': 'taxa_derrotas_mandante_ultimos5',

'Away_WinRate_last5': 'taxa_vitorias_visitante_ultimos5',
'Away_DrawRate_last5': 'taxa_empates_visitante_ultimos5',
'Away_LossRate_last5': 'taxa_derrotas_visitante_ultimos5',

'Home_Points_last5': 'pontos_mandante_ultimos5',
'Away_Points_last5': 'pontos_visitante_ultimos5',

# Elo rating
'home_elo_before': 'elo_mandante_antes',
'away_elo_before': 'elo_visitante_antes',
'home_elo_after': 'elo_mandante_depois',
'away_elo_after': 'elo_visitante_depois',

'elo_diff': 'diferenca_elo',
'elo_G': 'fator_g_elo',
'elo_change_home': 'variacao_elo_mandante',
'elo_change_away': 'variacao_elo_visitante',

# Tendências e rolling stats
'home_elo_trend_5': 'tendencia_elo_mandante_5jogos',
'home_goals_scored_rolling_mean_5': 'media_gols_marcados_mandante_5jogos',
'away_goals_conceded_rolling_mean_5': 'media_gols_sofridos_visitante_5jogos',

'home_form_ratio': 'indice_forma_mandante',
'away_form_ratio': 'indice_forma_visitante',

'home_gd_roll5': 'saldo_gols_mandante_5jogos',
'away_gd_roll5': 'saldo_gols_visitante_5jogos',

'avg_goal_diff_last5': 'media_saldo_gols_ultimos5',

'home_pts_last5_tmp': 'pontos_mandante_ultimos5_tmp',
'away_pts_last5_tmp': 'pontos_visitante_ultimos5_tmp',

'form_diff_points_last5': 'diferenca_pontos_forma_ultimos5',

# Diferenças diretas
'shots_on_target_diff': 'diferenca_finalizacoes_no_alvo',

# Defesa
'Home_CleanSheets_last5': 'jogos_sem_sofrer_gols_mandante_ultimos5',
'Away_CleanSheets_last5': 'jogos_sem_sofrer_gols_visitante_ultimos5',

'Home_AvgGoalsConceded_last5': 'media_gols_sofridos_mandante_ultimos5',
'Away_AvgGoalsConceded_last5': 'media_gols_sofridos_visitante_ultimos5',

'Home_BTTS_last5': 'ambos_marcam_mandante_ultimos5'
}
df_filtrado = df_filtrado.rename(columns=mapa_colunas)
df_filtrado.head()  


#%% Verificação básica das colunas disponíveis
for col in df_filtrado.columns:
    print(f"\n===== {col} =====")
    df_filtrado[col].info()
    df_filtrado[col].describe()
#%% Ajuste da base de campeonatos
mapa_camp = {
    'SP1': 'Espanhol',
    'E0': 'Inglês',
    'F1': 'Francês',
    'I1': 'Italiano',
    'D1': 'Alemão'
}

df_filtrado['divisao'] = df_filtrado['divisao'].map(mapa_camp)
df_filtrado = df_filtrado.rename(columns={'divisao': 'campeonato'})

   
#%% Criação da coluna auxliar resultado_final2, que nada mais é
# que a coluna resultado_final numérica

df_filtrado['resultado_final'].value_counts().sort_index()

#Dataframe auxiliar que não impacta o original
df_filtrado_multinomial=df_filtrado.copy()

df_filtrado_multinomial.loc[df_filtrado_multinomial['resultado_final']==
                            'H',
                            'resultado_final2'] = 0 #categoria de referência

df_filtrado_multinomial.loc[df_filtrado_multinomial['resultado_final']==
                            'D',
                            'resultado_final2'] = 1

df_filtrado_multinomial.loc[df_filtrado_multinomial['resultado_final']==
                            'A',
                            'resultado_final2'] = 2

# Definição do tipo 'int' para a variável dependente 'resultado_final2'
df_filtrado_multinomial['resultado_final2'] =\
    df_filtrado_multinomial['resultado_final2'].astype('int64')

df_filtrado_multinomial.info()

df_filtrado_multinomial
#%%
#Cálculo do Baseline de acurácia, baseado na Vitória do Mandante
#Esse será o ponto de partida para entender a eficiencia dos modelos propostos
df_filtrado_multinomial['resultado_final'].value_counts(normalize=True) 


#%% Estimação do modelo logístico multinomial 1

x = df_filtrado_multinomial[['elo_mandante_antes','elo_visitante_antes']]
y = df_filtrado_multinomial['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo 'modelo_elo'
modelo_elo.summary()

# Definindo uma função 'Qui2' para se extrair a estatística geral
# do modelo

def Qui2(modelo_elo):
    maximo = modelo_elo.llf
    minimo = modelo_elo.llnull
    qui2 = -2*(minimo - maximo)
    pvalue = stats.distributions.chi2.sf(qui2,4) # 4 graus de liberdade
    df = pd.DataFrame({'Qui quadrado':[qui2],
                       'pvalue':[pvalue]})
    return df

#%%: Estatística geral do 'modelo_elo'

Qui2(modelo_elo)


#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial['predicao'] = predicao
df_filtrado_multinomial

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial.loc[df_filtrado_multinomial['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial.loc[df_filtrado_multinomial['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial.loc[df_filtrado_multinomial['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'

df_filtrado_multinomial

#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial,
                       index=['predicao_label'],
                       columns=['resultado_final'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table

#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo = table.diagonal().sum()/table.sum()
acuracia_modelo_elo

#%% Plotagens das probabilidades

# Plotagem das smooth probability lines para a variável 'elo_mandante_antes'

# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

plt.figure(figsize=(15,10))

# Plot para "Vitória Mandante"
sns.regplot(x='elo_mandante_antes', y=df_filtrado_multinomial[0],
            data=df_filtrado_multinomial, ci=False, order=4,
            line_kws={'color':'indigo', 'linewidth':4,
                      'label':'Vitória Mandante'},
            scatter_kws={'color':'indigo', 's':80, 'alpha':0.5})

# Plot para "Empate"
sns.regplot(x='elo_mandante_antes', y=df_filtrado_multinomial[1],
            data=df_filtrado_multinomial, ci=None, order=4,
            line_kws={'color':'darkgreen', 'linewidth':4,
                      'label':'Empate'},
            scatter_kws={'color':'darkgreen', 's':80, 'alpha':0.5})

# Plot para "Vitória Visitante"
sns.regplot(x='elo_mandante_antes', y=df_filtrado_multinomial[2],
            data=df_filtrado_multinomial, ci=None, order=4,
            line_kws={'color':'darkorange', 'linewidth':4,
                      'label':'Vitória Visitante'},
            scatter_kws={'color':'darkorange', 's':80, 'alpha':0.5})

plt.xlabel('Elo Mandante Antes', fontsize=18)
plt.ylabel('Probabilidades', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='center left', fontsize=14)
plt.show()

#%% Plotagens das probabilidades

# Plotagem das smooth probability lines para a variável 'sem'

# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

plt.figure(figsize=(15,10))

# Plot para "Vitória Mandante"
sns.regplot(x='elo_visitante_antes', y=df_filtrado_multinomial[0],
            data=df_filtrado_multinomial, ci=None, order=4,
            line_kws={'color':'indigo', 'linewidth':4,
                      'label':'Vitória Mandante'},
            scatter_kws={'color':'indigo', 's':80, 'alpha':0.5})

# Plot para "Empate"
sns.regplot(x='elo_visitante_antes', y=df_filtrado_multinomial[1],
            data=df_filtrado_multinomial, ci=None, order=4,
            line_kws={'color':'darkgreen', 'linewidth':4,
                      'label':'Empate'},
            scatter_kws={'color':'darkgreen', 's':80, 'alpha':0.5})

# Plot para "Vitória Visitante"
sns.regplot(x='elo_visitante_antes', y=df_filtrado_multinomial[2],
            data=df_filtrado_multinomial, ci=None, order=4,
            line_kws={'color':'darkorange', 'linewidth':4,
                      'label':'Vitória Visitante'},
            scatter_kws={'color':'darkorange', 's':80, 'alpha':0.5})

plt.xlabel('Elo Visitante', fontsize=18)
plt.ylabel('Probabilidades', fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='upper center', fontsize=14)
plt.show()
#%%Dataframe para o segundo modelo
df_filtrado_multinomial2=df_filtrado_multinomial.drop(['predicao','predicao_label',0,1,2],axis=1)

df_filtrado_multinomial2.columns

#%% Estimação do modelo logístico multinomial 2

x = df_filtrado_multinomial2['diferenca_elo']
y = df_filtrado_multinomial2['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo2 = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo
modelo_elo2.summary()


#%% Estatística geral do

Qui2(modelo_elo2)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo2.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial2 = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial2

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial2['predicao'] = predicao
df_filtrado_multinomial2

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial2.loc[df_filtrado_multinomial2['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial2.loc[df_filtrado_multinomial2['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial2.loc[df_filtrado_multinomial2['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'

df_filtrado_multinomial2

#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial2,
                       index=['predicao_label'],
                       columns=['resultado_final'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo2 = table.diagonal().sum()/table.sum()
acuracia_modelo_elo2

#%% Dtaaframe para o terceiro modelo
df_filtrado_multinomial3=df_filtrado_multinomial.drop(['predicao','predicao_label',0,1,2],axis=1)

df_filtrado_multinomial3.columns

#%%Criação da variável diff_elo_abs para interpretação da variação do elo entre as equipes
df_filtrado_multinomial3['diff_elo_abs'] = abs(df_filtrado_multinomial3['diferenca_elo'])

#%% Estimação do modelo logístico multinomial 3

x = df_filtrado_multinomial3['diff_elo_abs']
y = df_filtrado_multinomial3['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo3 = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo
modelo_elo3.summary()


#%% Estatística geral do modelo

Qui2(modelo_elo3)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo3.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial3 = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial3

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial3['predicao'] = predicao
df_filtrado_multinomial3

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial3.loc[df_filtrado_multinomial3['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial3.loc[df_filtrado_multinomial3['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial3.loc[df_filtrado_multinomial3['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'

df_filtrado_multinomial3

#% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial3,
                       index=['predicao_label'],
                       columns=['resultado_final'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo3 = table.diagonal().sum()/table.sum()
acuracia_modelo_elo3


#%% Dataframe para o modelo 4
df_filtrado_multinomial4=df_filtrado_multinomial.drop(['predicao','predicao_label',0,1,2],axis=1)

df_filtrado_multinomial4.columns

#%% Criação da variável diff_elo_abs para interpretação da variação do elo entre as equipes
df_filtrado_multinomial4['diff_elo_abs'] = abs(df_filtrado_multinomial4['diferenca_elo'])

#%% Estimação do modelo logístico multinomial

x = df_filtrado_multinomial4[['diferenca_elo','diff_elo_abs']]
y = df_filtrado_multinomial4['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo4 = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo
modelo_elo4.summary()


#%% Estatística geral do modelo

Qui2(modelo_elo4)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo4.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial4 = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial4

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial4['predicao'] = predicao
df_filtrado_multinomial4

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial4.loc[df_filtrado_multinomial4['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial4.loc[df_filtrado_multinomial4['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial4.loc[df_filtrado_multinomial4['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'

df_filtrado_multinomial4

#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial4,
                       index=['predicao_label'],
                       columns=['resultado_final'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo4 = table.diagonal().sum()/table.sum()
acuracia_modelo_elo4

#%% dataframe para o modelo 5
colunas_modelo = [
    'diferenca_elo',
    'diferenca_pontos_forma_ultimos5',
    'media_saldo_gols_ultimos5',
    'resultado_final2'
]

df_filtrado_multinomial5 = (
    df_filtrado_multinomial[colunas_modelo]
    .dropna()
)

#%% Estimação do modelo logístico multinomial


x = df_filtrado_multinomial5[[ 'diferenca_elo',
                              'diferenca_pontos_forma_ultimos5',
                              'media_saldo_gols_ultimos5' ]]

y = df_filtrado_multinomial5['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo5 = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo
modelo_elo5.summary()

#%% Estatística geral do modelo

Qui2(modelo_elo5)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo5.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial5 = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial5

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial5['predicao'] = predicao
df_filtrado_multinomial5

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial5.loc[df_filtrado_multinomial5['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial5.loc[df_filtrado_multinomial5['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial5.loc[df_filtrado_multinomial5['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'

df_filtrado_multinomial5

#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial5,
                       index=['predicao_label'],
                       columns=['resultado_final'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo5 = table.diagonal().sum()/table.sum()
acuracia_modelo_elo5

#%%Entendendo como funciona a dinâmica dos empates

df_empates = df_filtrado[df_filtrado['resultado_final'] == 'D']
df_nao_empates = df_filtrado[df_filtrado['resultado_final'] != 'D']

estatisticas_empates = (
    df_filtrado.groupby('resultado_final')[
        [
            'diferenca_elo',
            'diferenca_pontos_forma_ultimos5'
        ]
    ]
    .agg([
        'mean',
        'median',
        'std',
        'min',
        'max'
    ])
    .round(2)
)
#%% Seguindo o estudo do comportamento dos empates
print(estatisticas_empates)

df_empates = df_filtrado[df_filtrado['resultado_final'] == 'D']
df_nao_empates = df_filtrado[df_filtrado['resultado_final'] != 'D']

estatisticas_empates = (
    df_filtrado.groupby('resultado_final')[
[
    'media_gols_marcados_mandante_5jogos',
    'media_gols_sofridos_visitante_ultimos5',
    'jogos_sem_sofrer_gols_mandante_ultimos5',
    'taxa_empates_mandante_ultimos5',
    'taxa_empates_visitante_ultimos5'
]
    ]
    .agg([
        'mean',
        'median',
        'std',
        'min',
        'max'
    ])
    .round(2)
)

print(estatisticas_empates)
#%% Criando as colunas abs_elo_diff e faixa_elo, para estudo das caracteristicas dos
#empates

df_filtrado['abs_elo_diff'] = abs(df_filtrado['diferenca_elo'])

df_filtrado['faixa_elo'] = pd.qcut(
    abs(df_filtrado['diferenca_elo']),
    q=5
)

tabela = pd.crosstab(
    df_filtrado['faixa_elo'],
    df_filtrado['resultado_final'],
    normalize='index'
).round(3)

print(tabela)
#%% Verificação gráfica dos empates por faixa de diferença de elo

taxa_empates = (
    df_filtrado.assign(abs_elo_diff=abs(df_filtrado['diferenca_elo']))
      .groupby('faixa_elo')['resultado_final']
      .apply(lambda x: (x == 'D').mean())
)

taxa_empates.plot(kind='bar')

#%% Criando a coluna de faixa de pontos

df_filtrado['faixa_pontos_recentes'] = pd.qcut(
    abs(df_filtrado['diferenca_pontos_forma_ultimos5']),
    q=5
)

tabela = pd.crosstab(
    df_filtrado['faixa_pontos_recentes'],
    df_filtrado['resultado_final'],
    normalize='index'
).round(3)

print(tabela)
#%% Criando a coluna de equilibrio extremo, colocando um intervalo no valor 22
#valor onde ocorre a mudança no comportamento da taxa de empates
df_filtrado['equilibrio_extremo_elo'] = (
    abs(df_filtrado['diferenca_elo']) <= 22
).astype(int)

pd.crosstab(
    df_filtrado['equilibrio_extremo_elo'],
    df_filtrado['resultado_final'],
    normalize='index'
).round(3)
#%% Verificando a correlação entre o equilibrio extremo e o resultado


tabela = pd.crosstab(
    df_filtrado['equilibrio_extremo_elo'],
    df_filtrado['resultado_final']
)

chi2, p, _, _ = chi2_contingency(tabela)

print(chi2, p)


#%% Dataframe do modelo 6
colunas_modelo = [
    'diferenca_elo',
    'diferenca_pontos_forma_ultimos5',
    'resultado_final2'
]

df_filtrado_multinomial6 = (
    df_filtrado_multinomial[colunas_modelo]
    .dropna()
)

df_filtrado_multinomial6['equilibrio_extremo_elo'] = (
    abs(df_filtrado_multinomial6['diferenca_elo']) <= 22
).astype(int)

#%% Estimação do modelo logístico multinomial


x = df_filtrado_multinomial6[[ 'diferenca_elo',
                              'diferenca_pontos_forma_ultimos5',
                              'equilibrio_extremo_elo' ]]

y = df_filtrado_multinomial6['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo6 = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo
modelo_elo6.summary()

#%%Estatística geral do modelo

Qui2(modelo_elo6)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo6.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial6 = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial6

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial6['predicao'] = predicao
df_filtrado_multinomial6

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial6.loc[df_filtrado_multinomial6['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial6.loc[df_filtrado_multinomial6['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial6.loc[df_filtrado_multinomial6['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'


#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial6,
                       index=['predicao_label'],
                       columns=['resultado_final'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo6= table.diagonal().sum()/table.sum()
acuracia_modelo_elo6

#%% Tentando entender melhor os empates

# 1. DISTRIBUIÇÃO DOS RESULTADOS POR CAMPEONATO


# Tabela percentual
resultado_campeonato = pd.crosstab(
    df_filtrado_multinomial['campeonato'],
    df_filtrado_multinomial['resultado_final'],
    normalize='index'
).round(3)

print(resultado_campeonato)

# Gráfico empilhado
resultado_campeonato.plot(
    kind='bar',
    stacked=True,
    figsize=(14,7)
)

plt.title('Distribuição dos Resultados por Campeonato')
plt.ylabel('Proporção')
plt.xlabel('Campeonato')
plt.legend(title='Resultado')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%% 2. BOXPLOT DO ELO_DIFF POR RESULTADO EM CADA CAMPEONATO

plt.figure(figsize=(16,8))

sns.boxplot(
    data=df_filtrado_multinomial,
    x='campeonato',
    y='diferenca_elo',
    hue='resultado_final'
)

plt.title('Distribuição do Elo Diff por Resultado e Campeonato')
plt.xlabel('Campeonato')
plt.ylabel('Diferença de Elo')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%% 3. MÉDIA DO ELO_DIFF POR RESULTADO EM CADA CAMPEONATO

media_elo_resultado = (
    df_filtrado_multinomial
    .groupby(['campeonato', 'resultado_final'])['diferenca_elo']
    .mean()
    .unstack()
    .round(2)
)

print(media_elo_resultado)

# Gráfico
media_elo_resultado.plot(
    kind='bar',
    figsize=(14,7)
)

plt.title('Média do Elo Diff por Resultado e Campeonato')
plt.ylabel('Média da Diferença de Elo')
plt.xlabel('Campeonato')
plt.axhline(0, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% 4. PROPORÇÃO DE EMPATES POR QUANTIS DE |ELO_DIFF|

# Valor absoluto
# Diferença absoluta de Elo
df_filtrado_multinomial['abs_elo_diff'] = (
    df_filtrado_multinomial['diferenca_elo'].abs()
)

# Criação dos quantis
df_filtrado_multinomial['faixa_elo'] = pd.qcut(
    df_filtrado_multinomial['abs_elo_diff'],
    q=5,
    duplicates='drop'
)

# Tabela consolidada
tabela_empates = pd.crosstab(
    df_filtrado_multinomial['faixa_elo'],
    df_filtrado_multinomial['resultado_final'],
    normalize='index'
)

# Taxa de empates
taxa_empates = tabela_empates['D'] * 100

# Rótulos mais amigáveis
labels = [
    'Muito Equilibrado',
    'Equilibrado',
    'Intermediário',
    'Desequilibrado',
    'Muito Desequilibrado'
]

plt.figure(figsize=(9,6))

ax = taxa_empates.plot(
    kind='bar'
)

plt.title(
    'Taxa de Empates por Faixa de Diferença Absoluta de Elo'
)

plt.ylabel('Taxa de Empates (%)')
plt.xlabel('Faixa de Diferença Absoluta de Elo')

for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.1f}%',
        (p.get_x() + p.get_width()/2, p.get_height()),
        ha='center',
        va='bottom'
    )

plt.tight_layout()
plt.show()

#%% 5. HEATMAP DA DISTRIBUIÇÃO DOS RESULTADOS

plt.figure(figsize=(10,8))

sns.heatmap(
    resultado_campeonato,
    annot=True,
    cmap='Blues',
    fmt='.2f'
)

plt.title('Distribuição Percentual dos Resultados por Campeonato')
plt.tight_layout()
plt.show()

#%% dataframe Modelo 7

colunas_modelo = [
    'diferenca_elo',
    'diferenca_pontos_forma_ultimos5',
    'campeonato',
    'resultado_final2'
]

df_filtrado_multinomial7 = (
    df_filtrado_multinomial[colunas_modelo]
    .dropna()
)

#%% Estimação do modelo logístico multinomial
#Novo modelo tentando diferenciar os resultados por campeonato

dummies_campeonato = pd.get_dummies(
    df_filtrado_multinomial7['campeonato'],
    drop_first=True
).astype(float)

x = pd.concat(
    [
        df_filtrado_multinomial7[
            [
                'diferenca_elo',
                'diferenca_pontos_forma_ultimos5'
            ]
        ],

        dummies_campeonato

    ],
    axis=1
)

x = x.astype(float)


y = df_filtrado_multinomial7['resultado_final2']

# Esse pacote precisa que a constante seja definida pelo usuário
X = sm.add_constant(x)

# Estimação do modelo - função 'MNLogit' ('statsmodels.discrete.discrete_model')
modelo_elo7 = MNLogit(endog=y, exog=X).fit()

# Parâmetros do modelo
modelo_elo7.summary()

#%% Estatística geral do modelo

Qui2(modelo_elo7)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_elo7.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial7 = pd.concat([df_filtrado_multinomial, phats], axis=1)
df_filtrado_multinomial7

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial7['predicao'] = predicao
df_filtrado_multinomial7

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Vitória Mandante
# 1: Empate
# 2: Vitória Visitante

df_filtrado_multinomial7.loc[df_filtrado_multinomial7['predicao']==0,
                            'predicao_label'] ='Vitória Mandante'
df_filtrado_multinomial7.loc[df_filtrado_multinomial7['predicao']==1,
                            'predicao_label'] ='Empate'
df_filtrado_multinomial7.loc[df_filtrado_multinomial7['predicao']==2,
                            'predicao_label'] ='Vitória Visitante'


#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial7,
                       index=['predicao_label'],
                       columns=['resultado_final2'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_elo7 = table.diagonal().sum()/table.sum()
acuracia_modelo_elo7

#%% Criando uma váriavel para o Empate no df base

df_filtrado_multinomial['empate'] = (
    df_filtrado_multinomial['resultado_final'] == 'D'
).astype(int)
#%% Criando a pontuação do campeonato

df_filtrado_multinomial = df_filtrado_multinomial.sort_values('data').reset_index(drop=True)

# ID único temporário para cada partida
df_filtrado_multinomial['match_id'] = df_filtrado_multinomial.index 

# Atribuindo pontos da partida
df_filtrado_multinomial['pontos_mandante'] = np.where(df_filtrado_multinomial['resultado_final'] == 'H', 3, 
                                             np.where(df_filtrado_multinomial['resultado_final'] == 'D', 1, 0))

df_filtrado_multinomial['pontos_visitante'] = np.where(df_filtrado_multinomial['resultado_final'] == 'A', 3, 
                                              np.where(df_filtrado_multinomial['resultado_final'] == 'D', 1, 0))

# Separar os dados em "Mandantes" e "Visitantes"
df_m = df_filtrado_multinomial[['match_id', 'data', 'time_mandante', 'pontos_mandante']].copy()
df_m.columns = ['match_id', 'data', 'time', 'pontos_ganhos']
df_m['mando'] = 'mandante'

df_v = df_filtrado_multinomial[['match_id', 'data', 'time_visitante', 'pontos_visitante']].copy()
df_v.columns = ['match_id', 'data', 'time', 'pontos_ganhos']
df_v['mando'] = 'visitante'

# Empilhando tudo em um único DataFrame ordenado cronologicamente
df_times = pd.concat([df_m, df_v]).sort_values('data')

# pontuação acumulada por time, removendo o pontos da partida da data atual
df_times['pontuacao_acumulada'] = df_times.groupby('time')['pontos_ganhos'].cumsum() - df_times['pontos_ganhos']

# Separar de volta os dados de mandantes e visitantes para cruzar
pontos_m = df_times[df_times['mando'] == 'mandante'][['match_id', 'pontuacao_acumulada']]
pontos_v = df_times[df_times['mando'] == 'visitante'][['match_id', 'pontuacao_acumulada']]

# Renomeando as colunas calculadas para o mapeamento correto
pontos_m = pontos_m.rename(columns={'pontuacao_acumulada': 'pts_acumulados_mandante'})
pontos_v = pontos_v.rename(columns={'pontuacao_acumulada': 'pts_acumulados_visitante'})

# merge de volta no df base
df_filtrado_multinomial = df_filtrado_multinomial.merge(pontos_m, on='match_id').merge(pontos_v, on='match_id')
#%% Limpando as colunas auxiliares que foram geradas no processo
df_filtrado_multinomial.drop(columns=['match_id', 'pontos_mandante', 'pontos_visitante'],inplace=True)
#%%

df_filtrado_multinomial['diferenca_pontuacao_mandante_visitante']=df_filtrado_multinomial['pts_acumulados_mandante']-df_filtrado_multinomial['pts_acumulados_visitante']
#%% Criando o modelo  Empate

colunas_modelo = [
    'diferenca_elo','diferenca_pontuacao_mandante_visitante',
    'diferenca_pontos_forma_ultimos5',
    'empate',
]

df_filtrado_multinomial8 = (
    df_filtrado_multinomial[colunas_modelo]
    .dropna()
)

#%% Estimação do modelo logístico multinomial
#Novo modelo tentando diferenciar os resultados por campeonato
from statsmodels.discrete.discrete_model import Logit

x = df_filtrado_multinomial8[
    [
        'diferenca_elo',
        'diferenca_pontos_forma_ultimos5',
        'diferenca_pontuacao_mandante_visitante'
    ]
].copy()

y = df_filtrado_multinomial8['empate'].copy()

x = x.replace([np.inf, -np.inf], np.nan)

dados_modelo = pd.concat([x, y], axis=1).dropna()

x = dados_modelo.drop(columns='empate')
y = dados_modelo['empate']

X = sm.add_constant(x)

modelo_empate = Logit(y, X).fit()

# Parâmetros do modelo
modelo_empate.summary()

#%%: Estatística geral do modelo

Qui2(modelo_empate)

#%% Adicionando as probabilidades de ocorrência de cada uma das
#categorias de Y definidas pela modelagem ao dataframe original, bem como a
#respectiva classificação

# Probabilidades de ocorrência das três categoriais
# Definição do array 'phats':
phats = modelo_empate.predict()
phats

# Transformação do array 'phats' para o dataframe 'phats':
phats = pd.DataFrame(phats)
phats

# Concatenando o dataframe original com o dataframe 'phats':
df_filtrado_multinomial8 = pd.concat([df_filtrado_multinomial8, phats], axis=1)

# Analisando o resultado de acordo com a categoria de resposta:
predicao = phats.idxmax(axis=1)
predicao

# Adicionando a categoria de resposta 'predicao' ao dataframe original,
#por meio da criação da variável 'predicao'
df_filtrado_multinomial8['predicao'] = predicao
df_filtrado_multinomial8

# Criando a variável 'predicao_label' a partir da variável 'predicao',
#respeitando os seguintes rótulos:
# 0: Não Empate
# 1: Empate

df_filtrado_multinomial8.loc[df_filtrado_multinomial8['predicao']==0,
                            'predicao_label'] ='Não Empate'
df_filtrado_multinomial8.loc[df_filtrado_multinomial8['predicao']==1,
                            'predicao_label'] ='Empate'


#%% Criação de tabela para cálculo da eficiência global do modelo

# Criando uma tabela para comparar as ocorrências reais com as predições
table = pd.pivot_table(df_filtrado_multinomial8,
                       index=['predicao'],
                       columns=['empate'],
                       aggfunc='size')
table

# Substituindo 'NaN' por zero
table = table.fillna(0)
table
#%% Eficiência global do modelo propriamente dita

# Transformando o dataframe 'table' para 'array', para que seja possível
#estabelecer o atributo 'diagonal'
table = table.to_numpy()
table

# Eficiência global do modelo
acuracia_modelo_empate = table.diagonal().sum()/table.sum()
acuracia_modelo_empate

#%%

pd.crosstab(
    df_filtrado_multinomial8['empate'],
    df_filtrado_multinomial8['predicao'],
    rownames=['Real'],
    colnames=['Previsto']
)
#%%

roc_auc = roc_auc_score(
    y,
    phats
)

print(roc_auc)

#%%
fpr, tpr, _ = roc_curve(y, phats)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr,
         label=f'ROC AUC = {roc_auc:.3f}')

plt.plot([0,1],[0,1],'--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC - Modelo de Empates')
plt.legend()
plt.show()

#%%
resultado_modelos = []
resultado_modelos.append({
    'Modelo': 'M1',
    'Variaveis': 'elo_mandante_antes + elo_visitante_antes',
    'LL': modelo_elo.llf,
    'Pseudo_R2': modelo_elo.prsquared,
    'Acuracia': acuracia_modelo_elo
})

resultado_modelos.append({
    'Modelo': 'M2',
    'Variaveis': 'diferenca_elo',
    'LL': modelo_elo2.llf,
    'Pseudo_R2': modelo_elo2.prsquared,
    'Acuracia': acuracia_modelo_elo2
})

resultado_modelos.append({
    'Modelo': 'M3',
    'Variaveis': 'abs_diferenca_elo',
    'LL': modelo_elo3.llf,
    'Pseudo_R2': modelo_elo3.prsquared,
    'Acuracia': acuracia_modelo_elo3
})

resultado_modelos.append({
    'Modelo': 'M4',
    'Variaveis': 'diferenca_elo + abs_diferenca_elo',
    'LL': modelo_elo4.llf,
    'Pseudo_R2': modelo_elo4.prsquared,
    'Acuracia': acuracia_modelo_elo4
})

resultado_modelos.append({
    'Modelo': 'M5',
    'Variaveis': 'diferenca_elo + diferenca_pontos_forma_ultimos5 + media_saldo_gols_ultimos5',
    'LL': modelo_elo5.llf,
    'Pseudo_R2': modelo_elo5.prsquared,
    'Acuracia': acuracia_modelo_elo5
})

resultado_modelos.append({
    'Modelo': 'M6',
    'Variaveis': 'diferenca_elo + diferenca_pontos_forma_ultimos5 + equilibrio_extremo_elo',
    'LL': modelo_elo6.llf,
    'Pseudo_R2': modelo_elo6.prsquared,
    'Acuracia': acuracia_modelo_elo6
})

resultado_modelos.append({
    'Modelo': 'M7',
    'Variaveis': 'diferenca_elo + diferenca_pontos_forma_ultimos5 + dummies_campeonato',
    'LL': modelo_elo7.llf,
    'Pseudo_R2': modelo_elo7.prsquared,
    'Acuracia': acuracia_modelo_elo7
})

resultado_modelos.append({
    'Modelo': 'M8',
    'Variaveis': 'Modelo Binário de Empate',
    'LL': modelo_empate.llf,
    'Pseudo_R2': getattr(modelo_empate, 'prsquared', np.nan),
    'Acuracia': acuracia_modelo_empate,
    'ROC_AUC': roc_auc
})

resultado_df = pd.DataFrame(resultado_modelos)

resultado_df = resultado_df.round({
    'LL': 2,
    'Pseudo_R2': 4,
    'Acuracia': 4
})

print(resultado_df)

#%%
tabela = pd.crosstab(
    df_filtrado['faixa_elo'],
    df_filtrado['resultado_final'],
    normalize='index'
)

#%%
pd.crosstab(
    df_filtrado_multinomial['faixa_elo'],
    df_filtrado_multinomial['empate']
)
chi2_contingency(
    pd.crosstab(
        df_filtrado_multinomial['faixa_elo'],
        df_filtrado_multinomial['empate']
    )
)
#%% Matriz de Confusão do Modelo 7

table = pd.pivot_table(
    df_filtrado_multinomial7,
    index=['predicao_label'],
    columns=['resultado_final2'],
    aggfunc='size'
)
cm = table.fillna(0)

cm = cm.reindex(
    index=['Vitória Visitante', 'Empate', 'Vitória Mandante'],
    columns=[0,1,2]
)

cm_pct = cm.div(cm.sum(axis=0), axis=1) * 100

labels = [
    [
        f'{int(cm.iloc[i,j])}\n({cm_pct.iloc[i,j]:.1f}%)'
        for j in range(cm.shape[1])
    ]
    for i in range(cm.shape[0])
]

plt.figure(figsize=(10,8))

sns.heatmap(
    cm_pct,
    annot=labels,
    fmt='',
    cmap='Blues',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label':'% da classe real'}
)

plt.title(
    'Matriz de Confusão - Modelo 7',
    fontsize=14
)

plt.xlabel('Resultado Real')
plt.ylabel('Resultado Previsto')

plt.xticks(
    [0.5,1.5,2.5],
    ['Vitória Visitante','Empate','Vitória Mandante'],
    rotation=0
)

plt.tight_layout()

plt.show()

#%% Inicio da comparação final entre modelos
# A partir de agora, dividirei o modelo em dataframes de treino e teste
# Pra isso, vou plotar os jogos por data, para identificar a data mais próxima onde
#terei uma divisão de 80/20 para o dataset

# Data no formato datetime
df_temporal = df_filtrado_multinomial.copy()
df_temporal['data'] = pd.to_datetime(df_temporal['data'])

# Ordenação cronológica
df_temporal = df_temporal.sort_values('data')

# Contagem de partidas por data
partidas_por_data = (
    df_temporal
    .groupby('data')
    .size()
    .reset_index(name='partidas')
)

# Número acumulado de partidas
partidas_por_data['partidas_acumuladas'] = (
    partidas_por_data['partidas'].cumsum()
)

# Percentual acumulado
partidas_por_data['percentual_acumulado'] = (
    partidas_por_data['partidas_acumuladas']
    / len(df_temporal)
    * 100
)

# Gráfico
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=partidas_por_data['data'],
        y=partidas_por_data['percentual_acumulado'],
        mode='lines',
        name='Partidas acumuladas',
        customdata=partidas_por_data[
            ['partidas_acumuladas', 'partidas']
        ],
        hovertemplate=(
            '<b>Data:</b> %{x|%d/%m/%Y}<br>'
            '<b>Partidas acumuladas:</b> %{customdata[0]}<br>'
            '<b>Partidas na data:</b> %{customdata[1]}<br>'
            '<b>Percentual acumulado:</b> %{y:.2f}%'
            '<extra></extra>'
        )
    )
)

# Linha de 2/3 da amostra
fig.add_hline(
    y=66.67,
    line_dash='dash',
    annotation_text='2/3 da amostra (66,67%)',
    annotation_position='top left'
)

# Linha de 80% da amostra
fig.add_hline(
    y=80,
    line_dash='dash',
    annotation_text='80% da amostra',
    annotation_position='bottom left'
)

# Layout
fig.update_layout(
    title='Distribuição temporal das partidas',
    xaxis_title='Data da partida',
    yaxis_title='Percentual acumulado de partidas (%)',
    hovermode='x unified',
    template='plotly_white',
    height=600
)

fig.show(renderer='browser')

#%% Preparação das variáveis para modelagem

# Garantindo o formato da data
df_filtrado_multinomial['data'] = pd.to_datetime(
    df_filtrado_multinomial['data'],
    dayfirst=True
)

# Ordenação cronológica
df_filtrado_multinomial = (
    df_filtrado_multinomial
    .sort_values('data')
    .reset_index(drop=True)
)

# Criação das dummies de campeonato
df_filtrado_multinomial = pd.get_dummies(
    df_filtrado_multinomial,
    columns=['campeonato'],
    drop_first=True,
    dtype=int
)
#%% Divisão temporal da base

data_corte = pd.Timestamp('2025-04-05')

df_treino = df_filtrado_multinomial[
    df_filtrado_multinomial['data'] < data_corte
].copy()

df_teste = df_filtrado_multinomial[
    df_filtrado_multinomial['data'] >= data_corte
].copy()

#%% Modelo M7 - Treino e Aaiação

# Definição das variáveis do M7


variaveis_m7 = [
    'diferenca_elo',
    'diferenca_pontos_forma_ultimos5',
    'campeonato_Espanhol',
    'campeonato_Francês',
    'campeonato_Inglês',
    'campeonato_Italiano'
]

variavel_alvo = 'resultado_final2'

# Seleção das variáveis


df_m7_treino = df_treino[
    variaveis_m7 + [variavel_alvo]
].copy()

df_m7_teste = df_teste[
    variaveis_m7 + [variavel_alvo]
].copy()

# Remoção de observações com valores ausentes

df_m7_treino = df_m7_treino.dropna()
df_m7_teste = df_m7_teste.dropna()

# Separação entre variáveis explicativas e variável resposta


X_treino = df_m7_treino[variaveis_m7].copy()
y_treino = df_m7_treino[variavel_alvo].astype(int)

X_teste = df_m7_teste[variaveis_m7].copy()
y_teste = df_m7_teste[variavel_alvo].astype(int)

# Inclusão da constante


X_treino = sm.add_constant(X_treino)
X_teste = sm.add_constant(X_teste)

# Estimação do modelo M7 utilizando somente o conjunto de treinamento


modelo_m7 = MNLogit(
    y_treino,
    X_treino
).fit()

print(modelo_m7.summary())


# Predição das probabilidades no conjunto de teste


probabilidades_m7_teste = modelo_m7.predict(X_teste)

# Conversão das probabilidades em classes previstas

predicoes_m7_teste = (
    np.asarray(probabilidades_m7_teste)
    .argmax(axis=1)
)


# Acurácia no conjunto de teste


acuracia_m7_teste = accuracy_score(
    y_teste,
    predicoes_m7_teste
)

print('\nAcurácia do M7 no conjunto de teste:')
print(
    f'{acuracia_m7_teste:.4f} '
    f'({acuracia_m7_teste * 100:.2f}%)'
)


# Matriz de confusão

matriz_confusao_m7 = confusion_matrix(
    y_teste,
    predicoes_m7_teste,
    labels=[0, 1, 2]
)

print('\nMatriz de confusão do M7:')
print(matriz_confusao_m7)



# Relatório de classificação


print('\nRelatório de classificação do M7:')

print(
    classification_report(
        y_teste,
        predicoes_m7_teste,
        labels=[0, 1, 2],
        target_names=[
            'Vitória Mandante',
            'Empate',
            'Vitória Visitante'
        ],
        digits=4,
        zero_division=0
    )
)

#%% Random Forest - Treino e avaliação

# 1. Variáveis utilizadas nos modelo

variaveis_modelo = [
    'diferenca_elo',
    'diferenca_pontos_forma_ultimos5',
    'campeonato_Espanhol',
    'campeonato_Francês',
    'campeonato_Inglês',
    'campeonato_Italiano'
]

variavel_alvo = 'resultado_final2'


# 2. Preparação dos dados

df_rf_treino = df_treino[
    variaveis_modelo + [variavel_alvo]
].dropna().copy()

df_rf_teste = df_teste[
    variaveis_modelo + [variavel_alvo]
].dropna().copy()


# Variáveis explicativas
X_rf_treino = df_rf_treino[variaveis_modelo].copy()
X_rf_teste = df_rf_teste[variaveis_modelo].copy()

# Variável resposta
y_rf_treino = df_rf_treino[variavel_alvo].astype(int)
y_rf_teste = df_rf_teste[variavel_alvo].astype(int)


# 3. Construção do Random Forest
# Usando o mesmo random_state que a professora Valquíria sugeriu
modelo_rf = RandomForestClassifier(
    random_state=42
)


# 4. Treinamento

modelo_rf.fit(
    X_rf_treino,
    y_rf_treino
)


# 5. Predição no conjunto de teste

predicoes_rf_teste = modelo_rf.predict(
    X_rf_teste
)

# 6. Acurácia

acuracia_rf = accuracy_score(
    y_rf_teste,
    predicoes_rf_teste
)

print('\nAcurácia do Random Forest no conjunto de teste:')
print(
    f'{acuracia_rf:.4f} '
    f'({acuracia_rf * 100:.2f}%)'
)


# 7. Matriz de confusão

matriz_confusao_rf = confusion_matrix(
    y_rf_teste,
    predicoes_rf_teste,
    labels=[0, 1, 2]
)

print('\nMatriz de confusão do Random Forest:')
print(matriz_confusao_rf)


# 8. Relatório de classificação


print('\nRelatório de classificação do Random Forest:')

print(
    classification_report(
        y_rf_teste,
        predicoes_rf_teste,
        labels=[0, 1, 2],
        target_names=[
            'Vitória Mandante',
            'Empate',
            'Vitória Visitante'
        ],
        digits=4,
        zero_division=0
    )
)
#%%
predicoes_rf_teste
#%% Matriz de Confusão do Modelo Random Forest

# Criar DataFrame temporário com valores reais e previstos
df_rf_conf = pd.DataFrame({
    'resultado_real': y_teste.astype(int),
    'predicao': predicoes_rf_teste.astype(int)
})

# Transformar os códigos das previsões em rótulos
df_rf_conf['predicao_label'] = df_rf_conf['predicao'].map({
    0: 'Vitória Visitante',
    1: 'Empate',
    2: 'Vitória Mandante'
})

# Tabela de frequência
table_rf = pd.pivot_table(
    df_rf_conf,
    index=['predicao_label'],
    columns=['resultado_real'],
    aggfunc='size'
)

cm_rf = table_rf.fillna(0)

# Garantir a mesma ordem das classes
cm_rf = cm_rf.reindex(
    index=['Vitória Visitante', 'Empate', 'Vitória Mandante'],
    columns=[0, 1, 2]
)

# Percentual dentro de cada classe real
cm_rf_pct = cm_rf.div(cm_rf.sum(axis=0), axis=1) * 100

# Criar rótulos com número absoluto + percentual
labels_rf = [
    [
        f'{int(cm_rf.iloc[i,j])}\n({cm_rf_pct.iloc[i,j]:.1f}%)'
        for j in range(cm_rf.shape[1])
    ]
    for i in range(cm_rf.shape[0])
]

# Criar heatmap
plt.figure(figsize=(10, 8))

sns.heatmap(
    cm_rf_pct,
    annot=labels_rf,
    fmt='',
    cmap='Blues',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': '% da classe real'}
)

plt.title(
    'Matriz de Confusão - Random Forest',
    fontsize=14
)

plt.xlabel('Resultado Real')
plt.ylabel('Resultado Previsto')

plt.xticks(
    [0.5, 1.5, 2.5],
    ['Vitória Visitante', 'Empate', 'Vitória Mandante'],
    rotation=0
)

plt.tight_layout()

plt.show()

#%% Otimização do Random Forest

# Variáveis utilizadas nos modelos serão as mesmas do modelo anterior

# Preparação das observações válidas

df_rf_desenvolvimento = df_treino[
    variaveis_modelo + [variavel_alvo]
].dropna().copy()


# Divisão da base de Treino: 80% treino / 20% validação

ponto_corte_rf = int(len(df_rf_desenvolvimento) * 0.80)

df_rf_treino = df_rf_desenvolvimento.iloc[
    :ponto_corte_rf
].copy()

df_rf_validacao = df_rf_desenvolvimento.iloc[
    ponto_corte_rf:
].copy()


# Separação entre variáveis explicativas e variável resposta

X_rf_treino = df_rf_treino[variaveis_modelo]
y_rf_treino = df_rf_treino[variavel_alvo].astype(int)

X_rf_validacao = df_rf_validacao[variaveis_modelo]
y_rf_validacao = df_rf_validacao[variavel_alvo].astype(int)


# Definição dos hiperparâmetros a serem testados

valores_n_estimators = [
    50,
    100,
    200,
    300,
    500
]

valores_min_samples_leaf = [
    1,
    2,
    5,
    10,
    20
]

#  Teste das combinações

resultados_rf = []

for n_estimators in valores_n_estimators:

    for min_samples_leaf in valores_min_samples_leaf:

        # Criação do modelo
        modelo_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        # Treinamento
        modelo_rf.fit(
            X_rf_treino,
            y_rf_treino
        )

        # Predição na validação
        predicoes = modelo_rf.predict(
            X_rf_validacao
        )

        # Métricas
        acuracia = accuracy_score(
            y_rf_validacao,
            predicoes
        )

        f1_macro = f1_score(
            y_rf_validacao,
            predicoes,
            average='macro'
        )

        f1_por_classe = f1_score(
            y_rf_validacao,
            predicoes,
            labels=[0, 1, 2],
            average=None
        )

        # Armazenamento
        resultados_rf.append({
            'n_estimators': n_estimators,
            'min_samples_leaf': min_samples_leaf,
            'acuracia': acuracia,
            'f1_macro': f1_macro,
            'f1_mandante': f1_por_classe[0],
            'f1_empate': f1_por_classe[1],
            'f1_visitante': f1_por_classe[2]
        })


# Tabela consolidada dos resultados

resultados_rf = pd.DataFrame(resultados_rf)

resultados_rf = resultados_rf.sort_values(
    'f1_macro',
    ascending=False
).reset_index(drop=True)


print('\nRESULTADOS DA OTIMIZAÇÃO')
print(resultados_rf)


# Melhores configurações segundo cada métrica

melhor_f1_macro = resultados_rf.loc[
    resultados_rf['f1_macro'].idxmax()
]

melhor_acuracia = resultados_rf.loc[
    resultados_rf['acuracia'].idxmax()
]

melhor_f1_empate = resultados_rf.loc[
    resultados_rf['f1_empate'].idxmax()
]


print('\nMELHOR CONFIGURAÇÃO POR F1 MACRO:')
print(melhor_f1_macro)

print('\nMELHOR CONFIGURAÇÃO POR ACURÁCIA:')
print(melhor_acuracia)

print('\nMELHOR CONFIGURAÇÃO POR F1 EMPATE:')
print(melhor_f1_empate)

#%%
resultados_rf[(resultados_rf['n_estimators']==100)&(resultados_rf['min_samples_leaf']==10)][['acuracia','f1_macro','f1_empate']]
#%% Heatmaps - Otimização do Random Forest

# Preparação das tabelas

tabela_acuracia = resultados_rf.pivot(
    index='min_samples_leaf',
    columns='n_estimators',
    values='acuracia'
)

tabela_f1_macro = resultados_rf.pivot(
    index='min_samples_leaf',
    columns='n_estimators',
    values='f1_macro'
)


#  Heatmap da Acurácia

plt.figure(figsize=(10, 6))

sns.heatmap(
    tabela_acuracia,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    linewidths=0.8,
    linecolor='white'
)

plt.title(
    'Acurácia na validação por hiperparâmetros',
    fontsize=16,
    pad=15
)

plt.xlabel(
    'Número de árvores (n_estimators)',
    fontsize=12
)

plt.ylabel(
    'Mínimo de observações por folha (min_samples_leaf)',
    fontsize=12
)

plt.tight_layout()
plt.show()


#  Heatmap do F1 Macro

plt.figure(figsize=(10, 6))

sns.heatmap(
    tabela_f1_macro,
    annot=True,
    fmt='.3f',
    cmap='RdBu_r',
    linewidths=0.8,
    linecolor='white'
)

plt.title(
    'F1 Macro na validação por hiperparâmetros',
    fontsize=16,
    pad=15
)

plt.xlabel(
    'Número de árvores (n_estimators)',
    fontsize=12
)

plt.ylabel(
    'Mínimo de observações por folha (min_samples_leaf)',
    fontsize=12
)

plt.tight_layout()
plt.show()

#%% Random Forest final - 100 árvores / min_samples_leaf = 10


#  Variáveis utilizadas

variaveis_modelo = [
    'diferenca_elo',
    'diferenca_pontos_forma_ultimos5',
    'campeonato_Espanhol',
    'campeonato_Francês',
    'campeonato_Inglês',
    'campeonato_Italiano'
]

variavel_alvo = 'resultado_final2'


#  Preparação do conjunto completo de treinamento

df_rf_treino_final = df_treino[
    variaveis_modelo + [variavel_alvo]
].dropna().copy()


#  Preparação do conjunto de teste

df_rf_teste_final = df_teste[
    variaveis_modelo + [variavel_alvo]
].dropna().copy()


#  Separação entre variáveis explicativas e variável resposta

X_rf_treino_final = df_rf_treino_final[
    variaveis_modelo
].copy()

y_rf_treino_final = df_rf_treino_final[
    variavel_alvo
].astype(int)


X_rf_teste_final = df_rf_teste_final[
    variaveis_modelo
].copy()

y_rf_teste_final = df_rf_teste_final[
    variavel_alvo
].astype(int)


#  Construção do Random Forest final

modelo_rf_final = RandomForestClassifier(
    n_estimators=100,
    min_samples_leaf=10,
    random_state=42
)


#  Treinamento utilizando todas as observações disponíveis

modelo_rf_final.fit(
    X_rf_treino_final,
    y_rf_treino_final
)


#  Predição no conjunto de teste

predicoes_rf_final = modelo_rf_final.predict(
    X_rf_teste_final
)


#  Acurácia

acuracia_rf_final = accuracy_score(
    y_rf_teste_final,
    predicoes_rf_final
)

print('ACURÁCIA DO RANDOM FOREST FINAL')
print(
    f'{acuracia_rf_final:.4f} '
    f'({acuracia_rf_final * 100:.2f}%)'
)


#  Matriz de confusão

matriz_confusao_rf_final = confusion_matrix(
    y_rf_teste_final,
    predicoes_rf_final,
    labels=[0, 1, 2]
)

print('\nMATRIZ DE CONFUSÃO')
print(matriz_confusao_rf_final)


#  Relatório de classificação

print('\nRELATÓRIO DE CLASSIFICAÇÃO')

print(
    classification_report(
        y_rf_teste_final,
        predicoes_rf_final,
        labels=[0, 1, 2],
        target_names=[
            'Vitória Mandante',
            'Empate',
            'Vitória Visitante'
        ],
        digits=4,
        zero_division=0
    )
)


#%% COMPARAÇÃO FINAL DOS MODELOS

# MODELOS COMPARADOS

# M7:
#   Regressão Logística Multinomial
#
# RF:
#   Random Forest original
#
# RF Otimizado:
#   100 árvores
#   min_samples_leaf = 10

# 1.1. RECRIAR O RANDOM FOREST PURO

# O objeto "modelo_rf" foi sobrescrito durante a otimização.

modelo_rf_puro = RandomForestClassifier(
    random_state=42
)


modelo_rf_puro.fit(
    X_rf_treino_final,
    y_rf_treino_final
)


#%%  1.2. PROBABILIDADES DOS TRÊS MODELOS

# ---------------------------------------------------------
# M7 — Regressão Logística Multinomial
# ---------------------------------------------------------

prob_m7 = np.asarray(
    modelo_m7.predict(X_teste)
)


# ---------------------------------------------------------
# RF Puro
# ---------------------------------------------------------

prob_rf = (
    modelo_rf_puro
    .predict_proba(X_rf_teste_final)
)


# ---------------------------------------------------------
# RF Otimizado
# ---------------------------------------------------------

prob_rf_otimizado = (
    modelo_rf_final
    .predict_proba(X_rf_teste_final)
)


#%%  1.3. CLASSES PREVISTAS

pred_m7 = np.argmax(
    prob_m7,
    axis=1
)

pred_rf = (
    modelo_rf_puro
    .predict(X_rf_teste_final)
)

pred_rf_otimizado = (
    modelo_rf_final
    .predict(X_rf_teste_final)
)


# Variável resposta
y_comparacao = y_teste.astype(int)


#%% 1.4. MÉTRICAS

acuracia_m7 = accuracy_score(
    y_comparacao,
    pred_m7
)

acuracia_rf = accuracy_score(
    y_comparacao,
    pred_rf
)

acuracia_rf_otimizado = accuracy_score(
    y_comparacao,
    pred_rf_otimizado
)


f1_m7 = f1_score(
    y_comparacao,
    pred_m7,
    average='macro'
)

f1_rf = f1_score(
    y_comparacao,
    pred_rf,
    average='macro'
)

f1_rf_otimizado = f1_score(
    y_comparacao,
    pred_rf_otimizado,
    average='macro'
)



#%% 1.6. TABELA FINAL DE MÉTRICAS

comparacao_modelos = pd.DataFrame({

    'Modelo': [
        'Logística Multinomial',
        'Random Forest',
        'Random Forest Otimizado'
    ],

    'Acurácia': [
        acuracia_m7,
        acuracia_rf,
        acuracia_rf_otimizado
    ],

    'F1 Macro': [
        f1_m7,
        f1_rf,
        f1_rf_otimizado
    ]}
)


print('COMPARAÇÃO FINAL DOS MODELOS')

print(
    comparacao_modelos.round(4)
)

#%% CURVAS ROC — COMPARAÇÃO DOS MODELOS

classes = [0, 1, 2]

nomes_classes = [
    'Vitória Mandante',
    'Empate',
    'Vitória Visitante'
]

cores_classes = [
    'indigo',
    'darkgreen',
    'darkorange'
]


# Probabilidades dos modelos

probabilidades_modelos = {
    'Logística Multinomial': prob_m7,
    'Random Forest': prob_rf,
    'Random Forest Otimizado': prob_rf_otimizado
}


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)


for ax, classe, nome_classe, cor in zip(
    axes,
    classes,
    nomes_classes,
    cores_classes
):

    for nome_modelo, probabilidades in (
        probabilidades_modelos.items()
    ):

        # Variável binária:
        # classe atual x todas as outras

        y_binario = (
            y_comparacao == classe
        ).astype(int)


        fpr, tpr, _ = roc_curve(
            y_binario,
            probabilidades[:, classe]
        )


        roc_auc = auc(
            fpr,
            tpr
        )


        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=(
                f'{nome_modelo} '
                f'(AUC = {roc_auc:.3f})'
            )
        )


    # Linha aleatória

    ax.plot(
        [0, 1],
        [0, 1],
        '--',
        color='gray',
        linewidth=1
    )


    ax.set_title(
        nome_classe,
        fontsize=14
    )

    ax.set_xlabel(
        'Chance de FP',
        fontsize=11
    )

    ax.set_ylabel(
        'Chance de TP',
        fontsize=11
    )

    ax.legend(
        fontsize=9,
        loc='lower right'
    )

    ax.grid(
        alpha=0.3
    )


fig.suptitle(
    'Curvas ROC — Comparação entre os modelos',
    fontsize=16,
    y=1.02
)

plt.tight_layout()

plt.show()

#%%  COMPARAÇÃO DAS MÉTRICAS
# Acurácia | Log-Likelihood | F1 Macro

fig, axes = plt.subplots(
    1,
    2,
    figsize=(18, 6)
)


# 3.1. ACURÁCIA

axes[0].barh(
    comparacao_modelos['Modelo'],
    comparacao_modelos['Acurácia']
)

axes[0].set_title(
    'Acurácia',
    fontsize=14
)

axes[0].set_xlabel(
    'Acurácia'
)

axes[0].set_xlim(
    0,
    1
)


for i, valor in enumerate(
    comparacao_modelos['Acurácia']
):

    axes[0].text(
        valor + 0.01,
        i,
        f'{valor:.3f}',
        va='center'
    )



# F1 MACRO

axes[1].barh(
    comparacao_modelos['Modelo'],
    comparacao_modelos['F1 Macro']
)

axes[1].set_title(
    'F1 Macro',
    fontsize=14
)

axes[1].set_xlabel(
    'F1 Macro'
)

axes[1].set_xlim(
    0,
    1
)


for i, valor in enumerate(
    comparacao_modelos['F1 Macro']
):

    axes[1].text(
        valor + 0.01,
        i,
        f'{valor:.3f}',
        va='center'
    )




fig.suptitle(
    'Comparação de desempenho dos modelos',
    fontsize=16,
    y=1.02
)

plt.tight_layout()

plt.show()

#%% MATRIZES DE CONFUSÃO

nomes_modelos = [
    'Logística Multinomial',
    'Random Forest',
    'Random Forest Otimizado'
]

predicoes_modelos = [
    pred_m7,
    pred_rf,
    pred_rf_otimizado
]

nomes_classes_cm = [
    'Mandante',
    'Empate',
    'Visitante'
]


fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 5.5)
)


for ax, nome_modelo, predicoes in zip(
    axes,
    nomes_modelos,
    predicoes_modelos
):

    cm = confusion_matrix(
        y_comparacao,
        predicoes,
        labels=[0, 1, 2]
    )


    # Percentual dentro de cada classe real

    cm_percentual = (
        cm /
        cm.sum(axis=1, keepdims=True)
        * 100
    )


    anotacoes = []

    for i in range(3):

        linha = []

        for j in range(3):

            linha.append(
                f'{cm[i, j]}\n'
                f'({cm_percentual[i, j]:.1f}%)'
            )

        anotacoes.append(linha)


    sns.heatmap(
        cm_percentual,
        annot=anotacoes,
        fmt='',
        cmap='Blues',
        linewidths=0.5,
        linecolor='white',
        cbar=False,
        ax=ax
    )


    ax.set_title(
        nome_modelo,
        fontsize=13
    )

    ax.set_xlabel(
        'Resultado Previsto'
    )

    ax.set_ylabel(
        'Resultado Real'
    )

    ax.set_xticklabels(
        nomes_classes_cm,
        rotation=0
    )

    ax.set_yticklabels(
        nomes_classes_cm,
        rotation=0
    )


fig.suptitle(
    'Matrizes de Confusão — Comparação dos modelos',
    fontsize=16,
    y=1.02
)

plt.tight_layout()

plt.show()

#%% F1-SCORE POR CLASSE

from sklearn.metrics import f1_score


f1_modelos = []


for nome_modelo, predicoes in zip(
    nomes_modelos,
    predicoes_modelos
):

    f1_classes = f1_score(
        y_comparacao,
        predicoes,
        labels=[0, 1, 2],
        average=None
    )


    for classe, f1 in zip(
        nomes_classes,
        f1_classes
    ):

        f1_modelos.append({

            'Modelo': nome_modelo,

            'Classe': classe,

            'F1': f1

        })


df_f1_classes = pd.DataFrame(
    f1_modelos
)

tabela_f1 = df_f1_classes.pivot(
    index='Modelo',
    columns='Classe',
    values='F1'
)


ax = tabela_f1.plot(
    kind='barh',
    figsize=(12, 6)
)


plt.title(
    'F1-Score por classe — Comparação dos modelos',
    fontsize=15
)

plt.xlabel(
    'F1-Score'
)

plt.ylabel(
    'Modelo'
)

plt.xlim(
    0,
    1
)

plt.legend(
    title='Resultado'
)

plt.grid(
    axis='x',
    alpha=0.3
)

for container in ax.containers:

    ax.bar_label(
        container,
        fmt='%.3f',
        padding=4,
        fontsize=10
    )


plt.tight_layout()

plt.show()