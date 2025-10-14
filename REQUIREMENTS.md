# Relatório de Elicitação de Requisitos - Avaliação A3

## Introdução

Este documento detalha o processo de elicitação de requisitos. O objetivo desta fase é compreender as necessidades e os desafios do nosso público-alvo, para que possamos construir uma ferramenta que agregue valor real.

Para esta análise, empregamos uma combinação de duas técnicas ágeis: **Mapeamento da Jornada do Usuário** e **Brainstorming**. A primeira nos ajudou a identificar os problemas (dores), e a segunda, a gerar soluções (funcionalidades).

---

## Técnica 1: Mapeamento da Jornada do Usuário (User Journey Mapping)

### Descrição do Processo

Para guiar o desenvolvimento, utilizamos a técnica de Mapeamento da Jornada do Usuário, que se baseia na criação de cenários para visualizar a experiência do usuário de ponta a ponta. Este método nos permitiu mapear as ações, emoções e frustrações de um usuário típico ao interagir com um problema que nossa ferramenta se propõe a resolver, revelando oportunidades claras para a criação de funcionalidades de alto valor.

### Perfil da Persona

Para tornar a jornada concreta, criamos uma persona que representa nosso público-alvo principal.

- **Nome:** Ana Costa
- **Idade:** 22 anos
- **Ocupação:** Estudante universitária de Economia.
- **Objetivo:** Coletar, visualizar e comparar dados socioeconômicos (PIB, uso de internet, etc.) de diferentes países para sua tese. Ela também deseja gerar uma previsão simples para fortalecer seus argumentos, mas não possui conhecimento avançado em Machine Learning.
- **Frustrações:** Perde muito tempo navegando em portais de dados governamentais, baixando e limpando planilhas. Acha o processo de treinar modelos de ML intimidante e tem dificuldade para interpretar suas predições.

### Cenário (Objetivo da Jornada)

Ana precisa comparar a evolução do PIB e do percentual de uso da internet entre Brasil e Canadá para sua tese. Além disso, ela quer gerar uma previsão do PIB para o próximo ano para incluir em sua análise de tendências.

### Evidência (Mapa da Jornada)

O mapa abaixo representa a jornada da Ana. As oportunidades identificadas foram a matéria-prima para a sessão de brainstorming.


| Etapas da Jornada                        | Descoberta e Acesso                                                            | Seleção e Visualização de Dados                                                                                | Treinamento do Modelo                                                                                              | Análise dos Resultados                                                                                             |
| :--------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| **História**                             | Ana ouve falar de uma nova ferramenta para análise de dados e decide testá-la. | Com a ferramenta aberta, Ana seleciona os países e os indicadores que precisa para sua pesquisa.               | Curiosa com a função de previsão, Ana decide treinar um modelo para estimar o PIB do próximo ano.                  | O aplicativo exibe a previsão e as métricas de performance do modelo treinado.                                     |
| **Ações**                                | Acessa o link do aplicativo.                                                   | 1. Seleciona "Brasil" e "Canadá".<br>2. Escolhe os indicadores (PIB, etc.).<br>3. Observa os gráficos gerados. | 1. Navega para a aba de ML.<br>2. Escolhe um modelo (ex: Regressão Linear).<br>3. Clica no botão "Treinar Modelo". | 1. Lê o valor da predição.<br>2. Olha as métricas (MAE, R², etc.).<br>3. Analisa o gráfico de "Previsto vs. Real". |
| **Pontos de Contato**                    | Página inicial do Streamlit.                                                   | Sidebar de configuração e a aba de "Exploração de Dados".                                                      | Sidebar e o botão de treino na aba de "Modelagem".                                                                 | Os containers de resultados na aba de "Modelagem".                                                                 |
| **Emoções**                              | 🤔 Curiosa                                                                     | 😊 Satisfeita                                                                                                  | 😬 Apreensiva                                                                                                      | 🤯 Confusa / 😄 Impressionada                                                                                      |
| **Pontos de Dor**                        | "Será que é confiável? De onde vêm os dados?"                                  | "Gostaria de comparar dois indicadores no mesmo gráfico."                                                      | "Qual modelo eu escolho? Não entendo a diferença entre eles."                                                      | "O que significa 'R² = 0.85'? Como cada dado impacta na predição do modelo?"                                                                 |
| **Oportunidades (Ações nos Bastidores)** | Exibir a fonte dos dados (Banco Mundial) e a data da última atualização.       | Criar um gráfico comparativo com múltiplos eixos.                                                              | Adicionar _tooltips_ ou textos de ajuda explicando cada modelo de forma simples.                                   | Apresentar métricas com textos explicativos e incluir técnicas de explicabilidade.                                 |

---

## Técnica 2: Brainstorming

### Descrição do Processo

Após mapear a jornada e identificar as dores da Ana, realizamos uma sessão de brainstorming para gerar ideias de funcionalidades. A sessão foi focada na seguinte pergunta-guia: **"Como podemos transformar as dores da Ana (complexidade, falta de confiança e dificuldade de interpretação) em funcionalidades que tornem nossa ferramenta poderosa, intuitiva e confiável?"**. As ideias foram geradas e depois agrupadas em temas, que se tornarão nossos Épicos.

### Evidência (Resultado do Brainstorming)

A estrutura abaixo representa o resultado da nossa sessão de brainstorming, com as ideias clusterizadas.

#### Tema 1: Análise e Visualização de Dados (Feature da Issue #1)

- **Ideias:**
  - Permitir a seleção de múltiplos países para comparação lado a lado.
  - Permitir a plotagem de dois indicadores diferentes no mesmo gráfico, com eixos Y distintos.
  - Adicionar um botão para "Exportar Gráfico como PNG".
  - Exibir a fonte dos dados e a data da última atualização de forma proeminente.
  - Adicionar um seletor de escala para os gráficos (Linear vs. Log).

#### Tema 2: Machine Learning Descomplicado (Feature da Issue #2)

- **Ideias:**
  - Adicionar um ícone de ajuda `(?)` ao lado de cada modelo com uma explicação simples do seu funcionamento.
  - Além das métricas, mostrar uma interpretação textual da performance (ex: "Este modelo teve uma boa aderência aos dados de teste.").
  - Exibir um gráfico de "Importância das Features" para mostrar o que mais influenciou a predição.
  - Permitir que o usuário ajuste a porcentagem de divisão entre treino e teste (ex: 80/20, 70/30).

#### Tema 3: Dashboard e Usabilidade (Feature da Issue #3)

- **Ideias:**
  - Criar uma aba/seção de "Relatório" que resume todas as seleções e resultados para fácil captura de tela.
  - Adicionar um botão para "Exportar dados da tabela como CSV".
  - Implementar um "Modo de Apresentação" que esconde os menus e deixa apenas os gráficos e resultados visíveis.
  - Guardar a última seleção do usuário (país, modelo) no cache do navegador para a próxima visita.

## Conclusão da Elicitação

A combinação das técnicas de Mapeamento da Jornada do Usuário e Brainstorming se mostrou extremamente eficaz. Conseguimos partir de um cenário de uso realista, identificar frustrações concretas e traduzi-las em um conjunto rico de ideias para funcionalidades. Este processo garante que nosso backlog não seja apenas uma lista de tarefas técnicas, mas sim um plano de ação orientado a gerar valor para nossa persona, Ana. As ideias agrupadas por temas servirão como base para a criação dos Épicos e Histórias de Usuário na próxima etapa do projeto.

---
