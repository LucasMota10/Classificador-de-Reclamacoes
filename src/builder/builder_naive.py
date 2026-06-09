import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.sparse
import nltk

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix

class ModeloNaiveBayes:
    def __init__(self, max_features: int | None = None) -> None:
        '''
        Inicializa o classificador instanciando o vetorizador TF-IDF e o Naive Bayes.
        O TF-IDF transforma as reclamações em matrizes de frequência de palavras,
        e o Multinomial Naive Bayes calcula a probabilidade de cada área baseada nessas palavras.

        :param max_features: Número máximo de palavras no vocabulário do TF-IDF.
        :return: Nenhum retorno.
        '''

        nltk.download('stopwords', quiet=True)
        stop_words_pt = stopwords.words('portuguese')
        padrao_so_letras = r'(?u)\b[a-zA-ZáéíóúâêôãõçÁÉÍÓÚÂÊÔÃÕÇ]{2,}\b'

        self.vetorizador = TfidfVectorizer(
            stop_words=stop_words_pt,
            max_features=max_features,
            token_pattern=padrao_so_letras
        )
        self.nb = MultinomialNB()
    
    def carregar_dados(self, caminho: str = '../data/*.json') -> pd.DataFrame:
        '''
        Carrega os dados de múltiplos arquivos JSON encontrados no caminho especificado.

        :param caminho: Caminho em formato de string para os arquivos JSON.
        :return: DataFrame do Pandas contendo os dados de todas as reclamações.
        '''
        reclamacoes = glob.glob(caminho)
        dados_completos = []

        for arquivo in reclamacoes:
            with open(arquivo, 'r', encoding='utf-8') as f:
                dados_completos.extend(json.load(f))

        return pd.DataFrame(dados_completos)

    def preparar_dados(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        '''
        Separa o DataFrame nas variáveis de entrada (textos) e na variável alvo (áreas).

        :param df: DataFrame contendo as colunas 'relato' e 'area'.
        :return: Uma tupla onde o primeiro elemento é a lista de textos e o segundo é a lista com as áreas.
        '''
        x = df['relato'].tolist()
        y = df['area'].tolist()
        
        return x, y

    def treinar_modelo(self, x: list[str], y: list[str]) -> tuple[scipy.sparse.csr_matrix, list[str]]:
        '''
        Divide os dados em treino/teste, vetoriza os textos via TF-IDF e treina o modelo Naive Bayes.
        O TF-IDF aprende o vocabulário estritamente com os dados de treino para evitar data leakage.

        :param x: Lista de strings contendo os relatos dos consumidores.
        :param y: Lista contendo os rótulos (áreas) correspondentes.
        :return: Uma tupla com a matriz de teste (X_test) e os rótulos de teste (y_test) para avaliação.
        '''
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=2,
            stratify=y
        )

        X_train = self.vetorizador.fit_transform(X_train_texts)
        
        X_test = self.vetorizador.transform(X_test_texts)

        self.nb.fit(X_train, y_train)

        return X_test, y_test

    def avaliar_modelo(self, X_test: scipy.sparse.csr_matrix, y_test: list[str]) -> None:
        '''
        Realiza as previsões com os dados de teste separados e exibe as métricas de performance.

        :param X_test: Matriz esparsa contendo os vetores TF-IDF de teste.
        :param y_test: Lista contendo os rótulos reais dos dados de teste.
        :return: Nenhum retorno.
        '''
        previsoes = self.nb.predict(X_test)
        
        print("\nRelatório de Classificação do Naive Bayes:\n")
        print(classification_report(y_test, previsoes))

    def imprimir_vocabulario(self) -> None:
        '''
        Extrai e imprime informações sobre o vocabulário aprendido pelo TF-IDF.
        O modelo já deve ter sido treinado (método treinar_modelo) antes de chamar esta função.

        :param:
        :return:
        '''
        vocabulario = self.vetorizador.get_feature_names_out()
        tamanho = len(vocabulario)

        print(f"\n--- ANÁLISE DO VOCABULÁRIO ---")
        print(f"Total de palavras únicas aprendidas: {tamanho}")
        
        if tamanho > 0:
            print("\nPrimeiras 20 palavras:")
            print(vocabulario[:20])
            
            print("\nÚltimas 20 palavras:")
            print(vocabulario[-20:])
            
            meio = tamanho // 2
            print("\nAmostra de 20 palavras do meio do vocabulário:")
            print(vocabulario[meio : meio + 20])
        else:
            print("O vocabulário está vazio")

    def plotar_matriz_confusao_geral(self, X_test: scipy.sparse.csr_matrix, y_test: list[str]) -> None:
        '''
        Gera e plota a Matriz de Confusão geral (12x12) como um Heatmap para visualizar os acertos e erros cruzados.

        :param X_test: Matriz esparsa contendo os vetores TF-IDF de teste.
        :param y_test: Lista contendo os rótulos reais dos dados de teste.
        :return: Nenhum retorno. Exibe o gráfico na tela.
        '''
        previsoes = self.nb.predict(X_test)
        classes_unicas = sorted(list(set(y_test)))
        matriz = confusion_matrix(y_test, previsoes, labels=classes_unicas)

        plt.figure(figsize=(12, 10))
        sns.set_theme(style="white")

        sns.heatmap(
            matriz, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes_unicas, yticklabels=classes_unicas, cbar=False
        )

        plt.title('Matriz de Confusão Geral (12x12) - Naive Bayes', fontsize=18, pad=20)
        plt.ylabel('Realidade (Área Correta)', fontsize=14, fontweight='bold')
        plt.xlabel('Previsão (Chute do Modelo)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    classificador_nb = ModeloNaiveBayes()
    df_dados = classificador_nb.carregar_dados()
    X, y = classificador_nb.preparar_dados(df_dados)
    X_test, y_test = classificador_nb.treinar_modelo(X, y)
    classificador_nb.avaliar_modelo(X_test, y_test)
    classificador_nb.plotar_matriz_confusao_geral(X_test, y_test)
    classificador_nb.imprimir_vocabulario()