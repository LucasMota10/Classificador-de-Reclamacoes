import pandas as pd
import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE



class ClassificadorReclamacao:
    def __init__(self, k: int = 5) -> None:
        '''
        Inicializa o classificador instanciando o modelo encoder e o KNN.
        O Modelo encoder transforma as reclamações em vetores com 384 dimensões, 
        e o KNN permite inferir a área da reclamação com base nos k vizinhos mais próximos.

        :param k: Número de vizinhos mais próximos a serem considerados pelo KNN.
        :return:
        '''
        self.modelo_encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        self.knn = KNeighborsClassifier(n_neighbors=k, metric='cosine')
    
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

    def treinar_modelo(self, x: list[str], y: list[str]) -> tuple[np.ndarray, list[str]]:
        '''
        Vetoriza os textos das reclamações, divide em treino/teste e treina o modelo KNN.

        :param x: Lista de strings contendo os relatos dos consumidores.
        :param y: Lista contendo os rótulos (áreas) correspondentes.
        :return: Uma tupla com os vetores de teste (X_test) e os rótulos de teste (y_test) para avaliação.
        '''

        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.2,
            random_state=2,
            stratify=y
        )

        X_train = self.modelo_encoder.encode(
            X_train_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        X_test = self.modelo_encoder.encode(
            X_test_texts,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        self.knn.fit(X_train, y_train)

        return X_test, y_test

    def avaliar_modelo(self, X_test: np.ndarray, y_test: list[str]) -> None:
        '''
        Realiza as previsões com os dados de teste separados e exibe as métricas de performance.

        :param X_test: Array NumPy contendo os vetores das reclamações separadas para teste.
        :param y_test: Lista contendo os rótulos reais dos dados de teste.
        :return:
        '''
        previsoes = self.knn.predict(X_test)
        
        print(f"\nRelatório de Classificação para k = {self.knn.n_neighbors}:\n")
        print(classification_report(y_test, previsoes))

    def visualize_embeddings(self, X_vetores: np.ndarray, y_labels: list[str]) -> None:
        '''
        Reduz a dimensionalidade dos vetores para 2D usando t-SNE e plota um gráfico de dispersão.

        :param X_vetores: Array NumPy com os vetores originais de 384 dimensões.
        :param y_labels: Lista com as áreas (rótulos) correspondentes a cada vetor.
        :return:
        '''

        tsne = TSNE(n_components=2, random_state=23, metric='cosine')
        vetores_2d = tsne.fit_transform(X_vetores)

        df_plot = pd.DataFrame({
            'Eixo X': vetores_2d[:, 0],
            'Eixo Y': vetores_2d[:, 1],
            'Área da Reclamação': y_labels
        })

        plt.figure(figsize=(14, 8))
        sns.set_theme(style="whitegrid")

        sns.scatterplot(
            data=df_plot, 
            x='Eixo X', 
            y='Eixo Y', 
            hue='Área da Reclamação',
            palette='tab20',          
            alpha=0.8,                
            s=60                     
        )

        plt.title('Mapa Semântico das Reclamações', fontsize=16)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    
    classificador = ClassificadorReclamacao(k=20)
    df_dados = classificador.carregar_dados()
    X, y = classificador.preparar_dados(df_dados)
    X_test, y_test = classificador.treinar_modelo(X, y)
    classificador.avaliar_modelo(X_test, y_test)
    classificador.visualize_embeddings(X_test, y_test)