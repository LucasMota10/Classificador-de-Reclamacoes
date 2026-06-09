from builder.builder_knn_transformers import ModeloKNN_Transformers 
from builder.builder_naive import ModeloNaiveBayes
from builder.builder_knn_tf_idf import ModeloKNN_TFIDF


if __name__ == "__main__":
    
    # Definição dos modelos de classificação:

    naive_bayes = ModeloNaiveBayes()
    knn_tfidf = ModeloKNN_TFIDF(k=15)
    knn_transformers = ModeloKNN_Transformers(k=15)

    # 1 - Naive Bayes com TF-IDF
    
    print("\n--- Classificação com Naive Bayes ---")
    
    df_dados = naive_bayes.carregar_dados()
    X, y = naive_bayes.preparar_dados(df_dados)
    X_test, y_test = naive_bayes.treinar_modelo(X, y)
    naive_bayes.avaliar_modelo(X_test, y_test)
    naive_bayes.plotar_matriz_confusao_geral(X_test, y_test)
    naive_bayes.imprimir_vocabulario()

    print("\n--- Classificação com KNN + TF-IDF ---")

    df_dados = knn_tfidf.carregar_dados()
    
    X, y = knn_tfidf.preparar_dados(df_dados)
    X_test, y_test = knn_tfidf.treinar_modelo(X, y)
    
    knn_tfidf.avaliar_modelo(X_test, y_test)
    knn_tfidf.plotar_matriz_confusao_geral(X_test, y_test)

    print("\n--- Classificação com KNN + Transformers ---")

    df_dados = knn_transformers.carregar_dados()
    X, y = knn_transformers.preparar_dados(df_dados)
    X_test, y_test = knn_transformers.treinar_modelo(X, y)
    
    knn_transformers.avaliar_modelo(X_test, y_test)

    knn_transformers.plotar_matriz_confusao_geral(X_test, y_test)
    knn_transformers.visualize_embeddings(X_test, y_test)