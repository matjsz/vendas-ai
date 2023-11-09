import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

# Exemplo de dados fictícios:
dados = pd.DataFrame({
    'mensagem': [
        "Endereço: Rua Exemplo, 123",
        "Adicionar produto 23423423",
        "Mensagem genérica sem rótulo",
        "Endereço de entrega: Avenida Principal, 456",
        "Entregar em: Rua Tal, 456, CEP 23145",
        "Mensagem aleatória",
        "Adicionar produto Tal no Pedido DFG456",
        "Entrega para Avenida Otório 543 CEP 123545",
        "Faturar produto HIJ345 junto do produto ACE324",
        "Rua Aquela, 34, CEP 1233243",
        "Avenida Essa, 120, Cidade - ET"
        # ... mais mensagens ...
    ]
})

# Vetorização das mensagens usando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dados['mensagem'])

# Aplicação do algoritmo de k-means
n_clusters = 3  # Número de clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
dados['cluster'] = kmeans.fit_predict(X)

# Examinando os clusters
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(dados[dados['cluster'] == cluster]['mensagem'].values)
    print("\n")

# Etapa opcional: Se desejar, você pode atribuir rótulos aos clusters manualmente
# rótulos_manual = ["Endereço", "Requisição", "Outro"]
# dados['rotulo'] = rótulos_manual

# Treinando um modelo de classificação para prever rótulos
# Esta parte é opcional, você pode ajustar os rótulos manualmente se preferir.
if 'rotulo' not in dados:
    dados['rotulo'] = LabelEncoder().fit_transform(dados['cluster'])

# Dividindo os dados para treinamento do modelo de classificação
X_train, X_test, y_train, y_test = train_test_split(X.toarray(), dados['rotulo'], test_size=0.2, random_state=42)

# Criando e treinando um modelo de classificação simples (usando TensorFlow aqui)
modelo_classificacao = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(n_clusters, activation='softmax')
])

modelo_classificacao.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
modelo_classificacao.fit(X_train, y_train, epochs=10, validation_split=0.2)

# Avaliando o modelo de classificação
resultado_classificacao = modelo_classificacao.evaluate(X_test, y_test)
print(f'Acurácia do modelo de classificação: {resultado_classificacao[1]*100:.2f}%')

# Fazendo previsões com novos dados
novas_mensagens = [
                   "Endereço de entrega: Rua Nova, 789",
                   "Incluir produto 2343953",
                   "Entregar em Rua Nome, 12, Cidade - ET"
                   # ... mais mensagens ...
                  ]

# Vetorizando as novas mensagens
novos_dados = vectorizer.transform(novas_mensagens).toarray()

# Fazendo previsões usando o modelo de classificação
previsoes = modelo_classificacao.predict(novos_dados)
previsoes_classes = [cluster for cluster in previsoes.argmax(axis=1)]

# Adicionando as novas mensagens ao DataFrame
novos_dados_df = pd.DataFrame({'mensagem': novas_mensagens, 'cluster': previsoes_classes})

# Exibindo as previsões para as novas mensagens
print("\nPrevisões para Novas Mensagens:")
print(novos_dados_df)

# Examinando os clusters das novas mensagens
for cluster in range(n_clusters):
    print(f"Cluster {cluster}:")
    print(novos_dados_df[novos_dados_df['cluster'] == cluster]['mensagem'].values)
    print("\n")
