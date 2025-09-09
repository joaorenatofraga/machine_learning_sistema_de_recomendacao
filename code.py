import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- 0. Configurações Globais ---
# Definir o número de usuários e produtos para a simulação
NUM_USERS = 500
NUM_PRODUCTS = 100
NUM_PURCHASES = 10000

print("--- Iniciando Projeto: Sistema de Recomendação de Produtos ---")
print(f"Configurações: {NUM_USERS} usuários, {NUM_PRODUCTS} produtos, {NUM_PURCHASES} compras simuladas.\n")

# --- 1. ETAPA: Criação e Preparação de Dados Fictícios ---
print("1. Criando e preparando dados de vendas fictícios...")

# 1.1. Gerar dados de compras
# Gerar compras aleatórias para simular o comportamento de usuários
np.random.seed(42)  # Para reprodutibilidade
data = {
    'user_id': np.random.randint(1, NUM_USERS + 1, size=NUM_PURCHASES),
    'product_id': np.random.randint(1, NUM_PRODUCTS + 1, size=NUM_PURCHASES),
    'rating': np.random.uniform(3.0, 5.0, size=NUM_PURCHASES).round(1) # Rating de 3 a 5
}
purchases_df = pd.DataFrame(data)

# Agrupar compras para obter uma contagem de interações por user-product
# Isso é mais realista do que ratings diretos para um cenário de e-commerce
interaction_counts = purchases_df.groupby(['user_id', 'product_id']).size().reset_index(name='purchase_count')
print("  - DataFrame de interações criado (contagem de compras por usuário/produto).")

# 1.2. Criar a Matriz de Usuário-Item
# Pivotar a tabela para que as linhas sejam os usuários e as colunas os produtos
# Os valores são a contagem de compras. Valores nulos significam que o usuário não comprou o produto.
user_item_matrix = interaction_counts.pivot_table(index='user_id', columns='product_id', values='purchase_count').fillna(0)
print("  - Matriz de Usuário-Item criada com sucesso.")
print(f"  - Dimensões da matriz (usuários x produtos): {user_item_matrix.shape}")
print("  - Exemplo da matriz (cabeçalho):\n", user_item_matrix.head())

# --- 2. ETAPA: Cálculo da Similaridade entre Usuários ---
print("\n2. Calculando a similaridade entre usuários...")

# Calcular a similaridade de cosseno entre cada usuário
# Isso resulta em uma matriz onde cada valor (i, j) é a similaridade entre o usuário i e o usuário j.
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
print("  - Matriz de similaridade de cosseno calculada.")
print(f"  - Dimensões da matriz de similaridade: {user_similarity_df.shape}")
print("  - Exemplo da matriz de similaridade (cabeçalho):\n", user_similarity_df.head())

# --- 3. ETAPA: Geração de Recomendações ---
print("\n3. Gerando recomendações para um usuário de teste...")

def get_recommendations(user_id, num_recommendations=5, num_similar_users=10):
    """
    Função para gerar recomendações para um usuário específico.
    Args:
        user_id (int): O ID do usuário para o qual gerar recomendações.
        num_recommendations (int): O número de produtos a serem recomendados.
        num_similar_users (int): O número de usuários mais similares a considerar.
    """
    print(f"  - Gerando recomendações para o usuário {user_id}...")
    
    # Obter os usuários mais similares
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    # Remover o próprio usuário da lista
    similar_users = similar_users.drop(user_id)
    # Selecionar os N usuários mais similares
    top_similar_users = similar_users.head(num_similar_users).index.tolist()
    
    # Identificar os produtos que o usuário de teste já comprou
    products_bought = user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist()
    
    # Coletar os produtos comprados pelos usuários similares
    similar_users_products = user_item_matrix.loc[top_similar_users]
    
    # Encontrar os produtos mais comprados por esses usuários similares
    recommendations = similar_users_products.sum(axis=0).sort_values(ascending=False)
    
    # Filtrar os produtos que o usuário de teste já comprou
    recommendations = recommendations.drop(products_bought, errors='ignore')
    
    # Retornar as N melhores recomendações
    final_recommendations = recommendations.head(num_recommendations).index.tolist()
    
    if not final_recommendations:
        print(f"  - Nenhum produto foi recomendado para o usuário {user_id}.")
        return []
    else:
        print(f"  - Produtos já comprados pelo usuário {user_id}: {products_bought}")
        print(f"  - As 5 principais recomendações para o usuário {user_id} são: {final_recommendations}")
        return final_recommendations

# Testar a função com um usuário específico
test_user = 250
get_recommendations(test_user)
