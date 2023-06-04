import streamlit as st
import lightgbm as lgb
import pandas as pd
import shap
import plotly.express as px
import numpy as np
import joblib
import requests
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from zipfile import ZipFile
from sklearn.neighbors import NearestNeighbors




# URL de l'API Flask
api_url = "http://p7api.pythonanywhere.com/predict"

# Page setting
#st.set_page_config(layout="wide")
st.set_page_config(
    page_title="Home Credit Default Risk",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        #'Get Help': 'https://www.extremelycoolapp.com/help',
        #'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a Home Credit Default Risk. This is an *extremely* cool app!"
    }
)
o1,o2,o3 = st.columns(3)

o1.image("im-p7/logo4.png", caption="")
st.markdown("------------")
o2.title("Prêt à dépenser Dashboard")
st.markdown("------------")

# Charger les données
@st.cache_data
def load_data():
    
    z1 = ZipFile("data/X_data_test.zip")
    data = pd.read_csv(z1.open('X_data_test.csv'), encoding ='utf-8')
    general = pd.read_csv('data/general_info_test.csv')
    
    z2 = ZipFile("data/X_data.zip")
    training = pd.read_csv(z2.open('X_data.csv'), encoding ='utf-8')
    training_target=training[['TARGET']]
    # Renommer les caractéristiques
    #echantillon= data.copy()
    echantillon = data.sample(n=800)
    general = general[general['SK_ID_CURR'].isin(echantillon['SK_ID_CURR'])]

    echantillon.columns = ["".join([c if c.isalnum() else "_" for c in str(col)]) for col in echantillon.columns]
    labels = echantillon["SK_ID_CURR"]
    echantillon=echantillon.sort_values("SK_ID_CURR")
    labels=labels.sort_values()
    general=general.sort_values("SK_ID_CURR")
    echantillon = echantillon.reset_index(drop=True)
    general=general.reset_index(drop=True)
    labels=labels.reset_index(drop=True)
    X = echantillon.drop(["SK_ID_CURR"], axis=1)  # Supprimer l'identifiant client SK_ID_CURR
    #training = training.drop(["TARGET"], axis=1)
    return  X, labels,general,training



@st.cache_data
def k_means_clustering(X, k):
    # Effectuer le clustering K-means
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    # Ajouter les labels de clustering au DataFrame d'origine
    X_with_labels = X.copy()
    X_with_labels['Cluster_Labels'] = kmeans.labels_
    X_with_labels.index = X.index
    return X_with_labels


X,labels, general,training= load_data()

# Charger le modèle
model = joblib.load("data/LGBM_Classifier_final_v2.pkl")

# Créer l'explainer SHAP avec la classe TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Vérifier et remodeler les valeurs SHAP si nécessaire
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Sélectionner les valeurs SHAP de la classe positive

    
# Sélectionner une observation pour le diagramme local
st.sidebar.header("Profil du client :")
st.sidebar.markdown("------------")
selected_observation = st.sidebar.selectbox('ID', labels)
# Obtenir l'index de la valeur sélectionnée
selected_index = labels.loc[labels == selected_observation].index[0]



###########################


a1, a2, a3 = st.columns(3)
a1.metric("ID", selected_observation, "")
st.markdown("------------")


##########################

b1,b2 = st.columns(2)


#infos générales du client 
#show_client_info = st.sidebar.checkbox("Voulez-vous avoir plus d'informations sur le client ?")
# Afficher les informations sur le client si la case à cocher est activée
selectedrow = general.iloc[selected_index, :]

if (selectedrow["CODE_GENDER"]=='F'):
    st.sidebar.image("im-p7/F.png", caption="Female")
else:
    st.sidebar.image("im-p7/M.png", caption="Male")

st.sidebar.markdown("**Age :** {} ans".format(int(selectedrow["DAYS_BIRTH"] / -365)))
st.sidebar.markdown("**Situation Familiale :** {}".format(selectedrow["NAME_FAMILY_STATUS"]))
st.sidebar.markdown("**Nombre d'enfant :** {:.0f}".format(int(selectedrow["CNT_CHILDREN"])))   

# Préparer les données de la requête
payload = {"client_id": selected_observation}

# Envoyer la requête à l'API Flask
response = requests.get(api_url, params=payload)

# Vérifier le code de réponse HTTP
if response.status_code == 200:
    # Afficher la réponse de l'API Flask
    if response.text == '0':
        a3.markdown('<p style="font-size: 16px; color: green; font-weight: bold;"> Félicitations ! Demande acceptée.</p>', unsafe_allow_html=True)
        a2.image("im-p7/haut.png", caption="Bon score")
    elif response.text == '1':
        a3.markdown('<p style="font-size: 16px; color: red; font-weight: bold;"> Malheureusement ! Demande non acceptée .</p>', unsafe_allow_html=True)
        a2.image("im-p7/bas.png", caption="Mauvais score")
    elif response.text == '3':
        a3.markdown('<p style="font-size: 16px; color: orange; font-weight: bold;"> Demande acceptée .</p>', unsafe_allow_html=True)

        a2.image("im-p7/milieu.png", caption="Score fragile")

else:
    st.sidebar.write("Une erreur s'est produite lors de l'appel à l'API.")


# Bouton pour appeler l'API
if st.sidebar.button("Résultat"):
    if response.text == '0' or response.text == '3':
        st.balloons()

# Calculer l'importance moyenne des caractéristiques
st.markdown("------------")

###############################################################################################################
#échantillon 
st.write("<b>Information socio-économique de l'ensemble des clients </b>", unsafe_allow_html=True)
# Liste de couleurs personnalisées
f1,f2 = st.columns(2)

fig = px.sunburst(general, path=['CODE_GENDER', 'NAME_FAMILY_STATUS'], values='AMT_CREDIT', color='AMT_INCOME_TOTAL', hover_data=['CNT_CHILDREN'])

f1.plotly_chart(fig)



fig2 = px.density_heatmap(general, x="AMT_INCOME_TOTAL", y="AMT_CREDIT", marginal_x="rug", marginal_y="histogram")

# Changer les titres des axes
fig2.update_layout(
    xaxis_title="Revenu total",
    yaxis_title="Crédit demandé"
)

f2.plotly_chart(fig2)
######################################################
h1,h2 = st.columns(2)




fig3 = px.bar(general, x="NAME_FAMILY_STATUS", y="AMT_CREDIT", color="CODE_GENDER", barmode="group")
fig3.update_layout(
    xaxis_title="Situation familiale",
    yaxis_title="Crédit demandé"

)

h1.plotly_chart(fig3)


fig4 = px.bar(general, x="NAME_FAMILY_STATUS", y="AMT_INCOME_TOTAL", color="CODE_GENDER", barmode="group")
fig4.update_layout(
    xaxis_title="Situation familiale",
    yaxis_title="Revenu total"

)

h2.plotly_chart(fig4)


########################################################################################

# Calculer l'importance moyenne des caractéristiques
st.markdown("------------")





st.write("<b>Les caractéristiques importantes pour accepter une demande </b>", unsafe_allow_html=True)
value_max = b1.slider("Comnbien de caractéristiques souhaitez vous afficher ? ", min_value=1, max_value=20, step=1,value=15)

shap_abs_mean = abs(shap_values).mean(axis=0)
top_features_indices = shap_abs_mean.argsort()[-value_max:]  # Sélectionner les indices des 15 caractéristiques les plus importantes

# Récupérer les noms des caractéristiques correspondantes
top_feature_names = X.columns[top_features_indices]

# Récupérer les valeurs d'importance associées
top_feature_importance = shap_abs_mean[top_features_indices]

# Créer un dataframe pour les caractéristiques et leurs importances
shap_df = pd.DataFrame({'Caractéristique': top_feature_names, 'Importance': top_feature_importance})

# Trier les caractéristiques par importance décroissante
shap_df = shap_df.sort_values('Importance', ascending=False)

# Créer le diagramme global avec Plotly Express
fig_global = px.bar(
    shap_df,
    x='Importance',
    y='Caractéristique',
    orientation='h',
    color='Importance',
    color_continuous_scale='RdBu',
    title=f'Top {value_max} Caractéristiques décisives générales'
)

fig_global.update_layout(
    height=400,
    width=300,
    xaxis=dict(title='Importance'),
    yaxis=dict(title='Caractéristique'),
    coloraxis_colorbar=dict(title='Importance')
)
st.sidebar.markdown("------------")



#107285

# Vérifier si les valeurs SHAP pour l'observation sélectionnée existent
if isinstance(shap_values, np.ndarray):
    shap_values_single = shap_values[selected_index]

    # Sélectionner les 15 caractéristiques les plus importantes
    top_features_indices_local = np.argsort(abs(shap_values_single))[-15:]
    top_feature_names_local = X.columns[top_features_indices_local]
    top_feature_importance_local = shap_values_single[top_features_indices_local]

    # Créer le diagramme local avec Plotly Express
    fig_local = px.bar(
        x=top_feature_importance_local,
        y=top_feature_names_local,
        orientation='h',
        color=top_feature_importance_local,
        color_continuous_scale='RdBu',
        title=f'Importance des Caractéristiques pour le Client {selected_observation} (Local)'
    )

    fig_local.update_layout(
        height=400,
        width=300,
        xaxis=dict(title='Importance'),
        yaxis=dict(title='Caractéristique'),
        coloraxis_colorbar=dict(title='Importance')
    )

    # Afficher les deux graphes de shap dans la même colonne
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_global, use_container_width=True)
    col2.plotly_chart(fig_local, use_container_width=True)
    
#####################################################################################
# Calculer l'importance moyenne des caractéristiques
st.markdown("------------")


# Création de la checkbox
checkbox_state = st.checkbox("Souhaitez vous voir un nuage de points entre deux carcatéristiques ? ")

# Si la checkbox est cochée, afficher le code conditionnel
if checkbox_state:
    selected_columns = st.multiselect("Veuillez choisir deux caractéristiques", X.columns)
    # Scatter plot des deux colonnes sélectionnées
    scores = model.predict_proba(X)
    X['Score'] = scores[:, 1]  # Utilisez le score de la classe positive

    if len(selected_columns) == 2:
        scatter_fig = px.scatter(
            data_frame=X,
            x=selected_columns[0],
            y=selected_columns[1],
            #color=labels.apply(lambda x: "Client sélectionné" if x == selected_observation else "Autres clients"),
            #color_discrete_map={"Client sélectionné": "red", "Autres clients": "darkturquoise"},
            color='Score',  # Utilisez le score comme variable de couleur
            color_continuous_scale='RdBu',  # Choisissez une échelle de couleurs appropriée
            title="Scatter Plot des deux colonnes sélectionnées"
        )
        scatter_fig.update_layout(
            height=600,
            width=300,
            xaxis=dict(title=selected_columns[0]),
            yaxis=dict(title=selected_columns[1])
        )
        scatter_fig.update_traces(
            marker=dict(size=[35 if label == selected_observation else 10 for label in labels])
            #,color=['red' if label == selected_observation else 'darkturquoise' for label in labels])
        )
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.warning("Veuillez sélectionner exactement deux caracréristiques.")



################################################################################
# Calculer l'importance moyenne des caractéristiques
st.markdown("------------")

show_histogram = st.checkbox("Souhaitez vous voir la dictribution des clients les plus proches ")

if show_histogram:
    # préparer selectd observatio
    selected_column_neighbor = st.selectbox('Veillez choisir une caractéristique', X.columns)

    observation = X.iloc[selected_index,:]
    observation['SK_ID_CURR'] = selected_observation

    # Ajouter la nouvelle ligne au DataFrame existant
    training = training.sample(300)
    training = training.append(observation, ignore_index=True)
    training = training.fillna(training.median())

    # Appeler la fonction k_means_clustering pour obtenir les indices des plus proches voisins
    training['SK_ID_CURR'] = training['SK_ID_CURR'].astype(int)

    df_training = training.set_index("SK_ID_CURR")
    X_with_labels = k_means_clustering(df_training , 5)
    selected_cluster_label = X_with_labels.loc[selected_observation, 'Cluster_Labels']

    # Sélectionner toutes les valeurs du même cluster que l'index sélectionné
    same_cluster_indices = X_with_labels[X_with_labels['Cluster_Labels'] == selected_cluster_label].index
    same_cluster_data = X_with_labels.loc[same_cluster_indices]

    # Tracer la distribution de la colonne selected_column_neighbor à l'aide de Plotly Express
    fig_histogram = px.histogram(same_cluster_data, x=selected_column_neighbor, title='Distribution de la colonne {}'.format(selected_column_neighbor))

    # Obtenir la valeur de la colonne selected_column_neighbor pour l'index sélectionné
    selected_value = same_cluster_data.loc[selected_observation, selected_column_neighbor]

    # Ajouter une ligne verte pointillée à la figure pour représenter la valeur sélectionnée
    fig_histogram.add_shape(
        type="line",
        x0=selected_value,
        y0=0,
        x1=selected_value,
        y1=300,
        line=dict(color="red", width=2, dash="dash")
    )

    # Personnaliser les axes et le titre
    fig_histogram.update_layout(
        xaxis=dict(title="Valeur"),
        yaxis=dict(title="Compte"),
        title="Histogramme avec ligne représentant une valeur précise"
    )

    # Afficher l'histogramme
    st.plotly_chart(fig_histogram, use_container_width=True)
# Calculer l'importance moyenne des caractéristiques
st.markdown("------------")





