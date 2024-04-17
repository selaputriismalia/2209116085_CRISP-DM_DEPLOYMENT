import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
from sklearn.preprocessing import StandardScaler

st.title('Anime Rating Analysis & Recommender System')

url = "Data_Cleaned.csv"
df = pd.read_csv(url)

st.subheader("Dataset")
st.write(df.head())

st.subheader('Distribusi Rating Anime')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['rating'], bins=20, kde=True, color='blue', ax=ax)
ax.set_title('Distribusi Rating Anime')
ax.set_xlabel('Rating')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

rating_counts = df['rating'].value_counts().sort_index()
total_ratings = rating_counts.sum()

# Round the count values
rating_counts_rounded = rating_counts.round().astype(int)

# Calculate percentage of rating distribution
rating_percentages = (rating_counts / total_ratings) * 100

# Display rating distribution and rounded counts
st.write("Distribusi Rating:")
st.write(rating_counts_rounded)

st.write('Distribusi rating anime menunjukkan kecenderungan mayoritas anime menerima penilaian yang positif dari penonton, dengan puncak frekuensi terjadi pada kisaran rating antara 7.4 hingga 7.6. Hal ini mengindikasikan bahwa sebagian besar anime mendapat sambutan yang baik di kalangan komunitas penggemar. Namun, terdapat variasi dalam kualitas dan penerimaan anime, dengan beberapa anime mendapatkan rating sangat tinggi atau rendah.')
st.write("Produsen anime dapat menggunakan wawasan ini untuk memahami preferensi dan harapan penonton, serta untuk merencanakan produksi anime yang lebih sesuai dengan selera pasar. Dengan memperhatikan faktor-faktor yang berkontribusi terhadap rating yang tinggi, seperti genre, plot, karakter, dan kualitas animasi, produsen anime dapat mengoptimalkan strategi produksi mereka untuk menciptakan konten yang lebih menarik dan memuaskan bagi penonton. Selain itu, mereka juga dapat menggunakan informasi ini untuk merancang kampanye promosi yang lebih efektif dan menjangkau audiens yang lebih luas, sehingga meningkatkan kesadaran dan penerimaan terhadap anime baru yang akan mereka produksi.")

st.subheader('10 Genre Anime Populer')
genres = df.columns[1:-7]
genres = df.columns[1:-7]
genre_counts = df[genres].sum().sort_values(ascending=False)
top_genres = genre_counts.head(10)

# Buat plot
fig, ax = plt.subplots(figsize=(10, 6))
top_genres.plot(kind='bar', color='green', ax=ax)

# Atur judul dan label sumbu
ax.set_title('10 Genre Anime Terpopuler')
ax.set_xlabel('Genre')
ax.set_ylabel('Jumlah Anime')
ax.set_xticklabels(top_genres.index, rotation=45)

# Tampilkan plot menggunakan Streamlit
st.pyplot(fig)
total_anime = genre_counts.sum()
genre_percentages = genre_counts / total_anime * 100

# Display percentages outside the plot
st.write("Presentasi Anime berdasarkan genre:")
st.write(genre_percentages.apply(lambda x: '{:.2f}%'.format(x)))

st.write('Dari data persentase genre anime di atas, dapat diamati bahwa genre Comedy mendominasi sebagai genre paling populer dengan persentase sebesar 11.77%, diikuti oleh Action dengan 8.57%, dan Drama dengan 7.06%. Hal ini menunjukkan bahwa anime dengan elemen komedi cenderung lebih diminati oleh penonton dibandingkan dengan genre lainnya. Action dan Drama juga merupakan genre yang cukup diminati, menunjukkan minat yang besar dari penonton terhadap cerita dengan adegan aksi dan dramatis. Di sisi lain, genre seperti Shounen, Adventure, dan Romance juga memiliki persentase yang signifikan, menunjukkan popularitas yang konsisten di kalangan penggemar anime.')
st.write("Dari segi actionable insight, produsen anime dapat mempertimbangkan untuk mengembangkan lebih banyak konten dengan genre Comedy, Action, dan Drama untuk menarik minat penonton yang lebih luas. Namun, mereka juga tidak boleh mengabaikan genre lain yang memiliki persentase yang signifikan, seperti Adventure, Romance, dan Fantasy. Dengan memahami preferensi genre yang diminati oleh penonton, produsen anime dapat mengalokasikan sumber daya secara efektif untuk menghasilkan konten yang lebih sesuai dengan selera pasar, sehingga meningkatkan daya saing dan penerimaan di industri anime. Selain itu, mereka juga dapat menggunakan informasi ini untuk merancang strategi pemasaran yang lebih efektif dan mengoptimalkan promosi untuk mencapai target audiens dengan lebih baik.")


st.subheader("Jumlah Anime berdasarkan Media:")
media_types = ['Movie', 'Music', 'ONA', 'OVA', 'Special', 'TV']
media_counts = df[media_types].sum().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
media_counts.plot(kind='bar', color='purple', ax=ax)
ax.set_title('Jumlah Anime berdasarkan Media')
ax.set_xlabel('Tipe Media')
ax.set_ylabel('Jumlah Anime')
ax.set_xticklabels(media_counts.index, rotation=45)
st.pyplot(fig)

total_anime = len(df)


media_percentages = (media_counts / total_anime) * 100
st.write("Jumlah Anime berdasarkan Media:")
st.write(media_percentages)

st.write('Dari data persentase media tayang anime di atas, dapat diamati bahwa TV merupakan media tayang yang paling dominan dengan persentase sebesar 50.6%, diikuti oleh Movie dengan 21.3%, dan OVA dengan 14.6%. Hal ini menunjukkan bahwa format TV lebih populer dan lebih banyak digunakan untuk menayangkan anime dibandingkan dengan format lainnya. Movie juga memiliki persentase yang cukup signifikan, menunjukkan minat yang besar dari penonton terhadap anime dalam format film. OVA (Original Video Animation) dan Special juga memiliki persentase yang cukup tinggi, menunjukkan bahwa konten-konten khusus dan episodik juga diminati oleh penonton anime')

st.write('Untuk produsen anime, fokus pada produksi untuk TV bisa menjadi strategi yang baik karena mayoritas anime tayang di sana. Namun, mereka juga perlu mempertimbangkan untuk berinvestasi dalam format lain seperti Movie dan OVA yang memiliki pangsa pasar yang cukup besar. Dengan demikian, diversifikasi portofolio produksi mereka dapat membantu mereka menjangkau audiens yang lebih luas dan meningkatkan potensi pendapatan.')

st.subheader("Agglomerative Clustering")
def load_data(file_path):
    return pd.read_csv(file_path)

# Perform Agglomerative Clustering
def perform_clustering(data, n_clusters):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA(n_components=2)  # Reduksi dimensi menjadi 2
    reduced_data = pca.fit_transform(scaled_data)

    model = AgglomerativeClustering(n_clusters=n_clusters)
    clusters = model.fit_predict(reduced_data)

    return clusters, reduced_data

# Sidebar for selecting number of clusters
st.sidebar.header('Select Number of Clusters')
n_clusters = st.sidebar.slider('Jumlah Cluster', min_value=2, max_value=10, value=2, step=1)


st.sidebar.header('Pilihan Tampilan')
jumlah_tampilan = st.sidebar.selectbox('Jumlah Rating:', [5, 10, 15])


# Load data
file_path = 'Data_Cleaned.csv'
data = load_data(file_path)

# Perform Agglomerative Clustering
clusters, reduced_data = perform_clustering(data, n_clusters)

# Add cluster information to the DataFrame
data['Cluster'] = clusters

# Display the clustering results
st.write(f"Hasil Clusters dengan jumlah {n_clusters} Clusters:")
st.write(data)

# Visualize clustered data points
plt.figure(figsize=(10, 8))
for cluster_num in range(n_clusters):
    cluster_data = reduced_data[data['Cluster'] == cluster_num]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster_num}')

plt.title('Visualisasi Kluster')
plt.legend()

st.pyplot(plt.gcf())

st.subheader('Rating Tertinggi')
highest_rated_anime = df.nlargest(jumlah_tampilan, 'rating')

if not highest_rated_anime.empty:
    st.table(highest_rated_anime)
else:
    st.write('Tidak ada data untuk anime dengan rating tertinggi.')

st.subheader('Rating Terendah')
lowest_rated_anime = df.nsmallest(jumlah_tampilan, 'rating')

if not lowest_rated_anime.empty:
    st.table(lowest_rated_anime)
else:
    st.write('Tidak ada data untuk anime dengan rating terendah.')
