import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import urllib.request
import urllib.parse
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# --- 1. Definición del Esquema Pydantic para el Agente Inteligente ---
class SongInfo(BaseModel):
    mood: str = Field(description="Estado de ánimo de la canción. Debe ser uno de: Empowering, Cool, Energizing, Yearning, Defiant, Excited, Sensual, Urgent, Sophisticated, Rowdy, Brooding, Upbeat, Aggressive, Melancholy, Gritty, Romantic, Sentimental, Fiery, Easygoing, Peaceful, Stirring, Lively, Other.")
    tempo: str = Field(description="Tempo de la canción. Debe ser uno de: Medium Tempo, Slow Tempo, Fast Tempo.")
    genre: str = Field(description="Género musical. Debe ser uno de: Urban, Pop, Traditional, Alternative & Punk, Electronica, Rock, Other, Soundtrack, Jazz.")
    artist_type: str = Field(description="Tipo de artista. Debe ser uno de: Female, Male, Mixed.")
    chart_date: int = Field(description="Fecha estimada de lanzamiento o entrada a listas en formato AAAAMMDD (ej. 20140515).")
    durationSeg: float = Field(description="Duración aproximada de la canción en segundos.")
    anioNacimiento: float = Field(description="Año de nacimiento del artista principal o año de formación de la banda (ej. 1985.0).")

# --- 2. Función para obtener datos usando Gemini ---
def obtener_info_cancion(cancion: str, artista: str, api_key: str) -> SongInfo:
    client = genai.Client(api_key=api_key)
    prompt = f"""
    Busca información real o estima de manera precisa los metadatos para la canción '{cancion}' del artista '{artista}'.
    Devuelve los datos estructurados que se solicitan en el esquema.
    """
    response = client.models.generate_content(
        model='gemini-3.5-flash-lite',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SongInfo,
            temperature=0.1
        ),
    )
    return SongInfo.model_validate_json(response.text)

# --- 3. Función para buscar una vista previa de audio e imagen de portada vía iTunes API ---
def buscar_demo_itunes(cancion: str, artista: str):
    try:
        query = f"{cancion} {artista}"
        url = f"https://itunes.apple.com/search?term={urllib.parse.quote(query)}&limit=1&entity=musicTrack"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode('utf-8'))
        if data['resultCount'] > 0:
            track = data['results'][0]
            return track.get('previewUrl'), track.get('artworkUrl100')
    except Exception as e:
        pass
    return None, None

# --- 4. Interfaz de la Aplicación Streamlit ---
st.set_page_config(page_title="Predicción Billboard Top 1", page_icon="🎵", layout="centered")

st.title("🎵 Predicción de Éxito: Billboard Top 1")
st.write("Ingresa una canción y un artista. Nuestro agente inteligente de IA recopilará su información y el modelo RandomForest entrenado predecirá si logrará ser Top 1 en Billboard.")

# Manejo Seguro de la Clave API (Primero busca en st.secrets, luego en variable de entorno, y finalmente da opción manual)
gemini_key = st.secrets.get("GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY", ""))

col1, col2 = st.columns(2)
with col1:
    cancion_input = st.text_input("Nombre de la canción:", placeholder="Ej. Royals")
with col2:
    artista_input = st.text_input("Artista / Grupo:", placeholder="Ej. Lorde")

if st.button("🔮 Analizar y Predecir Éxito"):
    if not cancion_input or not artista_input:
        st.warning("Por favor introduce tanto el nombre de la canción como el del artista.")
    elif not gemini_key:
        st.error("Se requiere una GEMINI_API_KEY configurada para que la Inteligencia Artificial pueda buscar la información de la canción.")
    else:
        with st.spinner("🤖 Agente inteligente buscando información de la canción con Gemini..."):
            try:
                # 1. Obtener metadatos estructurados con Gemini
                info = obtener_info_cancion(cancion_input, artista_input, gemini_key)

                # 2. Buscar audio de vista previa e imagen de portada
                demo_url, cover_url = buscar_demo_itunes(cancion_input, artista_input)

                st.success("¡Información de la canción recopilada exitosamente!")

                # Presentación visual
                col_img, col_txt = st.columns([1, 2])
                with col_img:
                    if cover_url:
                        st.image(cover_url, width=150)
                    else:
                        st.subheader("💿")
                with col_txt:
                    st.subheader(f"{cancion_input}")
                    st.write(f"**Artista:** {artista_input}")

                # Tabla con la ficha técnica provista por la IA
                st.markdown("### 📋 Ficha Técnica (Datos recuperados)")
                df_ficha = pd.DataFrame({
                    "Atributo": ["Estado de Ánimo", "Tempo", "Género Musical", "Tipo de Artista", "Fecha Estimada", "Duración (segundos)", "Año de Nacimiento/Origen"],
                    "Valor obtenido por IA": [info.mood, info.tempo, info.genre, info.artist_type, info.chart_date, f"{info.durationSeg} s", int(info.anioNacimiento)]
                })
                st.table(df_ficha)

                # Reproductor del audio preview de iTunes
                if demo_url:
                    st.markdown("### 🎧 Escuchar Demo de la Canción")
                    st.audio(demo_url)
                else:
                    st.info("No se encontró un preview de audio disponible para reproducción directa.")

                # 3. Preparación de datos y predicción con RandomForest
                with st.spinner("🧠 Procesando predicción en el modelo entrenado..."):
                    # Carga del modelo especificado por el usuario
                    modelo = joblib.load('modelo_random_forest.joblib')
                
                    # Estructura del dataset original tras One-Hot Encoding
                    columnas_entrenamiento = [
                        'chart_date', 'durationSeg', 'anioNacimiento',
                        'mood_Aggressive', 'mood_Brooding', 'mood_Cool', 'mood_Defiant',
                        'mood_Easygoing', 'mood_Empowering', 'mood_Energizing', 'mood_Excited',
                        'mood_Fiery', 'mood_Gritty', 'mood_Lively', 'mood_Melancholy',
                        'mood_Other', 'mood_Peaceful', 'mood_Romantic', 'mood_Rowdy',
                        'mood_Sensual', 'mood_Sentimental', 'mood_Sophisticated',
                        'mood_Stirring', 'mood_Upbeat', 'mood_Urgent', 'mood_Yearning',
                        'tempo_Fast Tempo', 'tempo_Medium Tempo', 'tempo_Slow Tempo',
                        'genre_Alternative & Punk', 'genre_Electronica', 'genre_Jazz',
                        'genre_Other', 'genre_Pop', 'genre_Rock', 'genre_Soundtrack',
                        'genre_Traditional', 'genre_Urban', 'artist_type_Female',
                        'artist_type_Male', 'artist_type_Mixed'
                    ]

                    # Crear vector de predicción lleno de ceros
                    X_pred = pd.DataFrame(0, index=[0], columns=columnas_entrenamiento)
                    X_pred['chart_date'] = info.chart_date
                    X_pred['durationSeg'] = info.durationSeg
                    X_pred['anioNacimiento'] = info.anioNacimiento

                    # Activación de dummies según la respuesta de la IA
                    cat_mood = f"mood_{info.mood.strip()}"
                    cat_tempo = f"tempo_{info.tempo.strip()}"
                    cat_genre = f"genre_{info.genre.strip()}"
                    cat_artist = f"artist_type_{info.artist_type.strip()}"

                    if cat_mood in X_pred.columns: X_pred[cat_mood] = 1
                    if cat_tempo in X_pred.columns: X_pred[cat_tempo] = 1
                    if cat_genre in X_pred.columns: X_pred[cat_genre] = 1
                    if cat_artist in X_pred.columns: X_pred[cat_artist] = 1

                    # Predicción final
                    prediccion = modelo.predict(X_pred)[0]
                    probabilidades = modelo.predict_proba(X_pred)[0]

                    st.markdown("--- ")
                    st.subheader("🔮 Resultado del Pronóstico")
                    if prediccion == 1:
                        st.balloons()
                        st.success(f"🔥 **¡Felicidades!** El modelo predice que la canción **ENTRARÁ al Top 1 de Billboard** con un **{probabilidades[1]:.2%}** de probabilidad.")
                    else:
                        st.error(f"📉 El modelo predice que la canción **NO logrará entrar al Top 1** de Billboard (Probabilidad de fracaso: **{probabilidades[0]:.2%}**).")

            except Exception as error:
                st.error(f"Ocurrió un error al procesar la predicción: {error}")
