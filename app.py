import requests 
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import gc 
# Importação da API do Gemini
from google import genai 
from google.genai.errors import APIError

# --- 1. CONFIGURAÇÃO GERAL ---
st.set_page_config(
    page_title="JumpPro Analytics",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# --- CONFIGURAÇÕES FIXAS ---
# URL do Google Sheets (Seus IDs estão aqui)
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScqve9FcZhMQkakXLGfnEiJzyKWAN8cLqaMCiLvRHez9NQYmg/formResponse"
ARQUIVO_DB = "base_de_dados.csv" # Apenas para download do Admin

# --- FUNÇÕES DE UTILIDADE E IA ---

def calcular_angulo(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def enviar_email_boas_vindas(nome_cliente, email_cliente):
    # Função para enviar e-mail (Mantida como estava)
    try:
        if "email" in st.secrets:
            usuario = st.secrets["email"]["usuario"]
            senha = st.secrets["email"]["senha"]
            
            msg = MIMEMultipart()
            msg['From'] = usuario
            msg['To'] = email_cliente
            msg['Subject'] = f"🚀 Análise Recebida - JumpPro"

            corpo_email = f"""
            <html>
              <body>
                <h2>Olá, {nome_cliente}!</h2>
                <p>Recebemos seu vídeo na <strong>JumpPro Analytics</strong>.</p>
                <p>Nossa IA já processou seus dados. Em breve um de nossos especialistas pode entrar em contato para discutir seus resultados.</p>
                <br>
                <p><em>Equipe JumpPro</em></p>
              </body>
            </html>
            """
            msg.attach(MIMEText(corpo_email, 'html'))

            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(usuario, senha)
            server.send_message(msg)
            server.quit()
            return True
        return False
    except Exception as e:
        print(f"Erro email: {e}")
        return False

def salvar_lead(dados_contato, dados_metricas):
    # Função para salvar no Google Sheets (Usa os 8 campos)
    
    dados_a_enviar = {
        # DADOS DO USUÁRIO
        "entry.1427267338": dados_contato['nome'],       
        "entry.597277093": dados_contato['email'],       
        "entry.1793364676": dados_contato['telefone'],   
        "entry.215882622": dados_contato['altura_user'], 

        # DADOS DA ANÁLISE (Métricas)
        "entry.1994800528": f"{dados_metricas['altura']:.1f}",   # SALTO
        "entry.1509204305": f"{dados_metricas['dip']:.0f}",      # DIP 
        "entry.1858263009": f"{dados_metricas['extensao']:.0f}", # EXTENSÃO 
        "entry.635471438": f"{dados_metricas['tempo']:.2f}",     # TEMPO CONTRACAO
    }

    try:
        response = requests.post(GOOGLE_FORM_URL, data=dados_a_enviar)
        if response.status_code == 200:
            return True
        else:
            print(f"ERRO DE ENVIO PARA O GOOGLE SHEETS: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erro ao enviar via requisição: {e}")
        return False

def gerar_plano_gemini(dados_contato, dados_metricas):
    # NOVO: Função para gerar o plano de treino usando a API
    
    if "gemini" not in st.secrets:
        return "Erro: Chave da API Gemini não configurada."

    api_key = st.secrets["gemini"]["api_key"]
    
    try:
        client = genai.Client(api_key=api_key)
        
        # Monta a string de dados
        dados_atleta_str = (
            f"Nome: {dados_contato['nome']}, "
            f"Altura User: {dados_contato['altura_user']:.2f}m, "
            f"Altura Calculada: {dados_metricas['altura']:.1f}cm, "
            f"Dip (graus): {dados_metricas['dip']:.0f}, "
            f"Extensão (graus): {dados_metricas['extensao']:.0f}, "
            f"Ritmo (s): {dados_metricas['tempo']:.2f}"
        )

        prompt_mestre = f"""
        ATUAÇÃO E IDENTIDADE: Você é um treinador de Força e Condicionamento (S&C Coach) de elite, especialista em Biomecânica de Salto Vertical (CMJ). Seu foco é em Pliometria e correção técnica.
        
        REGRAS DE INTERPRETAÇÃO:
        - DIP < 75º ou > 110º: Problema de Profundidade/Eficiência.
        - RITMO > 0.80s: Problema de Lentidão/Força.

        ESTRUTURA DA RESPOSTA:
        1. DIAGNÓSTICO FINAL: Classifique o atleta (Ex: Força-Dominante/Reativo).
        2. META: Projeção de ganho (Ex: +4.5cm).
        3. PLANO DE TREINO: Crie um plano de 4 semanas (3 treinos/semana: Potência, Reatividade, Técnica) para corrigir a falha principal. Use emojis e Markdown para formatar a resposta para o cliente.

        DADOS DO ATLETA PARA ANÁLISE: {dados_atleta_str}
        """

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_mestre
        )
        return response.text
        
    except APIError as e:
        return f"Erro na API Gemini: Falha de autenticação ou limite excedido. ({e})"
    except Exception as e:
        return f"Erro inesperado no Gemini: {e}"

# --- FUNÇÃO DE PROCESSAMENTO DE VÍDEO (Mantida) ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)

def processar_video(video_path):
    # ... (A função processar_video permanece inalterada) ...
    cap = cv2.VideoCapture(video_path)
    largura_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0
    
    fator_escala = 1.0
    if largura_orig > 640:
        fator_escala = 640 / largura_orig
    
    largura_nova = int(largura_orig * fator_escala)
    altura_nova = int(altura_orig * fator_escala)

    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    nome_saida = tfile_out.name
    
    saida = cv2.VideoWriter(nome_saida, cv2.VideoWriter_fourcc(*'vp80'), int(fps), (largura_nova, altura_nova))

    chao_y = 0; min_angulo_joelho = 180; max_extensao_joelho = 0; frames_no_ar = 0; estado = "CHAO"
    frame_inicio_dip = 0; frame_takeoff = 0; tempo_contracao = 0.0; altura_final_cm = 0.0
    lista_y_chao = []; frame_idx = 0; total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    barra = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        if fator_escala != 1.0: frame = cv2.resize(frame, (largura_nova, altura_nova))
        if total_frames > 0: barra.progress(min(frame_idx / total_frames, 1.0))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
