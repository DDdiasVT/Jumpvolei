import requests # <-- ADICIONE ESTA LINHA
# import streamlit as st
# import cv2
# ... (outras importações)
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

# --- 1. CONFIGURAÇÃO ---
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

# --- 2. SISTEMA DE EMAIL (Mantido para Boas-vindas) ---
def enviar_email_boas_vindas(nome_cliente, email_cliente):
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

# --- 3. BANCO DE DADOS ---
ARQUIVO_DB = "base_de_dados.csv"

def salvar_lead(nome, telefone, email, altura_user, peso_user):
    novo_dado = {
        "Data": [datetime.now().strftime("%d/%m/%Y %H:%M")],
        "Nome": [nome],
        "Telefone": [telefone],
        "Email": [email],
        "Altura": [altura_user],
        "Peso": [peso_user]
    }
    df_novo = pd.DataFrame(novo_dado)
    
    if not os.path.exists(ARQUIVO_DB):
        df_novo.to_csv(ARQUIVO_DB, index=False)
    else:
        df_novo.to_csv(ARQUIVO_DB, mode='a', header=False, index=False)

# --- MOTOR IA (Otimizado) ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)

def calcular_angulo(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def processar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Redimensionamento para evitar crash
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
        
        cv2.rectangle(image, (0, 0), (int(300*fator_escala), int(250*fator_escala)), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, frame, 0.3, 0, image)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            hip = [lms[23].x * largura_nova, lms[23].y * altura_nova]
            knee = [lms[25].x * largura_nova, lms[25].y * altura_nova]
            ankle = [lms[27].x * largura_nova, lms[27].y * altura_nova]
            angulo_joelho = calcular_angulo(hip, knee, ankle)
            pe_y = max(lms[31].y, lms[32].y) * altura_nova
            
            if estado == "CHAO":
                if frame_idx < 20: 
                    lista_y_chao.append(pe_y)
                else:
                    if chao_y == 0: chao_y = max(lista_y_chao)
                    if angulo_joelho < min_angulo_joelho: min_angulo_joelho = angulo_joelho
                    if angulo_joelho < 170 and frame_inicio_dip == 0: frame_inicio_dip = frame_idx
                    if pe_y < (chao_y - altura_nova * 0.03): 
                        estado = "NO AR"; frame_takeoff = frame_idx
                        if frame_inicio_dip > 0: tempo_contracao = (frame_takeoff - frame_inicio_dip) / fps

            elif estado == "NO AR":
                frames_no_ar += 1
                if frames_no_ar < 10: 
                    if angulo_joelho > max_extensao_joelho: max_extensao_joelho = angulo_joelho
                if pe_y >= (chao_y - altura_nova * 0.01):
                    estado = "POUSOU"; tempo_voo = frames_no_ar / fps
                    altura_final_cm = 122.6 * (tempo_voo * tempo_voo)

            font_scale = 0.8 * fator_escala
            cv2.putText(image, f"ALTURA: {altura_final_cm:.1f} cm", (20, int(60*fator_escala)), cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 255, 0), 2)
            cv2.putText(image, f"DIP: {int(min_angulo_joelho)} graus", (20, int(120*fator_escala)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            cv2.putText(image, f"RITMO: {tempo_contracao:.2f} s", (20, int(180*fator_escala)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 0), 2)
            
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        saida.write(image)
        frame_idx += 1
        if frame_idx % 30 == 0: gc.collect()
        
    cap.release(); saida.release(); barra.progress(100)
    
    stats = {"altura": altura_final_cm, "dip": min_angulo_joelho, "extensao": max_extensao_joelho, "tempo": tempo_contracao}
    return nome_saida, stats

# --- 4. LÓGICA DO APP (INTERFACE) ---

if 'cadastro_ok' not in st.session_state:
    st.session_state['cadastro_ok'] = False

col_a, col_b = st.columns([1, 5])
with col_a: st.write("# 🚀") 
with col_b: st.title("JumpPro Analytics")

if not st.session_state['cadastro_ok']:
    st.info("🔒 Cadastre-se para acessar a ferramenta gratuitamente.")
    
    with st.form("form_cadastro"):
        nome = st.text_input("Nome Completo")
        col1, col2 = st.columns(2)
        telefone = col1.text_input("WhatsApp (Importante para contato)")
        email = col2.text_input("E-mail")
        col3, col4 = st.columns(2)
        altura_user = col3.number_input("Altura (m)", 1.50, 2.30, 1.75)
        peso_user = col4.number_input("Peso (kg)", 40.0, 150.0, 70.0)
        
        submitted = st.form_submit_button("🚀 INICIAR ANÁLISE")
        
        if submitted:
            if nome and email and telefone:
                salvar_lead(nome, telefone, email, altura_user, peso_user)
                try:
                    enviar_email_boas_vindas(nome, email)
                except: pass
                st.session_state['cadastro_ok'] = True
                st.session_state['nome_user'] = nome
                st.rerun()
            else:
                st.error("Preencha todos os dados para continuarmos.")

else:
    st.write(f"Atleta: **{st.session_state['nome_user']}**")
    
    uploaded_file = st.file_uploader("Vídeo (Max 200MB)", type=["mp4", "mov"])

    if uploaded_file is not None:
        if uploaded_file.size > 200 * 1024 * 1024:
            st.error("Vídeo muito grande. Tente um menor que 200MB.")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            gc.collect()
            
            st.write("⏳ Otimizando e analisando vídeo...")
            
            try:
                video_saida_path, dados = processar_video(tfile.name)
                st.success("Análise Completa!")
                st.video(video_saida_path, format="video/webm")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Altura", f"{dados['altura']:.1f} cm")
                col2.metric("Dip", f"{int(dados['dip'])}°")
                col3.metric("Explosão", f"{int(dados['extensao'])}°")
                col4.metric("Ritmo", f"{dados['tempo']:.2f} s")
                
                st.divider()
                
                st.subheader("📋 Diagnóstico Automático")
                if dados['dip'] < 75: st.error(f"❌ Agachamento Excessivo ({int(dados['dip'])}°). Perda de energia elástica.")
                elif dados['dip'] > 110: st.warning(f"⚠️ Agachamento Curto ({int(dados['dip'])}°).")
                else: st.success(f"✅ Profundidade Ótima ({int(dados['dip'])}°).")
                
                st.info("ℹ️ Seus dados foram salvos. Nossa equipe entrará em contato via WhatsApp caso seja identificada uma oportunidade de melhoria no seu treino.")
                
                if st.button("Nova Análise"):
                    st.rerun()
                
            except Exception as e:
                st.error(f"Erro ao processar. Tente outro vídeo ou formato. ({e})")

with st.sidebar:
    st.divider()
    st.write("Admin")
    senha = st.text_input("Senha", type="password")
    if senha == "admin123":
        if os.path.exists(ARQUIVO_DB):
            with open(ARQUIVO_DB, "rb") as file:
                st.download_button("📥 Baixar Leads", file, "clientes.csv", "text/csv")
