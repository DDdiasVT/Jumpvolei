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

# --- 2. SISTEMA DE EMAIL AUTOMÁTICO ---
def enviar_email_boas_vindas(nome_cliente, email_cliente):
    try:
        # Pega as credenciais do cofre (Secrets)
        usuario = st.secrets["email"]["usuario"]
        senha = st.secrets["email"]["senha"]
        
        msg = MIMEMultipart()
        msg['From'] = usuario
        msg['To'] = email_cliente
        msg['Subject'] = f"🚀 Bem-vindo à JumpPro, {nome_cliente}!"

        # O CORPO DO EMAIL (HTML Bonito)
        corpo_email = f"""
        <html>
          <body>
            <h2>Olá, {nome_cliente}!</h2>
            <p>Seu cadastro na <strong>JumpPro Analytics</strong> foi confirmado.</p>
            <p>Você já pode usar nossa ferramenta de Inteligência Artificial para analisar seus saltos quantas vezes quiser.</p>
            <br>
            <h3>🎁 O que acontece agora?</h3>
            <p>1. Faça o upload do seu vídeo no site.<br>
            2. Veja o diagnóstico da sua biomecânica na hora.<br>
            3. Se quiser ir além, nosso <strong>Plano de Treino Personalizado</strong> está disponível para compra.</p>
            <br>
            <p>Qualquer dúvida, responda este e-mail.</p>
            <p><em>Equipe JumpPro</em></p>
          </body>
        </html>
        """
        msg.attach(MIMEText(corpo_email, 'html'))

        # Conecta no Gmail e envia
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(usuario, senha)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Erro ao enviar email: {e}")
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

# --- 4. MOTOR IA (Mantido Igual) ---
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
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.webm')
    nome_saida = tfile_out.name
    saida = cv2.VideoWriter(nome_saida, cv2.VideoWriter_fourcc(*'vp80'), int(fps), (largura, altura))

    chao_y = 0; min_angulo_joelho = 180; max_extensao_joelho = 0; frames_no_ar = 0; estado = "CHAO"
    frame_inicio_dip = 0; frame_takeoff = 0; tempo_contracao = 0.0; altura_final_cm = 0.0
    lista_y_chao = []; frame_idx = 0; total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    barra = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if total_frames > 0: barra.progress(min(frame_idx / total_frames, 1.0))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.rectangle(image, (0, 0), (300, 250), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, frame, 0.3, 0, image)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            hip = [lms[23].x * largura, lms[23].y * altura]
            knee = [lms[25].x * largura, lms[25].y * altura]
            ankle = [lms[27].x * largura, lms[27].y * altura]
            angulo_joelho = calcular_angulo(hip, knee, ankle)
            pe_y = max(lms[31].y, lms[32].y) * altura
            
            if estado == "CHAO":
                if frame_idx < 20: 
                    lista_y_chao.append(pe_y)
                    cv2.putText(image, "CALIBRANDO...", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    if chao_y == 0: chao_y = max(lista_y_chao)
                    if angulo_joelho < min_angulo_joelho: min_angulo_joelho = angulo_joelho
                    if angulo_joelho < 170 and frame_inicio_dip == 0: frame_inicio_dip = frame_idx
                    if pe_y < (chao_y - altura * 0.03): 
                        estado = "NO AR"; frame_takeoff = frame_idx
                        if frame_inicio_dip > 0: tempo_contracao = (frame_takeoff - frame_inicio_dip) / fps

            elif estado == "NO AR":
                frames_no_ar += 1
                if frames_no_ar < 10: 
                    if angulo_joelho > max_extensao_joelho: max_extensao_joelho = angulo_joelho
                if pe_y >= (chao_y - altura * 0.01):
                    estado = "POUSOU"; tempo_voo = frames_no_ar / fps
                    altura_final_cm = 122.6 * (tempo_voo * tempo_voo)

            cv2.putText(image, f"ALTURA: {altura_final_cm:.1f} cm", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(image, f"DIP: {int(min_angulo_joelho)} graus", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f"RITMO: {tempo_contracao:.2f} s", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        saida.write(image)
        frame_idx += 1
        
    cap.release(); saida.release(); barra.progress(100)
    
    stats = {"altura": altura_final_cm, "dip": min_angulo_joelho, "extensao": max_extensao_joelho, "tempo": tempo_contracao}
    return nome_saida, stats

# --- 5. LÓGICA DO APP ---

if 'cadastro_ok' not in st.session_state:
    st.session_state['cadastro_ok'] = False

col_a, col_b = st.columns([1, 5])
with col_a: st.write("# 🚀") 
with col_b: st.title("JumpPro Analytics")

if not st.session_state['cadastro_ok']:
    st.info("🔒 Preencha seus dados para liberar a ferramenta.")
    
    with st.form("form_cadastro"):
        nome = st.text_input("Nome Completo")
        col1, col2 = st.columns(2)
        telefone = col1.text_input("WhatsApp")
        email = col2.text_input("E-mail")
        col3, col4 = st.columns(2)
        altura_user = col3.number_input("Altura (m)", 1.50, 2.30, 1.75)
        peso_user = col4.number_input("Peso (kg)", 40.0, 150.0, 70.0)
        
        submitted = st.form_submit_button("🚀 ACESSAR FERRAMENTA")
        
        if submitted:
            if nome and email:
                # 1. Salva no CSV
                salvar_lead(nome, telefone, email, altura_user, peso_user)
                
                # 2. Tenta enviar o e-mail (não trava se falhar)
                try:
                    enviar_email_boas_vindas(nome, email)
                    st.toast(f"E-mail de boas-vindas enviado para {email}!", icon="📧")
                except:
                    pass
                
                # 3. Libera o acesso
                st.session_state['cadastro_ok'] = True
                st.session_state['nome_user'] = nome
                st.rerun()
            else:
                st.error("Preencha Nome e E-mail.")

else:
    st.write(f"Atleta: **{st.session_state['nome_user']}**")
    
    uploaded_file = st.file_uploader("Vídeo do Salto (MP4)", type=["mp4", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.write("⏳ Analisando...")
        
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
            
            if dados['dip'] < 75: st.error(f"❌ Agachamento Excessivo ({int(dados['dip'])}°).")
            elif dados['dip'] > 110: st.warning(f"⚠️ Agachamento Curto ({int(dados['dip'])}°).")
            else: st.success(f"✅ Profundidade Ótima ({int(dados['dip'])}°).")
                 
            st.markdown("### 🔓 Desbloquear Plano de Treino")
            st.write("Receba a planilha exata para corrigir esses erros.")
            st.link_button("👉 QUERO MEU TREINO (R$ 19,90)", "https://www.mercadopago.com.br", type="primary")
            
            if st.button("Sair"):
                st.session_state['cadastro_ok'] = False
                st.rerun()
            
        except Exception as e:
            st.error(f"Erro: {e}")

with st.sidebar:
    st.divider()
    st.write("Admin")
    senha = st.text_input("Senha", type="password")
    if senha == "admin123":
        if os.path.exists(ARQUIVO_DB):
            with open(ARQUIVO_DB, "rb") as file:
                st.download_button("📥 Baixar Leads", file, "clientes.csv", "text/csv")
