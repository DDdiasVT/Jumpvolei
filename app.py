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

# --- 2. SISTEMA DE EMAIL ---
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

# --- 3. BANCO DE DADOS (GOOGLE SHEETS) ---
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScqve9FcZhMQkakXLGfnEiJzyKWAN8cLqaMCiLvRHez9NQYmg/formResponse"

def salvar_lead(dados_contato, dados_metricas): # Recebe Contato + Métricas
    
    # Mapeamento FINAL com os IDs fornecidos (8 campos)
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
            print("Dados enviados com sucesso para o Google Sheets!")
            return True
        else:
            print(f"ERRO DE ENVIO PARA O GOOGLE SHEETS: {response.status_code}")
            return False
    except Exception as e:
        print(f"Erro ao enviar via requisição: {e}")
        return False

# --- 4. DIAGNÓSTICO COMPLEXO (NOVO) ---
def gerar_diagnostico_complexo(dados):
    dip = dados['dip']
    extensao = dados['extensao']
    ritmo = dados['tempo']
    
    perfil = ""
    solucao = ""
    
    # 1. ANÁLISE DO DIP (Profundidade)
    if dip < 75:
        perfil += f"❌ **Dominância de Força (Deep Squat):** O ângulo de agachamento de **{dip:.0f}º** está excessivamente baixo. Isso aumenta o tempo de contato e 'mata' o efeito mola (SSC)."
    elif dip > 110:
        perfil += f"⚠️ **Agachamento Curto:** Com **{dip:.0f}º**, você não está utilizando todo o potencial de alongamento do tendão."
    else:
        perfil += f"✅ **Profundidade Ótima:** O agachamento de **{dip:.0f}º** está na faixa ideal (90-100º). Base sólida."

    # 2. ANÁLISE DO RITMO (Velocidade)
    if ritmo > 0.85:
        perfil += f"\n\n❌ **Lentidão Crítica:** O Tempo de Contração de **{ritmo:.2f}s** é muito elevado (alvo ideal < 0.5s). Isso reforça o perfil 'Força-Dominante'."
        solucao = "Seu elo mais fraco é a **Reatividade**. O treino deve focar em força explosiva e diminuir a profundidade do agachamento (Drill work)."
    else:
        perfil += f"\n\n✅ **Ritmo Forte:** O Tempo de Contração de **{ritmo:.2f}s** demonstra boa capacidade de conversão de força em velocidade."
        solucao = "Seu foco deve ser **ganho de força pura** para aumentar o teto da altura."
    
    # 3. ANÁLISE DA EXTENSÃO (O 'Finish')
    if extensao < 165:
        solucao += "\n\n**Atenção:** Falta de Tripla Extensão. Você não está finalizando o salto com a máxima extensão possível, perdendo potência no momento da decolagem."
    
    st.markdown(f"**Análise de Perfil Biomecânico:**\n{perfil}", unsafe_allow_html=True)
    st.info(f"**Conclusão da IA:** {solucao}")


# --- 5. MOTOR IA (Manutenção) ---
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

# --- 6. LÓGICA DO APP (INTERFACE) ---

if 'cadastro_ok' not in st.session_state:
    st.session_state['cadastro_ok'] = False
if 'dados_contato' not in st.session_state:
    st.session_state['dados_contato'] = {}

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
        
        altura_user = st.number_input("Sua Altura (m)", 1.50, 2.30, 1.75) 
        
        submitted = st.form_submit_button("🚀 INICIAR ANÁLISE")
        
        if submitted:
            if nome and email and telefone:
                # 1. SALVA OS DADOS DE CONTATO NA MEMÓRIA
                st.session_state['dados_contato'] = {
                    'nome': nome,
                    'telefone': telefone,
                    'email': email,
                    'altura_user': altura_user
                }
                
                # 2. Envia e-mail de boas-vindas
                enviar_email_boas_vindas(nome, email)
                
                # 3. Continua o fluxo
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
                
                # CHAMA A FUNÇÃO E ENVIA TUDO PARA O GOOGLE SHEETS
                if salvar_lead(st.session_state['dados_contato'], dados):
                    st.success("✅ Análise Completa! Dados salvos no Sheets.")
                else:
                    st.error("⚠️ Análise Completa, mas falhou ao salvar o lead no Sheets.")
                
                st.video(video_saida_path, format="video/webm")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Altura", f"{dados['altura']:.1f} cm")
                col2.metric("Dip", f"{int(dados['dip'])}°")
                col3.metric("Explosão", f"{int(dados['extensao'])}°")
                col4.metric("Ritmo", f"{dados['tempo']:.2f} s")
                
                st.divider()
                
                st.subheader("📋 Diagnóstico de Potência JumpPro")
                # NOVO DIAGNÓSTICO COMPLEXO
                gerar_diagnostico_complexo(dados)
                
                st.info("ℹ️ Sua análise foi registrada. Nossa equipe de especialistas entrará em contato via WhatsApp caso seja identificada uma oportunidade de melhoria no seu treino.")
                
                if st.button("Nova Análise"):
                    st.rerun()
                
            except Exception as e:
                st.error(f"Erro ao processar. Tente outro vídeo ou formato. ({e})")

# O bloco da barra lateral de Admin (sem funcionalidade de download, pois o DB não existe mais)
with st.sidebar:
    st.divider()
    st.write("Admin")
    # A variável ARQUIVO_DB não é mais definida, mas o código de login ainda funciona
    senha = st.text_input("Senha", type="password")
    if senha == "admin123":
        st.success("Acesso Admin liberado, mas o banco de dados CSV local foi descontinuado.")
        st.write("Acompanhe os leads diretamente no Google Sheets.")
