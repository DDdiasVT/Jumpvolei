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

# --- 1. CONFIGURAÇÃO GERAL E VARIÁVEIS FIXAS ---
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

# URL do Google Sheets (SUA URL FINAL COM OS IDS CONFIRMADOS)
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLScqve9FcZhMQkakXLGfnEiJzyKWAN8cLqaMCiLvRHez9NQYmg/formResponse"
ARQUIVO_DB = "base_de_dados.csv" 

# --- 2. FUNÇÕES DE SERVIÇO (EMAIL, SAVE, IA) ---

def calcular_angulo(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

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

def salvar_lead(dados_contato, dados_metricas, plano_texto):
    
    dados_a_enviar = {
        # DADOS DO USUÁRIO (4 CAMPOS)
        "entry.1427267338": dados_contato['nome'],       
        "entry.597277093": dados_contato['email'],       
        "entry.1793364676": dados_contato['telefone'],   
        "entry.215882622": dados_contato['altura_user'], 

        # DADOS DA ANÁLISE (5 CAMPOS)
        "entry.1994800528": f"{dados_metricas['altura']:.1f}",   # SALTO
        "entry.1509204305": f"{dados_metricas['dip']:.0f}",      # DIP 
        "entry.1858263009": f"{dados_metricas['extensao']:.0f}", # EXTENSÃO 
        "entry.635471438": f"{dados_metricas['tempo']:.2f}",     # TEMPO CONTRACAO
        "entry.1582150062": plano_texto                          # PLANO COMPLETO (Texto)
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
    
    if "gemini" not in st.secrets:
        return "Erro: Chave da API Gemini não configurada."

    api_key = st.secrets["gemini"]["api_key"]
    
    try:
        client = genai.Client(api_key=api_key)
        
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

# --- FUNÇÃO DE PROCESSAMENTO DE VÍDEO ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7, model_complexity=1)

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
            lms = results.pose.landmarks.landmark
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
                # 1. Processa o vídeo e obtém as métricas
                video_saida_path, dados_metricas = processar_video(tfile.name)
                
                # 2. CHAMA O GEMINI (Gera o plano antes de salvar)
                dados_contato_session = st.session_state['dados_contato']
                with st.spinner("🧠 Gerando Plano de Treino Personalizado com IA..."):
                    plano_treino = gerar_plano_gemini(dados_contato_session, dados_metricas)
                
                # 3. SALVA O LEAD COMPLETO NO GOOGLE SHEETS
                if salvar_lead(dados_contato_session, dados_metricas, plano_treino):
                    st.success("✅ Análise Concluída e Dados Registrados no Sheets.")
                else:
                    st.error("⚠️ Análise Concluída, mas falhou ao salvar o lead no Sheets.")

                
                # 4. EXIBIÇÃO DE RESULTADOS
                st.video(video_saida_path, format="video/webm")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Altura", f"{dados_metricas['altura']:.1f} cm")
                col2.metric("Dip", f"{int(dados_metricas['dip'])}°")
                col3.metric("Explosão", f"{int(dados_metricas['extensao'])}°")
                col4.metric("Ritmo", f"{dados_metricas['tempo']:.2f} s")
                
                st.divider()
                
                st.subheader("📋 Plano de Ação (JumpPro Coach)")
                
                # --- LÓGICA DO PAYWALL (IMPLEMENTAÇÃO) ---
                separator = "3. PLANO DE TREINO:" 
                
                if plano_treino and separator in plano_treino:
                    free_content, paid_content_start = plano_treino.split(separator, 1)
                    
                    # 1. EXIBIR CONTEÚDO GRATUITO (Diagnóstico e Meta)
                    st.markdown(free_content)
                    
                    # 2. BARREIRA DE PAGAMENTO
                    st.divider()
                    st.subheader("🔒 Plano Detalhado de 30 Dias (Bloqueado)")
                    
                    st.info("O plano detalhado com séries, repetições e o calendário de 30 dias foi gerado e está pronto para ser enviado.")
                    
                    # Botão de Compra
                    st.link_button(
                        label="👉 ADQUIRIR PLANO COMPLETO (R$ 19,90)", 
                        url="https://link.mercadopago.com.br/SEU_LINK_AQUI", 
                        type="primary"
                    )
                    st.caption("Ao finalizar a compra, o plano será enviado para o seu e-mail.")
                    
                else:
                    # Fallback de erro se o Gemini não gerou o formato esperado
                    st.markdown(plano_treino) 
                    st.warning("Não foi possível formatar o paywall automaticamente. Se for o caso, aproveite o plano completo, mas sua contribuição ajuda a melhorar a IA!")
                    
                # --- FIM DO PAYWALL ---
                
                if st.button("Nova Análise"):
                    st.session_state['cadastro_ok'] = False
                    st.session_state['dados_contato'] = {}
                    st.rerun()
                
            except Exception as e:
                st.error(f"Erro ao processar: {e}")

with st.sidebar:
    st.divider()
    st.write("Admin")
    senha = st.text_input("Senha", type="password")
    if senha == "admin123":
        st.error("O download de CSV foi descontinuado. Acesse a lista de leads no Google Sheets.")
