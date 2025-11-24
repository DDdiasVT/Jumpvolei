import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os

# --- 1. CONFIGURAÇÃO VISUAL ---
st.set_page_config(
    page_title="JumpPro Analytics",
    page_icon="🏆",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Estilo CSS para esconder menus e deixar limpo
st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# Cabeçalho
col_a, col_b = st.columns([1, 5])
with col_a:
    st.write("# 🚀") 
with col_b:
    st.title("JumpPro Analytics")
    st.caption("Análise Biomecânica de Elite.")

# --- 2. CONFIGURAÇÃO DA IA ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Confiança 0.7 para evitar tremedeira, Complexity 1 para rodar na nuvem
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

    # Cria arquivo temporário
    tfile_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    nome_saida = tfile_out.name
    
    # CODEC DE SEGURANÇA: 'mp4v' evita o erro de tela cinza na maioria dos navegadores PC
    saida = cv2.VideoWriter(nome_saida, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (largura, altura))

    # Variáveis
    chao_y = 0
    min_angulo_joelho = 180 
    max_extensao_joelho = 0 
    frames_no_ar = 0
    estado = "CHAO"
    
    frame_inicio_dip = 0     
    frame_takeoff = 0        
    tempo_contracao = 0.0
    altura_final_cm = 0.0
    
    lista_y_chao = []
    frame_idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    barra = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Atualiza barra
        if total_frames > 0:
            barra.progress(min(frame_idx / total_frames, 1.0))

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # HUD Escuro
        cv2.rectangle(image, (0, 0), (300, 250), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.7, frame, 0.3, 0, image)

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            
            # Coordenadas
            hip = [lms[23].x * largura, lms[23].y * altura]
            knee = [lms[25].x * largura, lms[25].y * altura]
            ankle = [lms[27].x * largura, lms[27].y * altura]
            angulo_joelho = calcular_angulo(hip, knee, ankle)
            
            pe_y = max(lms[31].y, lms[32].y) * altura
            
            # Lógica
            if estado == "CHAO":
                if frame_idx < 20: 
                    lista_y_chao.append(pe_y)
                    cv2.putText(image, "CALIBRANDO...", (20, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    if chao_y == 0: chao_y = max(lista_y_chao)
                    
                    if angulo_joelho < min_angulo_joelho: min_angulo_joelho = angulo_joelho
                    if angulo_joelho < 170 and frame_inicio_dip == 0: frame_inicio_dip = frame_idx
                    
                    if pe_y < (chao_y - altura * 0.03): 
                        estado = "NO AR"
                        frame_takeoff = frame_idx
                        if frame_inicio_dip > 0:
                            tempo_contracao = (frame_takeoff - frame_inicio_dip) / fps

            elif estado == "NO AR":
                frames_no_ar += 1
                if frames_no_ar < 10: 
                    if angulo_joelho > max_extensao_joelho: max_extensao_joelho = angulo_joelho
                
                if pe_y >= (chao_y - altura * 0.01):
                    estado = "POUSOU"
                    tempo_voo = frames_no_ar / fps
                    altura_final_cm = 122.6 * (tempo_voo * tempo_voo)

            # Escreve Dados no Vídeo
            cv2.putText(image, f"ALTURA: {altura_final_cm:.1f} cm", (20, 60), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(image, f"DIP: {int(min_angulo_joelho)} graus", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(image, f"RITMO: {tempo_contracao:.2f} s", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        saida.write(image)
        frame_idx += 1
        
    cap.release()
    saida.release()
    barra.progress(100)
    
    stats = {
        "altura": altura_final_cm,
        "dip": min_angulo_joelho,
        "extensao": max_extensao_joelho,
        "tempo": tempo_contracao
    }
    return nome_saida, stats

# --- 3. INTERFACE DE USUÁRIO ---
uploaded_file = st.file_uploader("Carregue seu vídeo de Salto (MP4)", type=["mp4", "mov"])

if uploaded_file is not None:
    # Salva arquivo temporário
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    st.write("⏳ Analisando biomecânica...")
    
    try:
        # Processa
        video_saida_path, dados = processar_video(tfile.name)
        
        # --- AQUI ESTAVA O PROBLEMA: EXIBIR O VÍDEO ANTES DE TUDO ---
        st.success("Análise Completa!")
        
        # Exibe o vídeo processado
        st.video(video_saida_path)
        
        # Exibe as Métricas em Cartões
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Altura", f"{dados['altura']:.1f} cm")
        col2.metric("Dip", f"{int(dados['dip'])}°")
        col3.metric("Explosão", f"{int(dados['extensao'])}°")
        col4.metric("Ritmo", f"{dados['tempo']:.2f} s")
        
        st.divider()
        
        # --- 4. ÁREA DE VENDA (DIAGNÓSTICO) ---
        st.subheader("📋 Diagnóstico do Treinador")
        
        # Lógica Simples de Feedback
        if dados['dip'] < 75:
            st.error(f"❌ **Problema Crítico:** Agachamento Excessivo ({int(dados['dip'])}°). Você desce demais e perde potência.")
        elif dados['dip'] > 110:
             st.warning(f"⚠️ **Atenção:** Agachamento Curto ({int(dados['dip'])}°). Falta amplitude.")
        else:
             st.success(f"✅ **Ótimo:** Profundidade ideal ({int(dados['dip'])}°).")
             
        if dados['tempo'] > 0.85:
            st.error(f"❌ **Lentidão:** Ritmo de {dados['tempo']:.2f}s é muito lento para pliometria.")
        else:
            st.success(f"✅ **Velocidade:** Ritmo de {dados['tempo']:.2f}s está bom.")

        st.info("💡 Com base nesses números, a IA identificou que você pode ganhar até **5cm** corrigindo apenas o ritmo.")
        
        st.markdown("### 🔓 Desbloquear Plano de Treino")
        st.write("Receba a planilha exata de exercícios para corrigir esse 'Dip' e acelerar seu salto.")
        
        st.link_button("👉 QUERO MEU TREINO (R$ 19,90)", "https://www.mercadopago.com.br", type="primary")
        
    except Exception as e:
        st.error(f"Erro ao processar: {e}")
