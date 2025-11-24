# --- CÓDIGO V9: PAINEL DE CONTROLE COMPLETO (Vídeo + Relatório) ---
%%capture
!pip uninstall -y mediapipe
!pip install mediapipe==0.10.14 opencv-python-headless

import cv2
import mediapipe as mp
import numpy as np
import os
from google.colab import files

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=2)

def calcular_angulo(a, b, c):
    a = np.array(a); b = np.array(b); c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    if angle > 180.0: angle = 360-angle
    return angle

def analisar_hud_completo(nome_arquivo):
    if not os.path.exists(nome_arquivo):
        print(f"ERRO: Arquivo '{nome_arquivo}' não encontrado.")
        return None

    cap = cv2.VideoCapture(nome_arquivo)
    largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps): fps = 30.0

    nome_saida = 'painel_completo.mp4'
    saida = cv2.VideoWriter(nome_saida, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (largura, altura))
    
    print(f"Gerando vídeo com HUD Completo...")
    
    # Variáveis
    chao_y = 0
    min_angulo_joelho = 180 
    max_extensao_joelho = 0 
    frames_no_ar = 0
    estado = "CHAO"
    
    # Métricas Temporais
    frame_inicio_dip = 0     
    frame_takeoff = 0        
    tempo_contracao = 0.0
    altura_final_cm = 0.0
    
    lista_y_chao = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Cria um Painel Preto na esquerda para os dados
        cv2.rectangle(image, (0, 0), (300, 300), (0, 0, 0), -1)
        cv2.addWeighted(image, 0.8, frame, 0.2, 0, image) # Deixa meio transparente

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            
            # Coordenadas
            hip = [lms[23].x * largura, lms[23].y * altura]
            knee = [lms[25].x * largura, lms[25].y * altura]
            ankle = [lms[27].x * largura, lms[27].y * altura]
            
            angulo_joelho = calcular_angulo(hip, knee, ankle)
            
            # Ponta do Pé (31/32)
            pe_y = max(lms[31].y, lms[32].y) * altura
            
            # --- LÓGICA ---
            if estado == "CHAO":
                # Calibração
                if frame_idx < 30:
                    lista_y_chao.append(pe_y)
                    cv2.putText(image, "CALIBRANDO...", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    if chao_y == 0: chao_y = max(lista_y_chao)
                    
                    # Rastreia o ângulo mínimo (Agachamento máximo)
                    if angulo_joelho < min_angulo_joelho:
                        min_angulo_joelho = angulo_joelho
                    
                    # Detecta INÍCIO do movimento (Joelhos dobrando)
                    if angulo_joelho < 170 and frame_inicio_dip == 0:
                        frame_inicio_dip = frame_idx
                    
                    # Detecta SAÍDA
                    if pe_y < (chao_y - altura * 0.03): 
                        estado = "NO AR"
                        frame_takeoff = frame_idx
                        # Calcula tempo de contração
                        if frame_inicio_dip > 0:
                            tempo_contracao = (frame_takeoff - frame_inicio_dip) / fps

            elif estado == "NO AR":
                frames_no_ar += 1
                
                # Rastreia extensão máxima (Explosão)
                if frames_no_ar < 10: 
                    if angulo_joelho > max_extensao_joelho: max_extensao_joelho = angulo_joelho
                
                # Detecta POUSO
                if pe_y >= (chao_y - altura * 0.01):
                    estado = "POUSOU"
                    tempo_voo = frames_no_ar / fps
                    altura_final_cm = 122.6 * (tempo_voo * tempo_voo)

            # --- HUD (DESENHO NA TELA) ---
            # 1. Altura
            cv2.putText(image, "ALTURA:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            val_altura = f"{altura_final_cm:.1f} cm" if altura_final_cm > 0 else "--"
            cv2.putText(image, val_altura, (20, 80), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 0), 2)
            
            # 2. Dip (Agachamento)
            cv2.putText(image, "AGACHAMENTO (Dip):", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            val_dip = f"{int(min_angulo_joelho)} graus" if min_angulo_joelho < 179 else "--"
            cor_dip = (0, 255, 255) if 80 <= min_angulo_joelho <= 110 else (0, 0, 255)
            cv2.putText(image, val_dip, (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_dip, 2)
            
            # 3. Explosão
            cv2.putText(image, "EXPLOSAO (Ext):", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            val_ext = f"{int(max_extensao_joelho)} graus" if max_extensao_joelho > 0 else "--"
            cor_ext = (0, 255, 0) if max_extensao_joelho > 165 else (0, 0, 255)
            cv2.putText(image, val_ext, (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, cor_ext, 2)
            
            # 4. Tempo de Contração (Velocidade)
            if tempo_contracao > 0:
                cv2.putText(image, f"RITMO: {tempo_contracao:.2f}s", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
        saida.write(image)
        frame_idx += 1
        
    cap.release()
    saida.release()
    
    # Relatório TEXTO também (para garantir)
    print("\n" + "="*40)
    print("DADOS FINAIS:")
    print(f"Altura: {altura_final_cm:.1f} cm")
    print(f"Agachamento: {min_angulo_joelho:.0f} graus")
    print(f"Explosão: {max_extensao_joelho:.0f} graus")
    print(f"Tempo de Contração: {tempo_contracao:.2f} s")
    print("="*40)
    
    return nome_saida

arquivo = analisar_hud_completo('teste.mp4')
if arquivo:
    files.download(arquivo)
