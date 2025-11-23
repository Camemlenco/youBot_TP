import time
import math
import numpy as np
import cv2
from coppeliasim_zmqremoteapi_client import RemoteAPIClient

def main():
    print("Conectando ao CoppeliaSim...")
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    client.setStepping(True)
    sim.startSimulation()

    # --- SETUP DOS HANDLES ---
    arm_joints = [sim.getObject(f'/youBot/youBotArmJoint{i}') for i in range(5)]
    gripper_j1 = sim.getObject('/youBot/youBotGripperJoint1')
    gripper_j2 = sim.getObject('/youBot/youBotGripperJoint2')

    # Configurações de Controle
    current_joint_idx = 0
    joint_names = ["Base", "Ombro", "Cotovelo", "Punho 1", "Punho 2", "GARRA"]
    
    # Passo de movimento (Radianos para braço, Metros para garra)
    step_rad = 0.05 
    step_meter = 0.001

    print("Controle iniciado. Foque na janela 'PAINEL DE CONTROLE'.")

    try:
        while True:
            # 1. Cria o fundo preto para o painel (Altura 350, Largura 400, 3 canais de cor)
            # Usamos 3 canais (RGB) para poder usar cores
            board = np.zeros((350, 400, 3), dtype=np.uint8)

            # 2. Cabeçalho
            cv2.putText(board, "CONTROLE YOUBOT", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.line(board, (20, 40), (380, 40), (100, 100, 100), 1)

            # 3. Loop para desenhar cada junta na tela
            for i in range(6):
                # Ler valor atual
                if i < 5:
                    val = sim.getJointPosition(arm_joints[i])
                    unit = "rad"
                else:
                    val = sim.getJointPosition(gripper_j2)
                    unit = "m"

                # Define a cor: VERDE se selecionado, CINZA se não
                if i == current_joint_idx:
                    color = (0, 255, 0) # Verde (BGR)
                    thickness = 2
                    indicator = " >"
                else:
                    color = (200, 200, 200) # Cinza claro
                    thickness = 1
                    indicator = ""

                # Formata o texto: Ex: "J0 Base: 1.570 rad"
                text = f"J{i} {joint_names[i]}: {val:.4f} {unit}{indicator}"
                
                # Desenha na tela (posicão Y aumenta a cada item)
                y_pos = 80 + (i * 35)
                cv2.putText(board, text, (20, y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)

            # 4. Rodapé com instruções
            cv2.putText(board, "Teclas: [1-6] Selecionar | Setas: Mover", (20, 320), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)

            # 5. Mostra a imagem
            cv2.imshow("PAINEL DE CONTROLE", board)

            # 6. Captura Teclado
            key = cv2.waitKey(20) # Espera 20ms

            # Seleta de Juntas (Teclas 1-6)
            if key >= ord('1') and key <= ord('6'):
                current_joint_idx = key - ord('1')
            
            elif key == ord('q') or key == 27: # Q ou ESC
                break

            # Movimento (Setas ou WASD)
            move_dir = 0
            # Códigos de seta variam, mas no OpenCV geralmente: 82(Cima), 84(Baixo)
            if key == 82 or key == ord('w'): move_dir = 1
            elif key == 84 or key == ord('s'): move_dir = -1

            if move_dir != 0:
                if current_joint_idx < 5:
                    # Braço (Ângulos em Radianos)
                    h = arm_joints[current_joint_idx]
                    curr = sim.getJointPosition(h)
                    sim.setJointTargetPosition(h, curr + (move_dir * step_rad))
                else:
                    # Garra (Metros)
                    h = gripper_j2
                    curr = sim.getJointPosition(h)
                    # Limita entre 0 (fechado) e 0.025 (aberto)
                    target = max(0.0, min(0.025, curr + (move_dir * -step_meter)))
                    sim.setJointTargetPosition(gripper_j2, target)
                    sim.setJointTargetPosition(gripper_j1, target * -0.5)

            # Passo da simulação
            client.step()

    except Exception as e:
        print(f"Erro: {e}")
    finally:
        cv2.destroyAllWindows()
        sim.stopSimulation()
        print("Fim.")

if __name__ == "__main__":
    main()